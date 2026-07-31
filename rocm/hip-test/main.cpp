/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: hip-test/main.cpp
 *
 * License: MIT
 */

// Tiny program to test ROCm status
// #define __HIP_DISABLE_CPP_FUNCTIONS__
#include <cmath>
// #include <iostream>
#include <sstream>
#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>

#include "shark/shark.hpp"

namespace shark
{
std::integral_constant<float, 2e-5F> VERSION;


__global__ void same_matrix(const float* a, const float* b, int rows, int cols, int* is_diff)
{
	__shared__ int block_diff;
	
	if (threadIdx.x == 0 && threadIdx.y == 0)
		block_diff = 0;
	
	__syncthreads();
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows && col < cols) {
		float va = a[row * cols + col];
		float vb = b[row * cols + col];
		
		float diff = fabsf(va - vb);
		float tolerance = 1e-5F * fmaxf(1.0f, fmaxf(fabsf(va), fabsf(vb)));
		
		if (diff > tolerance) {
			// block_diff = 1;
			atomicAdd(is_diff, 1);
		}
	}
	/*
	__syncthreads();
	
	if (threadIdx.x == 0 && threadIdx.y == 0 && block_diff == 1)
		*is_diff = 1;
	*/
}

// template<size_t>
__global__ void permute_kernel(
	float* out, const float* in,
	const size_t* in_shape, const size_t* perm, const size_t* in_stride,
	size_t mat_dim, size_t n_elements)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_elements) {
		return;
	}
	
	size_t tmp = idx;
	size_t out_coord[16] = { }; // it should be very sufficient so far
	for (size_t i = mat_dim; i-- > 0; ) {
		out_coord[i] = tmp % in_shape[perm[i]];
		tmp /= in_shape[perm[i]];
	}
	
	size_t in_coord[16] = { };
	for (size_t i = mat_dim; i-- > 0; ) {
		in_coord[perm[i]] = out_coord[i];
	}
	
	// input coordinates -> linear input index
	size_t in_idx = 0;
	// size_t stride = 1;
	
	for (size_t i = mat_dim; i-- > 0; ) {
		in_idx += in_coord[i] * in_stride[i];
		// stride *= in_shape[i];
	}
	
	out[idx] = in[in_idx];
}

__global__ void contiguous_kernel(
	float* out, const float* in,
	const size_t* in_shape, const size_t* in_stride,
	size_t mat_dim, size_t n_elements)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_elements) {
		return;
	}
	
	size_t tmp = idx;
	size_t coord[16] = { }; // it should be very sufficient so far
	for (size_t i = mat_dim; i-- > 0; ) {
		coord[i] = tmp % in_shape[i];
		tmp /= in_shape[i];
	}
	size_t in_idx = 0;
	
	for (size_t i = mat_dim; i-- > 0; ) {
		in_idx += coord[i] * in_stride[i];
	}
	
	out[idx] = in[in_idx];
}

template<int TILE = 32>
__global__ void transpose_tiled_kernel(
	float* __restrict__ out,
	const float* __restrict__ in,
	size_t rows, size_t cols)
{
	__shared__ float tile[TILE][TILE + 1];
	
	size_t x = blockIdx.x * TILE + threadIdx.x;
	size_t y = blockIdx.y * TILE + threadIdx.y;
	
	
	if (x < cols && y < rows) {
		tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
	}
	
	__syncthreads();
	
	
	size_t trans_x = blockIdx.y * TILE + threadIdx.x;
	size_t trans_y = blockIdx.x * TILE + threadIdx.y;
	
	
	if (trans_x < rows && trans_y < cols) {
		out[trans_y * rows + trans_x] =
		tile[threadIdx.x][threadIdx.y];
	}
}

__global__ void tensor_mul_scalar_kernel(float* x, float value, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n)
		x[i] *= value;
}

// WARNING softmax requires power-of-two block
template<bool unfinished_softmax = false>
__global__ void mat_softmax_kernel(const float* x, float* output, size_t cols)
{
	extern __shared__ float smem[];
	float* reduce = smem;
	
	const size_t tid = threadIdx.x;
	const size_t row = blockIdx.x;
	const size_t block_size = blockDim.x;
	
	// causal limit:
	// normal softmax: all cols
	// causal softmax: only [0, row]
	size_t valid_cols;
	if constexpr (unfinished_softmax) {
		valid_cols = min(row + 1, cols);
	} else {
		valid_cols = cols;
	}
	// reduce max(x)
	float local_max = -INFINITY;
	// flexible loader, so I can cover much larger tensor regardless of block_size
	for (size_t i = tid; i < valid_cols; i += block_size) {
		local_max = std::fmaxf(local_max, x[row * cols + i]);
	}
	reduce[tid] = local_max;
	__syncthreads();
	
	for (size_t stride = block_size / 2; stride > 0; stride /= 2) {
		if (tid < stride) {
			reduce[tid] = std::fmaxf(reduce[tid], reduce[tid + stride]);
		}
		__syncthreads();
	}
	
	float max_value = reduce[0];
	__syncthreads();
	
	// reduce sum(exp(x-max))
	float local_sum = 0.f;
	
	for (size_t i = tid; i < valid_cols; i += block_size) {
		local_sum += std::expf(x[row * cols + i] - max_value);
	}
	reduce[tid] = local_sum;
	__syncthreads();
	
	for (size_t stride = block_size / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			reduce[tid] += reduce[tid + stride];
		}
		__syncthreads();
	}
	
	float sum = reduce[0];
	__syncthreads();
	
	// write softmax
	for (size_t i = tid; i < valid_cols; i += block_size) {
		if constexpr (unfinished_softmax) {
			if (i > row) {
				output[row * cols + i] = 0.f;
				continue;
			}
		}
		output[row * cols + i] = expf(x[row * cols + i] - max_value) / sum;
	}
}

__global__ void matadd_kernel(float* c, const float* a, const float* b, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}


template<int SAME_MATRIX_BLOCK = 16>
int same_matrix_helper(const float* d_a, const float* d_b, int rows, int cols)
{
	int diff_count = 0;
	int* d_diff_count = nullptr;
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	
	ck = hipMalloc(&d_diff_count, sizeof(int));
	ck = hipMemcpy(d_diff_count, &diff_count, sizeof(diff_count), hipMemcpyHostToDevice);
	
	dim3 block(SAME_MATRIX_BLOCK, SAME_MATRIX_BLOCK);
	dim3 grid(
		(cols + block.x - 1) / block.x,
			  (rows + block.y - 1) / block.y
	);
	
	same_matrix<<<grid, block>>>(d_a, d_b, rows, cols, d_diff_count);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
	
	ck = hipMemcpy(&diff_count, d_diff_count, sizeof(diff_count), hipMemcpyDeviceToHost);
	
	ck = hipFree(d_diff_count);
	
	return diff_count;
}

void print_matrix(const float* c, int rows, int cols, int rows_max, int cols_max)
{
	std::stringstream ss;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			ss << std::setw(8) << std::fixed << std::setprecision(2)
				<< c[i * cols + j] << " ";
		}
		ss << "\n";
	}
	shark::log::info("{}", ss.str());
}


__inline__ __device__
float warp_reduce_sum(float value)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		value += __shfl_down(value, offset);
	}
	return value;
}

__global__ void rmsnorm_kernel(
	float* o, const float* x, const float* weights,
	size_t rows, size_t cols, float eps)
{
	const size_t row = blockIdx.x;
	
	if (row >= rows)
		return;
	
	const size_t tid = threadIdx.x;
	const size_t block_size = blockDim.x;
	
	float sum_sq = 0.0f;
	
	// Each thread processes several elements
	for (size_t col = tid; col < cols; col += block_size) {
		float v = x[row * cols + col];
		sum_sq += v * v;
	}
	
	// Warp-level reduction
	sum_sq = warp_reduce_sum(sum_sq);
	
	// Reduce warp results
	__shared__ float warp_sum[32];
	
	const int lane = tid % warpSize;
	const int warp_id = tid / warpSize;
	
	if (lane == 0)
		warp_sum[warp_id] = sum_sq;
	
	__syncthreads();
	
	float block_sum = 0.0f;
	
	if (warp_id == 0) {
		const int num_warps = (block_size + warpSize - 1) / warpSize;
		
		if (lane < num_warps)
			block_sum = warp_sum[lane];
		
		block_sum = warp_reduce_sum(block_sum);
	}
	
	__shared__ float rms;
	
	if (tid == 0) {
		rms = sqrtf(block_sum / cols + eps);
	}
	
	__syncthreads();
	
	// Normalize
	for (size_t col = tid; col < cols; col += block_size) {
		o[row * cols + col] =
		x[row * cols + col] / rms * weights[col];
	}
}

// WARNING output should be contiguous, this restriction is applied for performance consideration.
template<size_t _block_dim, size_t _thread_dim>
__global__ void batched_stride_matmul_kernel(
	float* o, const float* a, const float* b,
	const size_t* a_stride,	const size_t* b_stride,
	size_t batch, size_t rows, size_t inner, size_t cols, bool b_batched, size_t b_batch_group)
{
	
	constexpr size_t _merged_dim = _block_dim * _thread_dim;
	
	__shared__ float a_shared[_merged_dim][_merged_dim];
	__shared__ float b_shared[_merged_dim][_merged_dim];
	
	size_t batch_idx = blockIdx.z;
	
	size_t row = blockIdx.y * _merged_dim + threadIdx.y * _thread_dim;
	size_t col = blockIdx.x * _merged_dim + threadIdx.x * _thread_dim;
	
	size_t b_batch_idx = batch_idx / b_batch_group;
	
	size_t o_offset = batch_idx * rows * cols;
	size_t a_offset = batch_idx * a_stride[0];
	size_t b_offset = b_batched ? b_batch_idx * b_stride[0] : 0;
	
	float sum[_thread_dim][_thread_dim] = { };
	
	for (size_t tile = 0; tile < inner; tile += _merged_dim) {
		
		// load a tile of A
		for (size_t local_row = 0; local_row < _thread_dim; ++local_row) {
			for (size_t local_col = 0; local_col < _thread_dim; ++local_col) {
				
				size_t row_s_a = row + local_row;
				size_t col_s_a = tile + threadIdx.x * _thread_dim + local_col;
				
				if (row_s_a < rows && col_s_a < inner) {
					a_shared[threadIdx.y * _thread_dim + local_row]
						[threadIdx.x * _thread_dim + local_col] =
					a[a_offset + row_s_a * a_stride[0] + col_s_a * a_stride[1]];
				} else {
					a_shared[threadIdx.y * _thread_dim + local_row]
						[threadIdx.x * _thread_dim + local_col] = 0.f;
				}
			}
		}
		
		// load a tile of B
		for (size_t local_row = 0; local_row < _thread_dim; ++local_row) {
			for (size_t local_col = 0; local_col < _thread_dim; ++local_col) {
				
				size_t row_s_b = tile + threadIdx.y * _thread_dim + local_row;
				size_t col_s_b = col + local_col;
				
				if (row_s_b < inner && col_s_b < cols) {
					b_shared[threadIdx.y * _thread_dim + local_row]
						[threadIdx.x * _thread_dim + local_col] =
					b[b_offset + row_s_b * b_stride[0] + col_s_b * b_stride[1]];
				} else {
					b_shared[threadIdx.y * _thread_dim + local_row]
						[threadIdx.x * _thread_dim + local_col] = 0.f;
				}
			}
		}
		
		__syncthreads();
		
		for (size_t k = 0; k < _merged_dim; ++k) {
			for (size_t local_row = 0; local_row < _thread_dim; ++local_row) {
				for (size_t local_col = 0; local_col < _thread_dim; ++local_col) {
					
					sum[local_row][local_col] +=
						a_shared[threadIdx.y * _thread_dim + local_row][k] *
						b_shared[k][threadIdx.x * _thread_dim + local_col];
				}
			}
		}
		
		__syncthreads();
	}
	
	for (size_t local_row = 0; local_row < _thread_dim; ++local_row) {
		for (size_t local_col = 0; local_col < _thread_dim; ++local_col) {
			
			size_t row_o = row + local_row;
			size_t col_o = col + local_col;
			
			if (row_o < rows && col_o < cols) {
				o[o_offset + row_o * cols + col_o] = sum[local_row][local_col];
			}
		}
	}
}

// ------------------------------------------------------------
// Generic benchmark helper
// ------------------------------------------------------------

struct BenchResult {
	float ms;
	double gflops;
};

// using KernelLauncher = void();
//template<typename KernelLauncher>

// template<int _block_dim>
float bench_kernel(
	const std::string& name, std::function<void()> launcher,
	int warmup = 2, int repeats = 10)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	// warmup
	for (int i = 0; i < warmup; ++i) {
		launcher();
		
		ck = hipGetLastError();
	}
	
	
	ck = hipDeviceSynchronize();
	
	hipEvent_t start;
	hipEvent_t stop;
	
	ck = hipEventCreate(&start);
	ck = hipEventCreate(&stop);
	
	ck = hipEventRecord(start);
	
	for (int i = 0; i < repeats; ++i) {
		launcher();
		
		// ck = hipGetLastError();
	}
	
	ck = hipEventRecord(stop);
	ck = hipEventSynchronize(stop);
	
	float total_ms = 0.0f;
	ck = hipEventElapsedTime(&total_ms, start, stop);
	
	ck = hipEventDestroy(start);
	ck = hipEventDestroy(stop);
	
	return total_ms / repeats;
}

void bench_report(size_t operations, 
				  uint32_t grid_rows, uint32_t grid_cols,
				  uint32_t block_rows, uint32_t block_cols, float avg_ms) {
	
	float seconds = avg_ms / 1000.f;
	// double operations = 2.0 * cols * rows * paired_dim;
	double gflops = double(operations) / seconds / 1e9;
	
	shark::log::info("[SHARK REPORT]\n"
		"operations: {}\ngrid: {}x{}\nblock: {}x{}\n"
		"threads per block: {}\naverage time per kernel: {}\ngflops: {}", 
		operations, grid_rows, grid_cols, block_rows, block_cols, 
		block_rows * block_cols, avg_ms, gflops);
}


template<typename Tp>
class LittleVector : public std::enable_shared_from_this<LittleVector<Tp>>
{

public:
	using value_type = Tp;
	using size_type = size_t;
	using host_container = std::vector<value_type>;
	
	LittleVector(size_type size) {
		data_.resize(size);
		r_alloc();
	}
	LittleVector(const host_container& raw) : data_(raw) {
		r_alloc();
		sync_HD();
	}
	LittleVector(host_container&& raw) : data_(raw) {
		r_alloc();
		sync_HD();
	}
	// copy of the object should be disabled because it may result in double free in destructor.
	LittleVector(const LittleVector& other)
		: data_(other.data_), rptr_(nullptr)
	{
		r_alloc();
		sync_HD();
	}
	
	LittleVector(LittleVector&& other) noexcept
		: data_(std::move(other.data_)), rptr_(other.rptr_)
	{
		other.rptr_ = nullptr;
	}
	LittleVector& operator=(LittleVector&& other) noexcept
	{
		if (this != &other) {
			r_dealloc();
			data_ = std::move(other.data_);
			rptr_ = other.rptr_;
			
			other.rptr_ = nullptr;
		}
		
		return *this;
	}
	/*
	template<typename ...Types>
	LittleVector(Types&& ...values) : data_(std::forward<Types>(values)...) {
		ck_ = hipMalloc(&rptr_, rbytes());
	}
	*/
	
	const size_t rbytes() const {
		return size() * sizeof(value_type);	
	}
	
	const size_t size() const {
		return data_.size();
	}
	
	void sync_HD() { 
		ck_ = hipMemcpy(rptr_, data_.data(), rbytes(), hipMemcpyHostToDevice);
	}
	void sync_DH() {
		ck_ = hipMemcpy(data_.data(), rptr_, rbytes(), hipMemcpyDeviceToHost);
	}
	value_type* rptr() {
		return this->rptr_;
	}
	const value_type* rptr() const {
		return this->rptr_;
	}
	
	auto begin() const { return data_.begin(); }
	auto end() const { return data_.end(); }
	
	LittleVector& operator=(const LittleVector&) = delete;
	
	void r_alloc() {
		shark::assert(rptr_ == nullptr);
		ck_ = hipMalloc(&rptr_, rbytes());
	}
	void r_dealloc() {
		if (rptr_) {
			ck_ = hipFree(rptr_);
			rptr_ = nullptr;
		}
	}
	~LittleVector() {
		r_dealloc();
	}
	
	void make_all(value_type value) {
		std::fill(data_.begin(), data_.end(), value);
		sync_HD();
	}
	void make_with(std::function<value_type()> generator) {
		for (float& x : data_) {
			x = generator();
		}
		sync_HD();
	}
	value_type operator[](size_type pos) const {
		return data_[pos];
	}
	void assign_(const host_container& new_data)
	{
		if (new_data.size() != data_.size()) {
			r_dealloc();
			data_.assign(new_data.begin(), new_data.end());
			r_alloc();
		} else {
			data_.assign(new_data.begin(), new_data.end());
		}
		
		sync_HD();
	}
	host_container& vec() {
		return data_;
	}
	const host_container& vec() const {
		return data_;
	}
	bool operator==(const host_container& other) const {
		return data_ == other;
	}
	bool operator==(const LittleVector& other) const {
		return data_ == other.data_;
	}
	
private:
	value_type* rptr_ = nullptr;  // ROCm Pointer
	host_container data_;
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck_; // It contains no member variables.
};



constexpr size_t numel(const std::vector<size_t>& shape)
{
	size_t total = 1;
	for (size_t mul : shape) {
		total *= mul;
	}
	return total;
}

constexpr std::vector<size_t> make_contiguous_stride(
	const std::vector<size_t>& shape)
{
	std::vector<size_t> stride(shape.size());
	size_t step = 1;
	
	for (size_t i = shape.size(); i-- > 0;) {
		stride[i] = step;
		step *= shape[i];
	}
	return stride;
}


class LittleTensor
{
public:
	using size_type = size_t;
	using value_type = float;
	
	struct from_shared_data_t { };
	inline static constexpr from_shared_data_t from_shared_data{};
	
	LittleTensor(const std::vector<size_type>& shape)
		: shape_(shape),
		data_(std::make_shared<decltype(data_)::element_type>(shark::numel(shape))),
		stride_(make_contiguous_stride(shape))
	{ }
	
	LittleTensor(LittleTensor&& other) noexcept :
		shape_(std::move(other.shape_)),
		stride_(std::move(other.stride_)), data_(std::move(other.data_))
	{ }
	LittleTensor& operator=(LittleTensor&& other) noexcept
	{
		if (this != &other)
		{
			shape_ = std::move(other.shape_);
			stride_ = std::move(other.stride_);
			data_ = std::move(other.data_);
		}
		return *this;
	}
	
	// This constructor will share the underlying data_.
	LittleTensor(from_shared_data_t, const LittleTensor& other) :
		shape_(other.shape_),
		stride_(other.stride_), data_(other.data_)
	{ }
	
	LittleTensor(const LittleTensor& other) :
		shape_(other.shape_), stride_(other.stride_), data_(other.data_)
	{ }
	
	int operator==(const LittleTensor& other) const
	{
		if (shape_ != other.shape_)
			return false;
		
		int rows = shape_[shape_.size()-2];
		int cols = shape_[shape_.size()-1];
		
		int wrong = same_matrix_helper(rptr(), other.rptr(), rows, cols);
		
		return wrong;
	}
	
	template<int64_t pos>
	constexpr size_type size() const {
		if constexpr (pos >= 0) {
			return shape()[pos];
		} else {
			return shape()[shape().size() + pos];
		}
	}
	
	constexpr size_type dim() const { return shape().size(); }
	
	LittleTensor rmsnorm_cpu(const LittleTensor& x, float eps = 1e-5f) const
	{
		size_type rows = x.size<-2>();
		size_type cols = x.size<-1>();
		
		LittleTensor o({rows, cols});
		
		const float* x_ptr = x.data_->vec().data();
		const float* w_ptr = data_->vec().data();
		float* o_ptr = o.data_->vec().data();
		
		for (size_t j = 0; j < rows; ++j) {
			float s = 0.f;
			
			for (size_t i = 0; i < cols; ++i) {
				float v = x_ptr[j * cols + i];
				s += v * v;
			}
			
			float m = sqrtf(s / cols + eps);
			
			for (size_t i = 0; i < cols; ++i) {
				o_ptr[j * cols + i] =
				x_ptr[j * cols + i] / m * w_ptr[i];
			}
		}
		
		o.sync_HD();
		return o;
	}
	
	
	LittleTensor rmsnorm(const LittleTensor& x, float eps = 1e-5f) const
	{
		size_t rows = x.size<-2>();
		size_t cols = x.size<-1>();
		
		if (dim() != 1 || size<0>() != cols) {
			shark::raise(
				"rmsnorm weight mismatch: input cols={}, weight={}, dim={}",
				cols, size<0>(), dim()
			);
		}
		// TODO: sync device -> host if rptr() is GPU memory
		// This version assumes CPU-accessible pointers.
		LittleTensor o({rows, cols});
		const float* x_ptr = x.rptr();
		const float* w_ptr = rptr();
		float* o_ptr = o.rptr();
		
		constexpr size_type _block = 256;
		// constexpr size_t _col_per_thread = 4;
		dim3 block(_block);
		dim3 grid(rows);
		rmsnorm_kernel<<<grid, block>>>(o_ptr, x_ptr, w_ptr, rows, cols, eps);
		
		return o;
	}
	
	void permute_(const std::vector<size_type>& perm)
	{
		if (perm.size() != dim()) {
			shark::raise("permute dimension mismatch");
		}
		
		auto shape = shape_.vec();
		auto stride = stride_.vec();
		
		for (size_type i = 0; i < dim(); ++i) {
			shape[i] = shape_.vec()[perm[i]];
			stride[i] = stride_.vec()[perm[i]];
		}
		
		shape_.assign_(shape);
		stride_.assign_(stride);
	}
	
	LittleTensor contiguous() const
	{
		if (is_contiguous()) {
			shark::log::info("You gotta be careful,"
				"contiguous_() is called when it's already contiguous");
		}
		
		LittleTensor out(shape_.vec());
		dim3 block(256);
		dim3 grid((numel() + block.x - 1) / block.x);
		contiguous_kernel<<<grid, block>>>
			(out.rptr(), rptr(), shape_.rptr(), stride_.rptr(), dim(), numel());
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
		
		return out;
		// allocate new contiguous storage
		// copy data according to current stride
		// replace data_
		// reset stride_
		// update flag
	}
	
	void transpose_(size_type dim0, size_type dim1)
	{
		auto shape = shape_.vec();
		auto stride = stride_.vec();
		
		std::swap(shape[dim0], shape[dim1]);
		std::swap(stride[dim0], stride[dim1]);
		
		shape_.assign_(shape);
		stride_.assign_(stride);
	}
	
	// When possible, adopt the more efficient kernels instead.
	LittleTensor transpose() const
	{
		shark::assert(dim() == 2, "transpose only supports 2D tensors, got dim={}", dim());
		
		size_type rows = size<-2>();
		size_type cols = size<-1>();
		
		LittleTensor o({cols, rows});
		
		constexpr int TILE = 32;
		
		dim3 block(TILE, TILE);
		dim3 grid(
			(cols + TILE - 1) / TILE,
				  (rows + TILE - 1) / TILE
		);
		
		transpose_tiled_kernel<TILE><<<grid, block>>>(o.rptr(), rptr(), rows, cols);
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
		return o;
	}
	
	void reshape_(const std::vector<size_type>& new_shape)
	{
		shark::assert(is_contiguous(), "reshape_() should not be called when it's not contiguous");
		
		size_type new_elements = 1;
		for (size_type s : new_shape) {
			new_elements *= s;
		}
		shark::assert(new_elements == numel(), 
			"reshape mismatch: old elements={}, new elements={}", numel(), new_elements);
		
		this->shape_.assign_(new_shape);
		this->stride_.assign_(make_contiguous_stride(new_shape));
	}
	
	LittleTensor view(const std::vector<size_type>& new_shape) const {
		shark::assert(is_contiguous(), "view currently requires contiguous tensor");
		
		LittleTensor tensor{ from_shared_data, *this };
		tensor.reshape_(new_shape);
		return tensor;
	}
	
	void multiply_(value_type value)
	{
		size_t n = numel();
		dim3 block(256);
		dim3 grid((n + block.x - 1) / block.x);
		tensor_mul_scalar_kernel<<<grid, block>>>(rptr(), value, n);
		
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
	}
	
	template<bool unfinished_softmax = false>
	LittleTensor softmax() const
	{
		LittleTensor o(shape());
		
		dim3 block(256);
		dim3 grid(size<-2>());
		
		mat_softmax_kernel<unfinished_softmax><<<grid, block, sizeof(float) * block.x>>>(rptr(), o.rptr(), size<-1>());
		return o;
	}
	
	
#pragma region ACCESSORS & SYNCHRONIZERS & SPECIAL_PROPERTIES

	bool is_contiguous() const
	{
		size_type expected = 1;
		
		for (size_type i = dim(); i-- > 0;) {
			if (shape_[i] == 1)
				continue;
			
			if (stride_[i] != expected)
				return false;
			
			expected *= shape_[i];
		}
		
		return true;
	}
	
	void sync_HD() { 
		shape_.sync_HD();
		data_->sync_HD();
		stride_.sync_HD();
	}
	void sync_DH() {
		shape_.sync_DH();
		data_->sync_DH();
		stride_.sync_DH();
	}
	
	void make_all(float value) {
		data_->make_all(value);
	}
	void make_rand(float begin = 0.F, float end = 1.F) {
		data_->make_with([begin, end]{
			return shark::math::random_float(begin, end);}
		);
	}
	const std::vector<size_type>& stride() const {
		return stride_.vec();
	}
	const size_type* stride_ptr() const {
		return stride_.rptr();
	}
	const std::vector<size_type>& shape() const {
		return shape_.vec();
	}
	const size_type* shape_ptr() const {
		return shape_.rptr();
	}
	auto& data() {
		return data_;
	}
	const auto& data() const {
		return data_;
	}
	value_type* rptr() {
		return data_->rptr();
	}
	const value_type* rptr() const {
		return data_->rptr();
	}
	constexpr size_t numel() const {
		return shark::numel(shape());
	}
	const size_t rbytes() const {
		return numel() * sizeof(value_type);	
	}
	
#pragma endregion
	
private:
	/* perm is enough for axis reordering.
	 * stride is more general because it directly describes address movement,
	 * including layouts that are not permutations of a contiguous tensor.
	 */
	LittleVector<size_type> shape_;
	LittleVector<size_type> stride_;
	std::shared_ptr<LittleVector<value_type>> data_;
	// std::vector<size_t> shape_;
	// std::vector<float> data_;
	
	// float* rptr_ = nullptr;
	// bool is_contiguous_ = true;
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck_; // It contains no member variables.
};

template<int _block_dim = 16, int _thread_dim = 2>
LittleTensor matmul(const LittleTensor& a, const LittleTensor& b)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	
	size_t rows = a.size<-2>();
	size_t inner = a.size<-1>();
	size_t cols = b.size<-1>();
	
	if (inner != b.size<-2>()) {
		shark::raise("matmul inner mismatch");
	}
	
	size_t batch = 1;
	for (size_t i = 0; i < a.dim() - 2; ++i) {
		batch *= a.shape()[i];
	}
	
	std::vector<size_t> out_shape;
	for (size_t i = 0; i < a.dim() - 2; ++i) {
		out_shape.push_back(a.shape()[i]);
	}
	out_shape.push_back(rows);
	out_shape.push_back(cols);
	
	LittleTensor C(out_shape);
	
	constexpr size_t TILE = _block_dim * _thread_dim;
	dim3 block(_block_dim, _block_dim);
	dim3 grid(
		(cols + TILE - 1) / TILE,
			  (rows + TILE - 1) / TILE,
			  batch
	);
	bool b_batched = b.dim() > 2;
	size_t b_batch = 1;
	for (size_t i = 0; i < b.dim() - 2; ++i) {
		b_batch *= b.shape()[i];
	}
	size_t b_batch_group = batch / b_batch;
	batched_stride_matmul_kernel<_block_dim, _thread_dim><<<grid, block>>>
		(C.rptr(), a.rptr(), b.rptr(), a.stride_ptr(), b.stride_ptr(),
		 batch, rows, inner, cols, b_batched, b_batch_group);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
	
	return C;
}


class Linear
{
public:
	// Invesered compared to PyTorch
	// In Out
	Linear(size_t rows, size_t cols) : weight_({rows, cols})
	{
		weight_.make_rand(1e-5, 1e-6);
	}
	
	LittleTensor forward(const LittleTensor& x) const 
	{
		shark::assert(x.size<-1>() == weight_.size<-2>());
		auto new_shape = x.shape();
		new_shape[new_shape.size() - 1] = weight_.size<-1>();
		LittleTensor o = matmul(x, weight_);
		return o;
	}
	
	const LittleTensor& weight() const {
		return weight_;
	}
private:
	
	LittleTensor weight_;
};

class Attention
{
	const size_t dim_;
	const size_t q_head_;
	const size_t kv_head_;
	const size_t head_dim_;
	
public:
	Attention(size_t dim, size_t q_head, size_t kv_head) :
		dim_(dim), q_head_(q_head), kv_head_(kv_head), head_dim_(dim / q_head),
		q_proj({dim, dim}), o_proj({dim, dim}),
		k_proj({dim, dim / (kv_head / q_head)}), v_proj({dim, dim / (kv_head / q_head)})
	{
		shark::assert(q_head % kv_head == 0);
		
	}
	
	LittleTensor attn(const LittleTensor& q, const LittleTensor& k, const LittleTensor& v) const
	{
		auto kt = k.view(k.shape());
		kt.transpose_(2, 3);
		
		shark::assert(q.size<-1>() % k.size<-1>() == 0);
		shark::assert(k.size<-2>() == v.size<-2>(), "K and V sequence mismatch");
		
		auto s = matmul(q, kt);
		
		s.multiply_(1.F / std::sqrt(static_cast<float>(q.size<-1>())));
		
		auto p = s.softmax<true>();
		auto o = matmul(p, v);
		return o;
	}
	
	LittleTensor forward(const LittleTensor& x) const
	{
		size_t sz_embd = x.size<-1>();
		shark::assert(q_proj.weight().size<-1>() == sz_embd, "Q and x hidden mismatch");
		
		auto q1 = q_proj.forward(x);
		auto k1 = k_proj.forward(x);
		auto v1 = v_proj.forward(x);
		
		size_t sz_batch = x.size<0>();
		size_t sz_seq = x.size<1>();
		
		LittleTensor q = q1.view({sz_batch, sz_seq, this->q_head_, this->head_dim_});
		LittleTensor k = k1.view({sz_batch, sz_seq, this->kv_head_, this->head_dim_});
		LittleTensor v = v1.view({sz_batch, sz_seq, this->kv_head_, this->head_dim_});
		q.transpose_(1, 2);
		k.transpose_(1, 2);
		v.transpose_(1, 2);
		
		auto y = attn(q,k,v);
		y.transpose_(1, 2);
		auto y1 = y.contiguous().view({sz_batch, sz_seq, sz_embd});
		
		return o_proj.forward(y1);
	}
	
private:
	Linear k_proj;
	Linear q_proj;
	Linear v_proj;
	Linear o_proj;
};


namespace neural_network
{

struct Module
{
	
	
};

	
	
}


class AdamW
{
	struct State
	{
		LittleTensor m;
		LittleTensor v;
	};
	
	std::unordered_map<LittleTensor*, State> states_;
	
	void step(const std::vector<LittleTensor*>& params)
	{
		for (auto* p : params)
		{
			auto& state = states_[p];
			
			if (!state exists)
				initialize m and v;
			
			update(*p, state);
		}
	}
};


class LittleGraph
{
public:
	using index_type = size_t;
	
	struct Value
	{
		Value() = default;
		Value(const std::shared_ptr<LittleTensor>& tensor) : tensor_(tensor) {}
		
		std::shared_ptr<LittleTensor> tensor_;
		std::optional<LittleTensor> grad_;
		bool requires_grad_ = false;
	};
	
	struct Node
	{
		enum class Operand {
			null, add, mul, matmul, transpose
		};
		
		Node(Operand operand, std::vector<index_type> input, index_type output) :
		input_(std::move(input)), output_(output), operand_(operand)
		{ }
		
		std::vector<index_type> input_;
		index_type output_;
		Operand operand_ = Operand::null;
		bool is_backward = false;
	};
	
public:
	index_type add_value_(const std::shared_ptr<LittleTensor>& tensor)
	{
		std::scoped_lock lk(mtx_);
		values_.emplace_back(tensor);
		return values_.size() - 1;
	}
	
	index_type add_node_(
		Node::Operand operand, std::vector<index_type> input, index_type output)
	{
		nodes_.emplace_back(operand, std::move(input), output);
		return nodes_.size() - 1;
	}
	
	index_type make_grad_node_(
		Node::Operand operand, std::vector<index_type> input)
	{
		index_type output = values_.size();
		values_.emplace_back();
		
		auto idx = add_node_(operand, std::move(input), output);
		nodes_[idx].is_backward = true;
		
		return output;
	}
	
	index_type accumulate(
		std::vector<std::optional<index_type>>& grad,
		index_type dst, index_type incoming)
	{
		if (!grad[dst]) {
			grad[dst] = incoming;
		} else {
			grad[dst] = make_grad_node_(
				Node::Operand::add, {*grad[dst], incoming}
			);
		}
		
		return *grad[dst];
	}
	
	std::vector<std::optional<index_type>>
	compile_backward(index_type loss)
	{
		shark::assert(loss < values_.size());
		shark::assert(values_[loss].tensor_ != nullptr);
		
		size_t forward_size = nodes_.size();
		size_t value_size = values_.size();
		
		std::vector<std::optional<index_type>> grad(value_size);
		
		auto one = std::make_shared<LittleTensor>(
			values_[loss].tensor_->shape()
		);
		one->make_all(1);
		grad[loss] = add_value_(one);
		
		for (size_t i = forward_size; i-- > 0;) {
			Node node = nodes_[i];
			
			if (node.is_backward || !grad[node.output_])
				continue;
			
			auto output_grad = *grad[node.output_];
			
			switch (node.operand_) {
				case Node::Operand::matmul: {
					shark::assert(node.input_.size() == 2);
					
					auto A = node.input_[0];
					auto B = node.input_[1];
					
					auto Bt = make_grad_node_(
						Node::Operand::transpose, {B}
					);
					auto dA = make_grad_node_(
						Node::Operand::matmul, {output_grad, Bt}
					);
					auto At = make_grad_node_(
						Node::Operand::transpose, {A}
					);
					auto dB = make_grad_node_(
						Node::Operand::matmul, {At, output_grad}
					);
					
					accumulate(grad, A, dA);
					accumulate(grad, B, dB);
					break;
				}
				
				case Node::Operand::add: {
					shark::assert(node.input_.size() == 2);
					
					accumulate(grad, node.input_[0], output_grad);
					accumulate(grad, node.input_[1], output_grad);
					break;
				}
				
				default:
					break;
			}
		}
		
		return grad;
	}
	
	void execute_backward(
		const std::vector<std::optional<index_type>>& grad)
	{
		shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
		
		for (auto& node : nodes_) {
			if (!node.is_backward)
				continue;
			
			for (auto input : node.input_) {
				shark::assert(input < values_.size());
				shark::assert(
					values_[input].tensor_ != nullptr,
				  "backward input tensor is null: {}", input
				);
			}
			
			switch (node.operand_) {
				case Node::Operand::matmul: {
					shark::assert(node.input_.size() == 2);
					
					auto result = matmul(
						*values_[node.input_[0]].tensor_,
						*values_[node.input_[1]].tensor_
					);
					
					values_[node.output_].tensor_ =
					std::make_shared<LittleTensor>(std::move(result));
					break;
				}
				
				case Node::Operand::transpose: {
					shark::assert(node.input_.size() == 1);
					
					auto result =
					values_[node.input_[0]].tensor_->transpose();
					
					values_[node.output_].tensor_ =
					std::make_shared<LittleTensor>(std::move(result));
					break;
				}
				
				case Node::Operand::add: {
					shark::assert(node.input_.size() == 2);
					
					auto& A = *values_[node.input_[0]].tensor_;
					auto& B = *values_[node.input_[1]].tensor_;
					
					shark::assert(A.shape() == B.shape(), "gradient add shape mismatch");
					
					auto result = std::make_shared<LittleTensor>(A.shape());
					size_t n = result->numel();
					
					dim3 block(256);
					dim3 grid((n + block.x - 1) / block.x);
					
					matadd_kernel<<<grid, block>>>(
						result->rptr(), A.rptr(), B.rptr(),
												   static_cast<int>(n)
					);
					
					ck = hipGetLastError();
					ck = hipDeviceSynchronize();
					
					values_[node.output_].tensor_ = std::move(result);
					break;
				}
				
				default:
					shark::raise("unsupported backward operand");
					break;
			}
		}
		
		for (size_t i = 0; i < grad.size(); ++i) {
			if (!grad[i])
				continue;
			
			auto result = *grad[i];
			
			shark::assert(result < values_.size());
			shark::assert(
				values_[result].tensor_ != nullptr,
				 "gradient tensor is null: value={}, result={}", i, result
			);
			
			values_[i].grad_.emplace(*values_[result].tensor_);
		}
	}
	
	const std::optional<LittleTensor>& grad(index_type idx) const
	{
		shark::assert(idx < values_.size());
		return values_[idx].grad_;
	}
	
	const std::shared_ptr<LittleTensor>& tensor(index_type idx) const
	{
		shark::assert(idx < values_.size());
		return values_[idx].tensor_;
	}
	
private:
	std::vector<Value> values_;
	std::vector<Node> nodes_;
	std::mutex mtx_;
};



// shark END
}

// ./build/hip-test/hip-test
int main(int argc, char** argv)
{
	using namespace shark;
	
	using large = LittleTensor;
	using little = std::shared_ptr<large>;
	
	LittleGraph graph;
	
	std::vector<size_t> basic_shape({1ULL, 1ULL});
	
	little A = std::make_shared<large>(basic_shape);
	A->make_all(2);
	auto iA = graph.add_value_(A);
	
	
	little B = std::make_shared<large>(basic_shape);
	B->make_all(3);
	auto iB = graph.add_value_(B);
	
	
	// C = A @ B
	auto C = std::make_shared<large>(matmul(*A, *B));
	auto iC = graph.add_value_(C);
	
	graph.add_node_(
		LittleGraph::Node::Operand::matmul,
		{iA, iB},
		iC
	);
	
	
	// D = C @ A
	auto D = std::make_shared<large>(matmul(*C, *A));
	auto iD = graph.add_value_(D);
	
	graph.add_node_(
		LittleGraph::Node::Operand::matmul,
		{iC, iA},
		iD
	);
	
	
	// F = 2A
	auto two = std::make_shared<large>(basic_shape);
	two->make_all(2);
	auto iTwo = graph.add_value_(two);
	
	auto F = std::make_shared<large>(matmul(*two, *A));
	auto iF = graph.add_value_(F);
	
	graph.add_node_(
		LittleGraph::Node::Operand::matmul,
		{iTwo, iA},
		iF
	);
	
	
	// E = D + F
	auto E = std::make_shared<large>(*D);
	E->make_all(16); // placeholder until add kernel exists
	auto iE = graph.add_value_(E);
	
	graph.add_node_(
		LittleGraph::Node::Operand::add,
		{iD, iF},
		iE
	);
	
	
	shark::log::info("C:");
	print_matrix(
		C->rptr(),
				 1, 1,
			  20, 20
	);
	
	shark::log::info("D:");
	print_matrix(
		D->rptr(),
				 1, 1,
			  20, 20
	);
	
	shark::log::info("E:");
	print_matrix(
		E->rptr(),
				 1, 1,
			  20, 20
	);
	
	auto grad = graph.compile_backward(iE);
	graph.execute_backward(grad);
	
	print_matrix(graph.grad(iA)->rptr(), 1, 1, 20, 20);
	print_matrix(graph.grad(iB)->rptr(), 1, 1, 20, 20);
	
	
	// Expected:
	// dE/dA = 14
	// dE/dB = 4
	
	// TODO: expose graph value/gradient accessor
	
	
	return 0;
}
