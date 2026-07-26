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
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
// #include <iostream>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>

#include "shark/shark.h"

__global__ void same_matrix(const float* a, const float* b, int rows, int cols, int* diff_count)
{
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row >= rows || col >= cols) {
		return;
	}
	
	float va = a[row * cols + col];
	float vb = b[row * cols + col];
	
	float diff = std::fabsf(va - vb);
	float tolerance = 1e-5f * std::max(1.0f, std::max(fabsf(va), fabsf(vb)));
	
	if (diff > tolerance) {
		// *diff_count += 1;
		// NOTICE: Actually, this will definitely deeply harm performance for SIMT.
		// But I would leave it alone so far.
		atomicAdd(diff_count, 1);
	}
	
}


__global__ void transpose_kernel(float* o, const float* in, size_t rows, size_t cols)
{
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < rows && col < cols) {
		o[col * rows + row] = in[row * cols + col];
	}
}

__global__ void tensor_mul_scalar_kernel(float* x, float value, size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n)
		x[i] *= value;
}

// WARNING softmax requires power-of-two block
__global__ void mat_softmax_kernel(const float* x, float* r, size_t cols)
{
	extern __shared__ float smem[];
	float* reduce = smem;
	
	const size_t tid = threadIdx.x;
	const size_t row = blockIdx.x;
	const size_t block_size = blockDim.x;
	
	// reduce max(x)
	float local_max = -INFINITY;
	// flexible loader, so I can cover much larger tensor regardless of block_size
	for (size_t i = tid; i < cols; i += block_size) {
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
	
	for (size_t i = tid; i < cols; i += block_size) {
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
	for (size_t i = tid; i < cols; i += block_size) {
		r[row * cols + i] = expf(x[row * cols + i] - max_value) / sum;
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


template<size_t _thread_dim>
__global__ void rmsnorm_kernel(
	float* o, const float* x, const float* weights, 
	size_t rows, size_t cols, float eps)
{
	// constexpr int _merged_dim = _block_dim * _thread_dim;
	// int row = blockIdx.y * _merged_dim + threadIdx.y * _thread_dim;
	// There's no way unless I use __global__ to support different blocks.
	// Disable it now.
	size_t row = blockIdx.x;
	size_t col = threadIdx.x * _thread_dim;
	
	__shared__ float sum;
	if (col == 0) {
		sum = 0.F;
	}
	__syncthreads();
	
	float local_sum = 0.F;
	if constexpr (_thread_dim == 1) {
		if (col < cols) {
			float tmp = x[row * cols + col] * x[row * cols + col];
			atomicAdd(&sum, tmp);
		}
	} else {
		for (size_t i = 0; i < _thread_dim; ++i) {
			size_t tmp = col + i;
			if (tmp < cols) {
				float tmp1 = x[row * cols + tmp] * x[row * cols + tmp];
				local_sum += tmp1;
			}
		}
		atomicAdd(&sum, local_sum);
	}
	
	__syncthreads();
	__shared__ float rms;
	if (threadIdx.x == 0) {
		rms = std::sqrt(sum / cols + eps);
	}
	__syncthreads();
	if constexpr (_thread_dim == 1) {
		if (col < cols) {
			o[row * cols + col] = x[row * cols + col] / rms * weights[col];
		}
	} else {
		for (size_t i = 0; i < _thread_dim; ++i) {
			size_t tmp = col + i;
			if (tmp < cols) {
				o[row * cols + tmp] = x[row * cols + tmp] / rms * weights[tmp];
			}
		}
	}
}
	

template<size_t _block_dim, size_t _thread_dim>
__global__ void matmul_kernel_2(
	float* c, const float* a, const float* b,
	size_t rows, size_t inner, size_t cols)
{
	constexpr size_t _merged_dim = _block_dim * _thread_dim;
	
	__shared__ float a_shared[_merged_dim][_merged_dim /*+ 1 BANK PADDING*/];
	__shared__ float b_shared[_merged_dim][_merged_dim /*+ 1 BANK PADDING*/];
	
	
	// __syncthreads();
	int row = blockIdx.y * _merged_dim + threadIdx.y * _thread_dim;
	int col = blockIdx.x * _merged_dim + threadIdx.x * _thread_dim;
	// block level coordinates: threadIdx.y, threadIdx.x
	// grid level coordinates: row, col
	
	float sum[_thread_dim][_thread_dim] = { };
	// sum for current position, each thread computes only one number of output.
	
	// loop to make a thick ROPE through A and B matrices respectively
	// notice we need numbers from A and B exceeds currently blocks.
	// From __global__ (VRAM) to __shared__ (LDS)
	for (int tile = 0; tile < inner; tile += _merged_dim) {
		// load a tile of A
		for (int local_row = 0; local_row < _thread_dim; ++local_row) {
			for (int local_col = 0; local_col < _thread_dim; ++local_col) {
				int row_s_a = local_row + row;
				int col_s_a = tile + threadIdx.x * _thread_dim + local_col;
				if (row_s_a < rows && col_s_a < inner) {
					// check whether it's still in matrix
					// `col` is irrelevant here because A does not use the output column
					
					a_shared[threadIdx.y * _thread_dim + local_row]
							[threadIdx.x * _thread_dim + local_col] = 
						a[row_s_a * inner + col_s_a];
				} else {
					a_shared[threadIdx.y * _thread_dim + local_row]
							[threadIdx.x * _thread_dim + local_col] = 0.f;
					// a_shared[threadIdx.y][threadIdx.x] = 0.f;
				}
			}
		}
		
		// load a tile of B
		for (int local_row = 0; local_row < _thread_dim; ++local_row) {
			for (int local_col = 0; local_col < _thread_dim; ++local_col) {
				int row_s_b = tile + threadIdx.y * _thread_dim + local_row;
				int col_s_b = col + local_col;
				if (row_s_b < inner && col_s_b < cols) {
					b_shared[threadIdx.y * _thread_dim + local_row]
							[threadIdx.x * _thread_dim + local_col] =
						b[row_s_b * cols + col_s_b];
				} else {
					b_shared[threadIdx.y * _thread_dim + local_row]
							[threadIdx.x * _thread_dim + local_col] = 0.f;
					// b_shared[threadIdx.y][threadIdx.x] = 0.f;
				}
			}
		}
		__syncthreads();
		
		// partial matrix multiplication results.
		for (int k = 0; k < _merged_dim; ++k) {
			for (int local_row = 0; local_row < _thread_dim; ++local_row) {
				for (int local_col = 0; local_col < _thread_dim; ++local_col) {
					sum[local_row][local_col] +=
						a_shared[threadIdx.y * _thread_dim + local_row][k] * 
						b_shared[k][threadIdx.x * _thread_dim + local_col];
				}
			}
		}
		
		__syncthreads();
	}
	
	for (int local_row = 0; local_row < _thread_dim; ++local_row) {
		for (int local_col = 0; local_col < _thread_dim; ++local_col) {
			if (row + local_row < rows && col + local_col < cols) {
				c[(row + local_row) * cols + col + local_col] = sum[local_row][local_col];
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

// ------------------------------------------------------------
// Convenient benchmark functions
// ------------------------------------------------------------


template<int _block_dim, int _thread_dim>
float bench_tiled_matmul(float* d_c, const float* d_a, const float* d_b, int dim)
{
	constexpr int TILE = _block_dim * _thread_dim;
	
	dim3 block(_block_dim, _block_dim);
	dim3 grid(
		(dim + TILE - 1) / TILE,
			  (dim + TILE - 1) / TILE
	);
	
	float avg_ms = bench_kernel(
		"shared tiled matmul",
		[d_c, d_a, d_b, dim, grid, block] {
			matmul_kernel_2<_block_dim, _thread_dim><<<grid, block>>>
				(d_c, d_a, d_b, dim, dim, dim);
		}
	);
	bench_report(2ull * dim * dim * dim, grid.y, grid.x, block.y, block.x, avg_ms);
	return avg_ms;
}

class TinyTensor
{
public:
	constexpr TinyTensor(const std::vector<size_t>& shape)
	: shape_(shape)
	{
		data_.resize(elements());
		
		ck_ = hipMalloc(&rptr_, rbytes());
		// sync_HD();
	}
	
	// Compiler should help me optimize it, no worry.
	const size_t elements() const {
		size_t elements = 1;
		
		for (size_t dim : shape_) {
			elements *= dim;
		}
		return elements;
	}
	
	const size_t rbytes() const {
		return elements() * sizeof(float);	
	}
	
	void sync_HD() {
		ck_ = hipMemcpy(rptr_, data_.data(), rbytes(), hipMemcpyHostToDevice);
	}
	void sync_DH() {
		ck_ = hipMemcpy(data_.data(), rptr_, rbytes(), hipMemcpyDeviceToHost);
	}
	
	void make_all(float value) {
		std::fill(data_.begin(), data_.end(), value);
		sync_HD();
	}
	void make_rand(float begin = 0.F, float end = 1.F) {
		for (float& x : data_) {
			x = shark::math::random_float(begin, end);
		}
		sync_HD();
	}
	
	float* rptr() {
		return rptr_;
	}
	const float* rptr() const {
		return rptr_;
	}
	const auto& shape() const {
		return shape_;
	}
	
	TinyTensor clone() const {
		TinyTensor ntt(shape());
		ck_ = hipMemcpy(ntt.rptr_, rptr_, rbytes(), hipMemcpyDeviceToDevice);
		
		return ntt;
	}
	
	TinyTensor(TinyTensor&& other) noexcept
	{
		shape_ = std::move(other.shape_);
		data_ = std::move(other.data_);
		
		rptr_ = other.rptr_;
		other.rptr_ = nullptr;
	}
	
	// copy of the object should be disabled because it may result in double free in destructor.
	TinyTensor(const TinyTensor&) = delete;
	TinyTensor& operator=(const TinyTensor&) = delete;
	
	~TinyTensor() {
		if (rptr_) {
			ck_ = hipFree(rptr_);
			rptr_ = nullptr;
		}
	}
	
	template<int _block_dim = 16, int _thread_dim = 2>
	TinyTensor matmul(const TinyTensor& other) const {
		size_t rows = size<-2>();
		size_t inner = size<-1>();
		size_t cols = other.size<-1>();
		
		if (inner != other.size<-2>()) {
			shark::raise("matmul is invalid for: {}x{} {}x{}",
				rows, inner, other.size<-2>(), cols);
		}
		TinyTensor C({rows, cols});
		
		dim3 block(_block_dim, _block_dim);
		constexpr int TILE = _block_dim * _thread_dim;
		dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
		
		matmul_kernel_2<_block_dim, _thread_dim><<<grid, block>>>(
			C.rptr(), rptr(), other.rptr(), rows, inner, cols);
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
		return C;
	}
	
	bool operator==(const TinyTensor& other) const
	{
		if (shape_ != other.shape_)
			return false;
		
		int rows = shape_[shape_.size()-2];
		int cols = shape_[shape_.size()-1];
		
		int wrong = same_matrix_helper(rptr(), other.rptr(), rows, cols);
		
		return wrong == 0;
	}
	
	template<int64_t pos>
	constexpr size_t size() const {
		if constexpr (pos >= 0) {
			return shape()[pos];
		} else {
			return shape()[shape().size() + pos];
		}
	}
	
	constexpr size_t dim() const { return shape().size(); }
	
	TinyTensor rmsnorm_cpu(const TinyTensor& x, float eps = 1e-5f) const
	{
		size_t rows = x.size<-2>();
		size_t cols = x.size<-1>();
		
		TinyTensor o({rows, cols});
		
		const float* x_ptr = x.data_.data();
		const float* w_ptr = data_.data();
		float* o_ptr = o.data_.data();
		
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
	
	
	TinyTensor rmsnorm(const TinyTensor& x, float eps = 1e-5f) const
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
		TinyTensor o({rows, cols});
		const float* x_ptr = x.rptr();
		const float* w_ptr = rptr();
		float* o_ptr = o.rptr();
		
		constexpr size_t _block = 256;
		constexpr size_t _col_per_thread = 4;
		dim3 block(_block);
		dim3 grid(rows);
		rmsnorm_kernel<_col_per_thread><<<grid, block>>>(o_ptr, x_ptr, w_ptr, rows, cols, eps);
		
		return o;
	}
	
	TinyTensor transpose() const
	{
		if (dim() != 2) {
			shark::raise("transpose only supports 2D tensors, got dim={}", dim());
		}
		size_t rows = size<-2>();
		size_t cols = size<-1>();
		
		TinyTensor o({cols, rows}); // flipped
		
		dim3 block(16, 16);
		dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
		transpose_kernel<<<grid, block>>>(o.rptr(), rptr(), rows, cols);
		
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
		
		return o;
	}
	
	constexpr size_t numel() const
	{
		size_t n = 1;
		for (size_t s : shape()) {
			n *= s;
		}
		return n;
	}
	
	void multiply_(float value)
	{
		size_t n = numel();
		dim3 block(256);
		dim3 grid((n + block.x - 1) / block.x);
		tensor_mul_scalar_kernel<<<grid, block>>>(rptr(), value, n);
		
		ck_ = hipGetLastError();
		ck_ = hipDeviceSynchronize();
	}
	
	TinyTensor softmax() const
	{
		TinyTensor o(shape());
		
		dim3 block(256);
		dim3 grid(size<-2>());
		
		mat_softmax_kernel<<<grid, block, sizeof(float) * block.x>>>(rptr(), o.rptr(), size<-1>());
		return o;
	}
private:
	std::vector<size_t> shape_;
	std::vector<float> data_;
	
	float* rptr_ = nullptr; // ROCm Pointer
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck_; // It contains no member variables.
};

TinyTensor attention(const TinyTensor& q, const TinyTensor& k, const TinyTensor& v) {
	if (q.size<-1>() != k.size<-1>()) {
		shark::raise("Q and K hidden mismatch");
	}
	
	if (k.size<-2>() != v.size<-2>()) {
		shark::raise("K and V sequence mismatch");
	}
	
	auto s = q.matmul(k.transpose());
	s.multiply_(1.F / std::sqrt(static_cast<float>(q.size<-1>())));
	
	auto p = s.softmax();
	auto o = p.matmul(v);
	return o;
}

int main(int argc, char** argv)
{
	/*
	// Matrix multiplication
	const size_t dim = 2048;
	std::vector<size_t> shape = { dim, dim, };
	TinyTensor A(shape);
	TinyTensor B(shape);
	// TinyTensor C(shape);
	
	A.make_rand();
	B.make_rand();
	
	auto C = A.matmul(B);
	
	auto D = C.clone();
	
	bench_tiled_matmul<16,2>(C.rptr(), A.rptr(), B.rptr(), dim);
	
	shark::log::info("Equal: {}", C == D);
	// bench_tiled_matmul<16,2>(C.gptr(), A.gptr(), B.gptr(), dim);
	// bench_tiled_matmul<32>(C.gptr(), A.gptr(), B.gptr(), dim);
	*/
	
	const size_t dim = 2048;
	TinyTensor A( { dim, dim, });
	TinyTensor B( { dim, } );
	A.make_rand();
	B.make_rand();
	
	auto C = B.rmsnorm(A);
	auto D = B.rmsnorm_cpu(A);
	
	shark::log::info("Equal: {}", C == D);
	
	return 0;
}
