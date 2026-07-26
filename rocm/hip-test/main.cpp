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
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
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

int same_matrix_helper(dim3 grid, dim3 block, 
					   const float* d_a, const float* d_b, int rows, int cols) {
	int diff_count = 0;
	int* d_diff_count = nullptr;
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipMalloc(&d_diff_count, sizeof(int));
	ck = hipMemcpy(d_diff_count, &diff_count, sizeof(diff_count), hipMemcpyHostToDevice);
	same_matrix<<<grid, block>>>(d_a, d_b, rows, cols, d_diff_count);
	ck = hipDeviceSynchronize();
	ck = hipMemcpy(&diff_count, d_diff_count, sizeof(diff_count), hipMemcpyDeviceToHost);
	
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

__global__ void matadd_kernel(float* c, const float* a, const float* b, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

// constexpr int row = 8;
// constexpr int col = 8;

template<int _block_dim>
__global__ void matmul_kernel(
	float* c, const float* a, const float* b, int dim)
{
	int row = blockIdx.y * _block_dim + threadIdx.y;
	int col = blockIdx.x * _block_dim + threadIdx.x;
	
	// __syncthreads();
	if (dim <= row || dim <= col) {
		return;
	}
	float sum = 0.f;
	
	for (int k = 0; k < dim; ++k) {
		sum += a[row * dim + k] * b[k * dim + col];
	}
	
	c[row * dim + col] = sum;
}


template<int _block_dim>
__global__ void matmul_kernel_1(
	float* c, const float* a, const float* b, int dim)
{
	__shared__ float a_shared[_block_dim][_block_dim];
	__shared__ float b_shared[_block_dim][_block_dim];
	
	
	// __syncthreads();
	int row = blockIdx.y * _block_dim + threadIdx.y;
	int col = blockIdx.x * _block_dim + threadIdx.x;
	// block level coordinates: threadIdx.y, threadIdx.x
	// grid level coordinates: row, col
	
	float sum = 0.0f;
	// sum for current position, each thread computes only one number of output.
	
	// loop to make a thick ROPE through A and B matrices respectively
	// notice we need numbers from A and B exceeds currently blocks.
	for (int tile = 0; tile < dim; tile += _block_dim) {
		// load a tile of A
		int col_s_a = tile + threadIdx.x;
		if (row < dim && col_s_a < dim) {
			// check whether it's still in matrix
			// `col` is irrelevant here because A does not use the output column
			
			a_shared[threadIdx.y][threadIdx.x] = a[row * dim + col_s_a];
		} else {
			a_shared[threadIdx.y][threadIdx.x] = 0.f;
		}
		
		// load a tile of B
		int row_s_b = tile + threadIdx.y;
		if (row_s_b < dim && col < dim) {
			b_shared[threadIdx.y][threadIdx.x] = b[row_s_b * dim + col];
		} else {
			b_shared[threadIdx.y][threadIdx.x] = 0.f;
		}
		
		__syncthreads();
		
		// partial matrix multiplication results.
		for (int k = 0; k < _block_dim; ++k) {
			sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		}
		
		__syncthreads();
	}
	
	if (row < dim && col < dim) {
		c[row * dim + col] = sum;
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
	int warmup = 10, int repeats = 100)
{
	// warmup
	for (int i = 0; i < warmup; ++i) {
		launcher();
	}
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	
	ck = hipDeviceSynchronize();
	
	hipEvent_t start;
	hipEvent_t stop;
	
	ck = hipEventCreate(&start);
	ck = hipEventCreate(&stop);
	
	ck = hipEventRecord(start);
	
	for (int i = 0; i < repeats; ++i) {
		launcher();
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

template<int BLOCK>
float bench_naive_matmul(float* d_c, const float* d_a, const float* d_b, int dim)
{
	return bench_kernel(
		"naive matmul",
		[d_c, d_a, d_b, dim]
		{
			dim3 block(BLOCK, BLOCK);
			dim3 grid((dim + BLOCK - 1) / BLOCK, (dim + BLOCK - 1) / BLOCK);
			
			matmul_kernel<BLOCK><<<grid, block>>>(d_c, d_a, d_b, dim);
		}
	);
}


template<int BLOCK>
float bench_tiled_matmul(float* d_c, const float* d_a, const float* d_b, int dim)
{
	dim3 block(BLOCK, BLOCK);
	dim3 grid((dim + BLOCK - 1) / BLOCK, (dim + BLOCK - 1) / BLOCK);
	
	float avg_ms = bench_kernel(
		"shared tiled matmul",
		[d_c, d_a, d_b, dim, grid, block]
		{
			
			matmul_kernel_1<BLOCK><<<grid, block>>>(d_c, d_a, d_b, dim);
		}
	);
	
	
	bench_report(2ull * dim * dim * dim, grid.y, grid.x, block.y, block.x, avg_ms);
	
	return avg_ms;
}

class TinyTensor
{
public:
	TinyTensor(const std::vector<size_t>& shape)
	: shape_(shape)
	{
		data_.resize(elements());
		
		ck_ = hipMalloc(&rptr_, rbytes());
		sync_HD();
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
	
	float* gptr() {
		return rptr_;
	}
	const float* gptr() const {
		return rptr_;
	}
	const auto& shape() const {
		return shape_;
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
private:
	std::vector<size_t> shape_;
	std::vector<float> data_;
	
	float* rptr_ = nullptr; // ROCm Pointer
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck_; // It contains no member variables.
};


int main(int argc, char** argv)
{
	const size_t dim = 2048;
	std::vector<size_t> shape = {dim, dim, };
	TinyTensor A(shape);
	TinyTensor B(shape);
	TinyTensor C(shape);
	
	A.make_rand();
	B.make_rand();
	
	bench_tiled_matmul<32>(C.gptr(), A.gptr(), B.gptr(), dim);
	bench_tiled_matmul<32>(C.gptr(), A.gptr(), B.gptr(), dim);
	bench_tiled_matmul<32>(C.gptr(), A.gptr(), B.gptr(), dim);
	
	return 0;
}
