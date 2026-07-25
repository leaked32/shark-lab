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

using KernelLauncher = void(float*, const float*, const float*, int);
//template<typename KernelLauncher>
BenchResult bench_matmul(
	const std::string& name, KernelLauncher launcher,
	float* d_c, const float* d_a, const float* d_b,
	int dim, int warmup = 10, int repeats = 100)
{
	// warmup
	for (int i = 0; i < warmup; ++i) {
		launcher(d_c, d_a, d_b, dim);
	}
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipDeviceSynchronize();
	
	hipEvent_t start;
	hipEvent_t stop;
	
	ck = hipEventCreate(&start);
	ck = hipEventCreate(&stop);
	ck = hipEventRecord(start);
	
	for (int i = 0; i < repeats; ++i) {
		launcher(d_c, d_a, d_b, dim);
	}
	
	ck = hipEventRecord(stop);
	ck = hipEventSynchronize(stop);
	
	float total_ms = 0.f;
	
	ck = hipEventElapsedTime(&total_ms, start, stop);
	
	ck = hipEventDestroy(start);
	ck = hipEventDestroy(stop);
	
	float avg_ms = total_ms / repeats;
	
	
	// C=A*B:
	// each output does dim multiplications + additions
	double operations = 2.0 * dim * dim * dim;
	
	double seconds = avg_ms / 1000.0;
	
	double gflops =
	operations / seconds / 1e9;
	
	std::stringstream ss;
	ss << name << "\n" << "  time: " << avg_ms << " ms\n"
		<< "  perf: " << gflops << " GFLOPS\n\n";
	
	shark::log::info("{}", ss.str());
	
	return { avg_ms, gflops };
}


// ------------------------------------------------------------
// Convenient benchmark functions
// ------------------------------------------------------------

template<int BLOCK>
BenchResult bench_naive_matmul(float* d_c, const float* d_a, const float* d_b, int dim)
{
	return bench_matmul(
		"naive matmul",
		[](float* c, const float* a, const float* b, int dim)
		{
			dim3 block(BLOCK, BLOCK);
			dim3 grid((dim + BLOCK - 1) / BLOCK, (dim + BLOCK - 1) / BLOCK);
			
			matmul_kernel<BLOCK><<<grid, block>>>(c, a, b, dim);
		},
		d_c, d_a, d_b, dim
	);
}


template<int BLOCK>
BenchResult bench_tiled_matmul(float* d_c, const float* d_a, const float* d_b, int dim)
{
	return bench_matmul(
		"shared tiled matmul",
		[](float* c, const float* a, const float* b, int dim)
		{
			dim3 block(BLOCK, BLOCK);
			dim3 grid((dim + BLOCK - 1) / BLOCK, (dim + BLOCK - 1) / BLOCK);
			
			matmul_kernel_1<BLOCK><<<grid, block>>>(c, a, b, dim);
		},
		d_c, d_a, d_b, dim
	);
}

int main()
{
	
	int dim = 2048;
	// int threads = 256;
	
	const int elements = dim * dim;
	
	size_t bytes = elements * sizeof(float);
	
	std::vector<float> h_a(elements, 1.f);
	std::vector<float> h_b(elements, 2.f);
	std::vector<float> h_c(elements);
	float* d_a = nullptr;
	float* d_b = nullptr;
	float* d_c = nullptr;
	
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	
	hipEvent_t start;
	hipEvent_t stop;
	
	ck = hipEventCreate(&start);
	ck = hipEventCreate(&stop);
	
	ck = hipEventRecord(start);
	
	shark::profiler pf(64, true);
	
	pf.reset();
	
	ck = hipMalloc(&d_a, bytes);
	ck = hipMalloc(&d_b, bytes);
	ck = hipMalloc(&d_c, bytes);
	
	pf.lap("hipMalloc");
	
	pf.reset();
	ck = hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
	ck = hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
	pf.lap("H->D");
	// bench_naive_matmul<16>(d_c, d_a, d_b, dim);
	// bench_tiled_matmul<8>(d_c, d_a, d_b, dim);
	// bench_tiled_matmul<16>(d_c, d_a, d_b, dim);
	// bench_tiled_matmul<32>(d_c, d_a, d_b, dim);
	
	
	auto df = [&]<int _block_dim> {
		
		dim3 block(_block_dim, _block_dim);
		dim3 grid((dim + block.x - 1) / block.x, (dim + block.y - 1) / block.y);
		matmul_kernel_1<_block_dim><<<grid, block>>>(d_c,d_a,d_b,dim);
		ck = hipGetLastError();
		ck = hipDeviceSynchronize();
		pf.lap("matmul_kernel_1");
		
		pf.reset();
		ck = hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);
		pf.lap("D->H");
	};
	df.template operator()<16>();
	std::vector<float> cp_h_c(h_c.begin(), h_c.end());
	ck = hipMemset(d_c, 0, bytes);
	df.template operator()<8>();
	
	float* cp_d_c = nullptr;
	ck = hipMalloc(&cp_d_c, bytes);
	ck = hipMemcpy(cp_d_c, cp_h_c.data(), bytes, hipMemcpyHostToDevice);
	// same_matrix<<<grid, block>>>(d_a, d_c, dim, dim);
	// same_matrix<<<grid, block>>>(cp_d_c, d_c, dim, dim);
	dim3 block(16,16);
	dim3 grid((dim+block.x - 1)/block.x, (dim+block.y - 1)/block.y);
	int diff_count = same_matrix_helper(grid, block, cp_d_c, d_c, dim, dim);
	shark::log::info("diff_count: {}", diff_count);
	// diff_count = same_matrix_helper(grid, block, d_a, d_c, dim, dim);
	// shark::log::info("diff_count: {}", diff_count);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
	pf.lap("same_matrix");
	
	
	pf.reset();
	ck = hipFree(d_a);
	ck = hipFree(d_b);
	ck = hipFree(d_c);
	pf.lap("hipFree");
	ck = hipEventDestroy(start);
	ck = hipEventDestroy(stop);
}

/*
int main()
{
	
	try{
		const int dim = 8;
		const int elements = dim * dim;
		
		size_t bytes = elements * sizeof(float);
		
		std::vector<float> h_a(elements, 1.f);
		std::vector<float> h_b(elements, 2.f);
		std::vector<float> h_c(elements);
		float* d_a = nullptr;
		float* d_b = nullptr;
		float* d_c = nullptr;
		
		shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
		
		hipEvent_t start;
		hipEvent_t stop;
		
		ck = hipEventCreate(&start);
		ck = hipEventCreate(&stop);
		
		ck = hipEventRecord(start);
		
		shark::profiler pf(64, true);
		
		pf.reset();
		
		ck = hipMalloc(&d_a, bytes);
		ck = hipMalloc(&d_b, bytes);
		ck = hipMalloc(&d_c, bytes);
		
		pf.lap("hipMalloc");
		
		int threads = 256;
		int blocks = (dim + threads - 1) / threads;
		
		for (int iteration = 0; iteration < 20; ++iteration) {
			pf.reset();
			ck = hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
			ck = hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
			pf.lap("H->D");
			
			// constexpr int repeats = 100;
			
			ck = hipEventRecord(start);
			
			// for (int i = 0; i < repeats; ++i) {
			matmul_kernel<<<blocks, threads>>>(d_c, d_a, d_b, dim);
			// }
			
			ck = hipGetLastError();
			ck = hipEventRecord(stop);
			ck = hipEventSynchronize(stop);
			
			float total_ms = 0.f;
			ck = hipEventElapsedTime(&total_ms, start, stop);
			
			// std::cout << "Average kernel: "
			// << total_ms / repeats
			// << " ms\n";
			
			pf.reset();
			ck = hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);
			pf.lap("D->H");
			
			print_matrix(d_c, dim, dim);
		}
		
		
		std::cout << h_c[0] << std::endl;
		
		pf.reset();
		ck = hipFree(d_a);
		ck = hipFree(d_b);
		ck = hipFree(d_c);
		pf.lap("hipFree");
		ck = hipEventDestroy(start);
		ck = hipEventDestroy(stop);
		
	}
	catch(std::exception& exc) {
		std::cout << exc.what() << std::endl;
		
	}
	
	
	return 0;
}
*/
/*
int main1()
{
	try{
		const int n = 32 * 1024 * 1024; // 128 MiB per float array
		size_t bytes = n * sizeof(float);
		
		std::vector<float> h_a(n, 1.f);
		std::vector<float> h_b(n, 2.f);
		std::vector<float> h_c(n);
		float* d_a = nullptr;
		float* d_b = nullptr;
		float* d_c = nullptr;
		
		shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
		
		hipEvent_t start;
		hipEvent_t stop;
		
		ck = hipEventCreate(&start);
		ck = hipEventCreate(&stop);
		
		ck = hipEventRecord(start);
		
		shark::profiler pf(64, true);
		
		pf.reset();
		
		ck = hipMalloc(&d_a, bytes);
		ck = hipMalloc(&d_b, bytes);
		ck = hipMalloc(&d_c, bytes);
		
		pf.lap("hipMalloc");
		
		int threads = 256;
		int blocks = (n + threads - 1) / threads;
		
		for (int iteration = 0; iteration < 20; ++iteration) {
			pf.reset();
			ck = hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
			ck = hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
			pf.lap("H->D");
			
			constexpr int repeats = 1000;
			
			ck = hipEventRecord(start);
			
			for (int i = 0; i < repeats; ++i) {
				matadd_kernel<<<blocks, threads>>>(d_c, d_a, d_b, n);
			}
			
			ck = hipGetLastError();
			ck = hipEventRecord(stop);
			ck = hipEventSynchronize(stop);
			
			float total_ms = 0.f;
			ck = hipEventElapsedTime(&total_ms, start, stop);
			
			std::cout << "Average kernel: "
			<< total_ms / repeats
			<< " ms\n";
			
			pf.reset();
			ck = hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);
			pf.lap("D->H");
		}
		
		
		std::cout << h_c[0] << std::endl;
		
		pf.reset();
		ck = hipFree(d_a);
		ck = hipFree(d_b);
		ck = hipFree(d_c);
		pf.lap("hipFree");
		ck = hipEventDestroy(start);
		ck = hipEventDestroy(stop);
		
	}
	catch(std::exception& exc) {
		std::cout << exc.what() << std::endl;
		
	}
	
	
}
*/

/*
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>


__global__ void vector_add(
	const float* a,
	const float* b,
	float* c,
	int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n)
	{
		c[i] = a[i] + b[i];
	}
}


int main1()
{
	constexpr int N = 1024;
	
	std::vector<float> h_a(N, 1.0f);
	std::vector<float> h_b(N, 2.0f);
	std::vector<float> h_c(N, 0.0f);
	
	
	float* d_a;
	float* d_b;
	float* d_c;
	
	
	hipMalloc(&d_a, N * sizeof(float));
	hipMalloc(&d_b, N * sizeof(float));
	hipMalloc(&d_c, N * sizeof(float));
	
	
	hipMemcpy(
		d_a,
		h_a.data(),
			N * sizeof(float),
			hipMemcpyHostToDevice);
	
	hipMemcpy(
		d_b,
		h_b.data(),
			N * sizeof(float),
			hipMemcpyHostToDevice);
	
	
	int threads = 256;
	int blocks = (N + threads - 1) / threads;
	
	
	vector_add<<<blocks, threads>>>(
		d_a,
		d_b,
		d_c,
		N);
	
	
	hipDeviceSynchronize();
	
	
	hipMemcpy(
		h_c.data(),
			d_c,
		N * sizeof(float),
			hipMemcpyDeviceToHost);
	
	
	std::cout << "c[0] = " << h_c[0] << "\n";
	std::cout << "c[1023] = " << h_c[1023] << "\n";
	
	
	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_c);
	
	return 0;
}
*/
