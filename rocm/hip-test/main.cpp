// Tiny program to test ROCm status
// #define __HIP_DISABLE_CPP_FUNCTIONS__
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#include "shark/shark.h"

__global__ void matadd_kernel(float* c, const float* a, const float* b, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

int main()
{
	try{
		const int n = 1024 * 1024;
		size_t bytes = n * sizeof(float);
		
		std::vector<float> h_a(n, 1.f);
		std::vector<float> h_b(n, 2.f);
		std::vector<float> h_c(n);
		float* d_a = nullptr;
		float* d_b = nullptr;
		float* d_c = nullptr;
		
		shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
		
		ck = hipMalloc(&d_a, bytes);
		ck = hipMalloc(&d_b, bytes);
		ck = hipMalloc(&d_c, bytes);
		
		ck = hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
		ck = hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);
		
		int threads = 256;
		int blocks = (n + threads - 1) / threads;
		
		matadd_kernel<<<blocks, threads>>>(d_c, d_a, d_b, n);
		
		ck = hipDeviceSynchronize();
		ck = hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost);
		
		std::cout << h_c[0] << std::endl;
		
		ck = hipFree(d_a);
		ck = hipFree(d_b);
		ck = hipFree(d_c);
	}
	catch(std::exception& exc) {
		std::cout << exc.what() << std::endl;
		
	}
}


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
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>


__global__ void thread_id_test(int* output)
{
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	output[global_id] = global_id;
}


int main()
{
	constexpr int N = 64;
	
	std::vector<int> h_output(N, -1);
	
	int* d_output;
	
	hipMalloc(&d_output, N * sizeof(int));
	
	
	int threads = 16;
	int blocks = N / threads;
	
	
	thread_id_test<<<blocks, threads>>>(d_output);
	
	hipDeviceSynchronize();
	
	
	hipMemcpy(
		h_output.data(),
			  d_output,
			  N * sizeof(int),
			  hipMemcpyDeviceToHost);
	
	
	for (int i = 0; i < N; i++)
	{
		std::cout << i << " -> "
		<< h_output[i]
		<< "\n";
	}
	
	
	hipFree(d_output);
	
	return 0;
}
*/
