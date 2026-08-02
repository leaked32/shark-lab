#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace shark
{

void mhip_malloc(void** ptr, size_t bytes);


template<typename Tp>
void mhip_malloc(Tp** ptr, size_t bytes)
{
	mhip_malloc(reinterpret_cast<void**>(ptr), bytes);
}


void mhip_free(void* ptr);


void mhip_memcpy_HD(void* dst, const void* src, size_t bytes);


void mhip_memcpy_DH(void* dst, const void* src, size_t bytes);


void same_matrix_kernel_launch(
	const float* a, const float* b,
	int rows, int cols, int* is_diff
);


int same_matrix_helper(
	const float* d_a, const float* d_b,
	int rows, int cols
);


void permute_kernel_launch(
	float* out, const float* in,
	const size_t* in_shape, const size_t* perm, const size_t* in_stride,
	size_t mat_dim, size_t n_elements
);


void contiguous_kernel_launch(
	float* out, const float* in,
	const size_t* in_shape, const size_t* in_stride,
	size_t mat_dim, size_t n_elements
);


template<int TILE>
void transpose_tiled_kernel_launch(
	float* out, const float* in,
	size_t rows, size_t cols
);


void tensor_mul_scalar_kernel_launch(float* x, float value, size_t n);


template<bool unfinished_softmax>
void mat_softmax_kernel_launch(
	const float* x, float* output,
	size_t rows, size_t cols
);


void matadd_kernel_launch(float* c, const float* a, const float* b, int n);


void rmsnorm_kernel_launch(
	float* o, const float* x, const float* weights,
	size_t rows, size_t cols, float eps
);


template<size_t _block_dim, size_t _thread_dim>
void batched_stride_matmul_kernel_launch(
	float* o, const float* a, const float* b,
	const size_t* a_stride, const size_t* b_stride,
	size_t batch, size_t rows, size_t inner, size_t cols,
	bool b_batched, size_t b_batch_group
);


float bench_kernel(
	const std::string& name, std::function<void()> launcher,
	int warmup = 2, int repeats = 10
);


}
