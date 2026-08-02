/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: hip-test/mhip.hip
 *
 * License: MIT
 */

#include <cmath>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>

#include "mhip.hpp"
#include "shark/shark.hpp"

namespace shark
{

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


void mhip_malloc(void** ptr, size_t bytes)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipMalloc(ptr, bytes);
}


void mhip_free(void* ptr)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipFree(ptr);
}


void mhip_memcpy_HD(void* dst, const void* src, size_t bytes)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
}


void mhip_memcpy_DH(void* dst, const void* src, size_t bytes)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	ck = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
}


void same_matrix_kernel_launch(
	const float* a, const float* b,
	int rows, int cols, int* is_diff)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	constexpr int SAME_MATRIX_BLOCK = 16;
	dim3 block(SAME_MATRIX_BLOCK, SAME_MATRIX_BLOCK);
	dim3 grid(
		(cols + block.x - 1) / block.x,
			  (rows + block.y - 1) / block.y
	);
	
	same_matrix<<<grid, block>>>(a, b, rows, cols, is_diff);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


int same_matrix_helper(const float* d_a, const float* d_b, int rows, int cols)
{
	int diff_count = 0;
	int* d_diff_count = nullptr;
	
	mhip_malloc(&d_diff_count, sizeof(int));
	mhip_memcpy_HD(d_diff_count, &diff_count, sizeof(diff_count));
	
	same_matrix_kernel_launch(d_a, d_b, rows, cols, d_diff_count);
	
	mhip_memcpy_DH(&diff_count, d_diff_count, sizeof(diff_count));
	mhip_free(d_diff_count);
	
	return diff_count;
}


void permute_kernel_launch(
	float* out, const float* in,
	const size_t* in_shape, const size_t* perm, const size_t* in_stride,
	size_t mat_dim, size_t n_elements)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	dim3 block(256);
	dim3 grid((n_elements + block.x - 1) / block.x);
	
	permute_kernel<<<grid, block>>>
		(out, in, in_shape, perm, in_stride, mat_dim, n_elements);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


void contiguous_kernel_launch(
	float* out, const float* in,
	const size_t* in_shape, const size_t* in_stride,
	size_t mat_dim, size_t n_elements)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	dim3 block(256);
	dim3 grid((n_elements + block.x - 1) / block.x);
	
	contiguous_kernel<<<grid, block>>>
		(out, in, in_shape, in_stride, mat_dim, n_elements);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


template<int TILE>
void transpose_tiled_kernel_launch(
	float* out, const float* in,
	size_t rows, size_t cols)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	dim3 block(TILE, TILE);
	dim3 grid(
		(cols + TILE - 1) / TILE,
			  (rows + TILE - 1) / TILE
	);
	
	transpose_tiled_kernel<TILE><<<grid, block>>>(out, in, rows, cols);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


void tensor_mul_scalar_kernel_launch(float* x, float value, size_t n)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	dim3 block(256);
	dim3 grid((n + block.x - 1) / block.x);
	
	tensor_mul_scalar_kernel<<<grid, block>>>(x, value, n);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


template<bool unfinished_softmax>
void mat_softmax_kernel_launch(
	const float* x, float* output,
	size_t rows, size_t cols)
{
	dim3 block(256);
	dim3 grid(rows);
	
	mat_softmax_kernel<unfinished_softmax>
		<<<grid, block, sizeof(float) * block.x>>>(x, output, cols);
}


void matadd_kernel_launch(float* c, const float* a, const float* b, int n)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	dim3 block(256);
	dim3 grid((n + block.x - 1) / block.x);
	
	matadd_kernel<<<grid, block>>>(c, a, b, n);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


void rmsnorm_kernel_launch(
	float* o, const float* x, const float* weights,
	size_t rows, size_t cols, float eps)
{
	constexpr size_t _block = 256;
	dim3 block(_block);
	dim3 grid(rows);
	rmsnorm_kernel<<<grid, block>>>(o, x, weights, rows, cols, eps);
}


template<size_t _block_dim, size_t _thread_dim>
void batched_stride_matmul_kernel_launch(
	float* o, const float* a, const float* b,
	const size_t* a_stride, const size_t* b_stride,
	size_t batch, size_t rows, size_t inner, size_t cols,
	bool b_batched, size_t b_batch_group)
{
	shark::rcheck<hipError_t, hipError_t::hipSuccess> ck;
	constexpr size_t TILE = _block_dim * _thread_dim;
	dim3 block(_block_dim, _block_dim);
	dim3 grid(
		(cols + TILE - 1) / TILE,
			  (rows + TILE - 1) / TILE,
			  batch
	);
	
	batched_stride_matmul_kernel<_block_dim, _thread_dim><<<grid, block>>>
		(o, a, b, a_stride, b_stride,
		 batch, rows, inner, cols, b_batched, b_batch_group);
	
	ck = hipGetLastError();
	ck = hipDeviceSynchronize();
}


template void transpose_tiled_kernel_launch<32>(
	float*, const float*, size_t, size_t
);


template void mat_softmax_kernel_launch<false>(
	const float*, float*, size_t, size_t
);


template void mat_softmax_kernel_launch<true>(
	const float*, float*, size_t, size_t
);


template void batched_stride_matmul_kernel_launch<16, 2>(
	float*, const float*, const float*,
	const size_t*, const size_t*,
	size_t, size_t, size_t, size_t, bool, size_t
);


float bench_kernel(
	const std::string& name, std::function<void()> launcher,
	int warmup, int repeats)
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


}
