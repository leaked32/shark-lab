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

#include "mhip.hpp"

#include "shark/shark.hpp"

namespace shark
{
std::integral_constant<float, 2e-5F> VERSION;


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
		mhip_malloc(&rptr_, rbytes());
	}
	*/
	
	const size_t rbytes() const {
		return size() * sizeof(value_type);	
	}
	
	const size_t size() const {
		return data_.size();
	}
	
	void sync_HD() { 
		mhip_memcpy_HD(rptr_, data_.data(), rbytes());
	}
	void sync_DH() {
		mhip_memcpy_DH(data_.data(), rptr_, rbytes());
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
		mhip_malloc(&rptr_, rbytes());
	}
	void r_dealloc() {
		if (rptr_) {
			mhip_free(rptr_);
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
		
		rmsnorm_kernel_launch(o_ptr, x_ptr, w_ptr, rows, cols, eps);
		
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
		contiguous_kernel_launch(
			out.rptr(), rptr(), shape_.rptr(), stride_.rptr(), dim(), numel()
		);
		
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
		
		transpose_tiled_kernel_launch<TILE>(o.rptr(), rptr(), rows, cols);
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
		tensor_mul_scalar_kernel_launch(rptr(), value, n);
	}
	
	template<bool unfinished_softmax = false>
	LittleTensor softmax() const
	{
		LittleTensor o(shape());
		
		mat_softmax_kernel_launch<unfinished_softmax>(rptr(), o.rptr(), size<-2>(), size<-1>());
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
};

template<int _block_dim = 16, int _thread_dim = 2>
LittleTensor matmul(const LittleTensor& a, const LittleTensor& b)
{
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
	
	bool b_batched = b.dim() > 2;
	size_t b_batch = 1;
	for (size_t i = 0; i < b.dim() - 2; ++i) {
		b_batch *= b.shape()[i];
	}
	size_t b_batch_group = batch / b_batch;
	batched_stride_matmul_kernel_launch<_block_dim, _thread_dim>
		(C.rptr(), a.rptr(), b.rptr(), a.stride_ptr(), b.stride_ptr(),
		 batch, rows, inner, cols, b_batched, b_batch_group);
	
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
			// auto& state = states_[p];
			
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
					
					matadd_kernel_launch(
						result->rptr(), A.rptr(), B.rptr(),
											   static_cast<int>(n)
					);
					
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
