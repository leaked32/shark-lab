
#include "shark/model.hpp"

#include <torch/nn.h>
#include <torch/nn/functional.h>

namespace shark{

RoPE::RoPE(int64_t head_dim, double rope_theta) {
	TORCH_CHECK(head_dim % 2 == 0, "RoPE head dimension must be even");
	
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto dimensions = torch::arange(0, head_dim, 2, options);
	
	inv_freq_ = 1.0 / torch::pow(
		rope_theta,
		dimensions / static_cast<double>(head_dim)
	);
}


torch::Tensor RoPE::rotate_half(const torch::Tensor& x) {
	const int64_t half = x.size(-1) / 2;
	
	auto x1 = x.slice(-1, 0, half);
	auto x2 = x.slice(-1, half);
	
	return torch::cat({-x2, x1}, -1);
}

std::pair<torch::Tensor, torch::Tensor> RoPE::apply_rotary(
	const torch::Tensor& q,
	const torch::Tensor& k,
	const torch::Tensor& cos,
	const torch::Tensor& sin
) {
	auto typed_cos = cos.to(q.device(), q.scalar_type());
	auto typed_sin = sin.to(q.device(), q.scalar_type());
	
	auto rotated_q = q * typed_cos + rotate_half(q) * typed_sin;
	auto rotated_k = k * typed_cos + rotate_half(k) * typed_sin;
	
	return {rotated_q, rotated_k};
}

std::pair<torch::Tensor, torch::Tensor> RoPE::get_rope_cache(
	int64_t start_pos,
	int64_t seq_len,
	const torch::Device& device
) const {
	auto inv_freq = inv_freq_.to(device);
	
	auto positions = torch::arange(
		start_pos,
		start_pos + seq_len,
		torch::TensorOptions()
		.dtype(inv_freq.scalar_type())
		.device(device)
	);
	
	auto frequencies = torch::outer(positions, inv_freq);
	auto embedding = torch::cat({frequencies, frequencies}, -1);
	
	auto cos = embedding.cos().unsqueeze(0).unsqueeze(0);
	auto sin = embedding.sin().unsqueeze(0).unsqueeze(0);
	
	return {cos, sin};
}

RMSNormImpl::RMSNormImpl(int64_t chan, double eps)
: eps_(eps) {
	weight_ = register_parameter("weight", torch::ones({chan}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& acts) {
	auto variance = acts.to(torch::kFloat32).pow(2).mean(-1, true);
	auto normalized = acts * torch::rsqrt(variance + eps_);
	return normalized.to(acts.scalar_type()) * weight_;
}




}


