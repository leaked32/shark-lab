#pragma once

#include <torch/torch.h>


namespace shark {
	
struct GPTOption {
	int64_t vocab;
	int64_t layer;
	int64_t chan;
	int64_t q_head;
	int64_t mlp_mul;
	double drop;
	double eps;
	int64_t kv_head;
	double rope_theta;
};

class RoPE {
	using Tensor = torch::Tensor;
	using Device = torch::Device;
	
public:
	RoPE(int64_t head_dim, double rope_theta);
	
	std::pair<Tensor, Tensor> get_rope_cache(
		int64_t start_pos,
		int64_t seq_len,
		const Device& device
	) const;
	
	static Tensor rotate_half(const Tensor& x);
	static std::pair<Tensor, Tensor> apply_rotary(
		const Tensor& q,
		const Tensor& k,
		const Tensor& cos,
		const Tensor& sin
	);
	
private:
	Tensor inv_freq_;
	
};

class RMSNormImpl : public torch::nn::Module {
	
public:
	RMSNormImpl(int64_t chan, double eps);
	torch::Tensor forward(const torch::Tensor& acts);

private:
	torch::Tensor weight_;
	double eps_;
};

TORCH_MODULE(RMSNorm);

}
