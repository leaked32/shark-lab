import math
import inspect
from typing import cast, Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass
from shared.util import apply_rotary
# acts: activations



class KVCache:
	def __init__(self):
		self.k: Tensor | None = None
		self.v: Tensor | None = None

@dataclass
class DecoderState:
	layer_id: int
	kv_cache: list[KVCache] | None = None


class INorm(Protocol):
	def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
	
class IAttention(Protocol):
	def __call__(self, x: torch.Tensor, state: DecoderState) -> torch.Tensor: ...
	
class Norm0(nn.Module):
	""" Super Neko Nyan: The trend now is LayerNorm -> RMSNorm """
	def __init__(self, chan: int, eps: float, bias: bool = False):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(chan))
		self.bias = nn.Parameter(torch.zeros(chan)) if bias else None
		self.eps = eps
		
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)

class Norm1(nn.Module):
	""" Implemented as RMSNorm instead """
	def __init__(self, chan: int, eps: float):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(chan))
		self.eps = eps
	
	def forward(self, acts: Tensor) -> Tensor:
		return F.rms_norm(acts, self.weight.shape, self.weight, self.eps)


class Attention2(nn.Module):
	""" Causal Multi-head Self-Attention
					by Chumbud
	"""
	def __init__(self, chan: int, q_head: int, kv_head: int, drop: float, sdpa: bool = True, rope_base: float = 10000.0):
		super().__init__()
		assert chan % kv_head == 0
		assert chan % q_head == 0
		assert q_head % kv_head == 0
		assert (chan // q_head) % 2 == 0 # RoPE requires even head_dim
		
		self.q_proj = nn.Linear(chan, chan, False)
		self.k_proj = nn.Linear(chan, (chan // (q_head // kv_head)), False)
		self.v_proj = nn.Linear(chan, (chan // (q_head // kv_head)), False)
		self.o_proj = nn.Linear(chan, chan, False)
		
		# Deprecated, never used since F.scaled_dot_product_attention
		self.attn_dropout = nn.Dropout(drop) # Compatibility
		self.residual_dropout = nn.Dropout(drop)
		
		self.chan = chan
		self.kv_head = kv_head
		self.q_head = q_head
		self.drop = drop
		self.rope_base = rope_base
		self.sdpa = sdpa
		
		self.init_rope_cache()
		self.register_buffer("causal_mask_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)
		
	def get_causal_mask(self, start_pos: int, q_len: int, kv_len: int, device) -> Tensor:
		need = max(start_pos + q_len, kv_len)
		old = self.causal_mask_cache.size(0)

		if self.causal_mask_cache.device != device or old < need:
			new_size = max(need, old * 2 if old > 0 else 16)
			pos_q = torch.arange(new_size, device=device)[:, None]
			pos_k = torch.arange(new_size, device=device)[None, :]
			self.causal_mask_cache = pos_k <= pos_q

		return self.causal_mask_cache[start_pos:start_pos + q_len, :kv_len].view(1, 1, q_len, kv_len)
		
	def init_rope_cache(self):
		head_dim = self.chan // self.q_head
		inv_freq = 1.0 / (
			# Explicit: self.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
			self.rope_base ** (torch.arange(0, head_dim, 2) / head_dim)
		)
		self.register_buffer("inv_freq", inv_freq)
		
	
	def get_rope_cache(self, start_pos: int, seq_len: int, device):
		# Explicit: pos = torch.arange(start_pos, start_pos + seq_len, device=device).float()
		pos = torch.arange(start_pos, start_pos + seq_len, device=device)
		freqs = torch.outer(pos, self.get_buffer("inv_freq"))
		emb = torch.cat((freqs, freqs), dim=-1)

		cos = emb.cos()[None, None, :, :]
		sin = emb.sin()[None, None, :, :]

		return cos, sin
	
	def forward(self, acts: Tensor, state: DecoderState) -> Tensor:
		""" TODO This is configured for training.
			For inference use, KV cache is considered necessary
		"""
		sz_batch, sz_seq, sz_embd = acts.size()
		assert sz_embd == self.chan
		
		q: Tensor = self.q_proj(acts)
		k: Tensor = self.k_proj(acts)
		v: Tensor = self.v_proj(acts)
		
		head_dim = self.chan // self.q_head

		q = q.view(sz_batch, sz_seq, self.q_head, head_dim).transpose(1, 2)
		k = k.view(sz_batch, sz_seq, self.kv_head, head_dim).transpose(1, 2)
		v = v.view(sz_batch, sz_seq, self.kv_head, head_dim).transpose(1, 2)
		
		# print(q.dtype, k.dtype, v.dtype)
		
		cache = None if state.kv_cache is None else state.kv_cache[state.layer_id]
		
		if cache is not None and cache.k is not None:
			start_pos = cast(Tensor, cache.k).size(2)
		else:
			start_pos = 0
		
		cos, sin = self.get_rope_cache(start_pos, sz_seq, acts.device)
		# print(q.dtype, k.dtype, v.dtype, cos.dtype, sin.dtype)
		q, k = apply_rotary(q, k, cos, sin)
		
		# print(q.dtype, k.dtype, v.dtype)
		if cache is not None:
			# print(cache)
			if cache.k is None or cache.v is None:
				cache.k = k
				cache.v = v
			else:
				cache.k = torch.cat((cache.k, k), dim=2)
				cache.v = torch.cat((cache.v, v), dim=2)

			k = cache.k
			v = cache.v
		
		# print(q.dtype, k.dtype, v.dtype)
		# PyTorch SDPA supports KV-cache style (Q_len != KV_len) causal attention.
		
		group_size = self.q_head // self.kv_head
		q_len = q.size(2)
		kv_len = k.size(2)
		
		causal_mask: Tensor | None
		is_causal: bool
		if cache is None:
			causal_mask = None
			is_causal = True
		elif q_len == 1:
			causal_mask = None
			is_causal = False
		else:
			causal_mask = self.get_causal_mask(start_pos, q_len, kv_len, acts.device)
			is_causal = False
		# KV-cache issue: q_len != kv_len, needs offset causal mask
		# GQA issue: q_head != kv_head, needs enable_gqa=True or repeat_interleave
		if self.sdpa:
			# no repeat_interleave
			y = F.scaled_dot_product_attention(
				q, k, v,
				attn_mask=causal_mask,
				is_causal=is_causal,
				dropout_p=self.drop if self.training else 0.0,
				enable_gqa=True
			)
			"""	Typically, it chooses among:
				
				FlashAttention kernel (fastest, when supported)
				Memory-efficient attention (also fused, but not FlashAttention)
				Plain math implementation (fallback)
			"""
		else:
			outs = []
			for kv_idx in range(self.kv_head):
				qg = q[:, kv_idx * group_size: (kv_idx + 1) * group_size]
				kg = k[:, kv_idx: kv_idx + 1]
				vg = v[:, kv_idx: kv_idx + 1]

				att = (qg @ kg.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
				if causal_mask is not None:
					att = att.masked_fill(~causal_mask, float("-inf"))
				elif is_causal:
					manual_mask = self.get_causal_mask(0, q_len, kv_len, acts.device)
					att = att.masked_fill(~manual_mask, float("-inf"))
				att = F.softmax(att, dim=-1)
				att = self.attn_dropout(att)
				outs.append(att @ vg)

			y = torch.cat(outs, dim=1)
		
		y = y.transpose(1, 2).contiguous().view(sz_batch, sz_seq, sz_embd)
		y = self.o_proj(y)
		y = self.residual_dropout(y)
		return y


class FeedForward(nn.Module):
	def __init__(self, chan: int, drop: float, mlp_mul: int):
		super().__init__()
		hidden = chan * mlp_mul
		self.c_fc1 = nn.Linear(chan, hidden, False)
		self.c_fc2 = nn.Linear(chan, hidden, False)
		self.silu = nn.SiLU()
		self.c_proj = nn.Linear(hidden, chan, False)
		self.residual_dropout = nn.Dropout(drop)

	def forward(self, acts: Tensor, state: DecoderState) -> Tensor:
		x1 = self.c_fc1(acts)
		x2 = self.c_fc2(acts)
		x = self.silu(x1) * x2
		return self.residual_dropout(self.c_proj(x))


class TransformerBlock(nn.Module):
	def __init__(self, norm: INorm, norm1: INorm, attn: IAttention, chan: int, drop: float, mlp_mul: int):
		super().__init__()
		self.ln_1 = norm
		# self.attn = Attention(chan, head, drop) deprecated
		self.attn = attn
		self.ln_2 = norm1
		self.mlp = FeedForward(chan, drop, mlp_mul)
	
	def forward(self, acts, state: DecoderState):
		acts = acts + self.attn(self.ln_1(acts), state)
		acts = acts + self.mlp(self.ln_2(acts), state)
		return acts

@dataclass
class GPTOption:
	vocab: int
	layer: int
	chan: int
	q_head: int
	mlp_mul: int
	drop: float
	eps: float
	kv_head: int # deprecated for GQA is not well-implemented yet.
	# bias: bool = False # not supported yet
	# rope_base: float = 10000.0 # not supported yet

class GPT(nn.Module):
	def __init__(self, opt: GPTOption):
		super().__init__()
		assert opt.layer != None
		assert opt.vocab != None
		
		self.opt = opt
		
		self.transformer = nn.ModuleDict(
			dict(
				wte=nn.Embedding(opt.vocab, opt.chan),
				drop=nn.Dropout(opt.drop),
				h=nn.ModuleList(
					[TransformerBlock(
						Norm1(opt.chan, opt.eps), Norm1(opt.chan, opt.eps),
						Attention2(opt.chan, opt.q_head, opt.kv_head, opt.drop), 
						opt.chan, opt.drop, opt.mlp_mul) for _ in range(opt.layer)]),
				ln_f=Norm1(opt.chan, opt.eps)
			)
		)
		self.lm_head = nn.Linear(opt.chan, opt.vocab, bias=False)
		cast(nn.Embedding, self.transformer.wte).weight = self.lm_head.weight
		
		self.apply(self._init_weights)
		
		# apply special scaled init to the residual projections, per GPT-2 paper
		for pn, p in self.named_parameters():
			if pn.endswith("c_proj.weight"):
				torch.nn.init.normal_(
					p, mean=0.0, std=0.02 / math.sqrt(2 * opt.layer)
				)
		print(f"number of parameters: {(self.get_num_params() / 1e6):.2f}M")
		
	def get_num_params(self, exclude_embeddings=True):
		# Many people have extremely weird preferences that they don't want embeddings are counted
		n_params = sum(p.numel() for p in self.parameters())
		if exclude_embeddings:
			n_params -= cast(nn.Embedding, self.transformer.wte).weight.numel()
		return n_params

	def _init_weights(self, module: Any):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
	
	def optimizer_adamw(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str) -> torch.optim.AdamW:
		params = {pn: p for pn, p in self.named_parameters()}
		# filter all parameters which requires gradients
		params = [p for _, p in  params.items() if p.requires_grad]
		
		# separate parameters by if their dim() less than 2 or not 
		decay_params = [p for p in params if p.dim() >= 2]
		nodecay_params = [p for p in params if p.dim() < 2]
		optim_groups = [
			{"params": decay_params, "weight_decay": weight_decay},
			{"params": nodecay_params, "weight_decay": 0.0},
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(
			f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
		)
		print(
			f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
		)
		# create AdamW optimizer and use the fused version if it is available
		fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == "cuda"
		extra_args = dict(fused=True) if use_fused else dict()
		print(f'learning_rate={learning_rate}')
		optimizer = torch.optim.AdamW(
			optim_groups, lr=learning_rate, betas=betas, **extra_args
		)
		print(f"using fused AdamW: {use_fused}")
		return optimizer
	
	# DecoderState(0) is created once, then reused forever.
	# If any field of state changes (and you do mutate layer_id), every future call shares that same object.
	def forward(self, input: Tensor, targets: Tensor | None = None, state: DecoderState | None = None) -> tuple[Tensor, Tensor | None]:
		if state is None:
			state = DecoderState(0)
		assert not (self.training and state.kv_cache is not None)
		
		# predicts = self(input)
		tok_emb: Tensor = cast(nn.Embedding, self.transformer.wte)(input)
		x: Tensor = cast(nn.Embedding, self.transformer.drop)(tok_emb)
		
		for i in range(len(cast(nn.ModuleList, self.transformer.h))):
			# or block in cast(nn.ModuleList, self.transformer.h):
			state.layer_id = i
			x = cast(nn.ModuleList, self.transformer.h)[i](x, state)
			
		x = cast(Norm1, self.transformer.ln_f)(x)
		logits: torch.Tensor
		if targets is not None:
			logits = self.lm_head(x)
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
				)
		else:
			logits = self.lm_head(x[:, -1, :])
			loss = None
		return logits, loss
		
	
	@torch.no_grad
	def generate(self, idx, max_new_tokens: int, temperature: float=1.0, top_k=None) -> Tensor:
		state = DecoderState(0, [KVCache() for _ in range(self.opt.layer)])
		
		for _ in range(max_new_tokens):

			if state.kv_cache is not None and state.kv_cache[0].k is None:
				idx_cond = idx          # first pass
			else:
				idx_cond = idx[:, -1:]  # only newest token
			
			logits, _ = self(idx_cond, None, state)
			logits = logits / temperature
			if top_k is not None:
				# Sorting the logits is very expensive, instead, we try to find the
				#   ones with biggest logits. (Partial sort)
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				# select all the logits below the min(topest top_k) and set them to
				#   -float("Inf"): $-\infty$
				logits[logits < v[:, [-1]]] = -float("Inf")
			probs = F.softmax(logits, dim=-1)
			# Randomly choose one from the probs table to be the next token.
			idx_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
		return idx
		



