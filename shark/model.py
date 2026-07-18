"""
shark-lab
shared/model.py

The module defines the model architecture
"""

import math
import inspect
from typing import cast, Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass
# acts: activations

@dataclass
class GPTOption:
	"""	NOTICE
		This dataclass should not be changed once the model
	"""
	vocab: int
	layer: int
	chan: int
	q_head: int
	mlp_mul: int
	drop: float
	eps: float
	kv_head: int
	rope_theta: float# = 10000.0
	# bias: bool = False # not supported yet
	
class RoPE():
	"""
	Meta's implementation and Hugging Face's implementation both keep
	the RoPE cache generation in float32, then cast to the model dtype when using it.
	That gives stable phases while keeping attention fast.
	`inv_freq: float32`
	`pos: float32`
	`freqs: float32`
	`cos/sin: float32`
	before apply_rotary: cos = cos.to(q.dtype) and sin = sin.to(q.dtype)
	"""
	
	@staticmethod
	def rotate_half(x: Tensor):
		x1 = x[..., : x.shape[-1] // 2]
		x2 = x[..., x.shape[-1] // 2 :]
		return torch.cat((-x2, x1), dim=-1)
	
	@staticmethod
	def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
		"""	In this implementation, q has nothing to do with k if dtype and device are the same.
		"""
		cos = cos.to(dtype=q.dtype, device=q.device)
		sin = sin.to(dtype=q.dtype, device=q.device)
		q = q * cos + RoPE.rotate_half(q) * sin
		k = k * cos + RoPE.rotate_half(k) * sin
		return q, k
	
	def __init__(self, head_dim: int, rope_theta: float):
		assert head_dim % 2 == 0
		
		# super().__init__()
		
		# def init_rope_cache(self) -> None:
		#	head_dim = self.chan // self.q_head
		self.inv_freq = (
			1.0 /
			(rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
		)
		# self.register_buffer("inv_freq", inv_freq, persistent=False)
	
	def get_rope_cache(self, start_pos: int, seq_len: int, device):
		# Explicit: pos = torch.arange(start_pos, start_pos + seq_len, device=device).float()
		inv_freq: Tensor = self.inv_freq.to(device=device)
		pos = torch.arange(start_pos, start_pos + seq_len, dtype=inv_freq.dtype, device=device)
		freqs = torch.outer(pos, inv_freq)
		emb = torch.cat((freqs, freqs), dim=-1)

		cos = emb.cos()[None, None, :, :]
		sin = emb.sin()[None, None, :, :]
		return cos, sin


class IKVCache(Protocol):
	def clear_slot(self, slot: int) -> None: ...

	def in_forward(self, q: Tensor, k: Tensor, v: Tensor,
			acts: Tensor, rope: RoPE, active_slots: Tensor
			) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

class INorm(Protocol):
	def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

class DecoderState:
	def __init__(self, kv_cache: list[IKVCache] | None = None,
			  active_slots: Tensor | None = None):
		self.kv_cache: list[IKVCache] | None = kv_cache
		self.active_slots: Tensor | None = active_slots
		"""	The mappings
			For example, `active_slots: tensor([1, 0])` means
			`row 0 -> slot 1` and `row 1 -> slot 0`
		"""
	def get_active_slots(self, sz_batch: int, device) -> Tensor:
		"""	If forward is called without it,
			like simple generate function, the program will be falsely asserted.
		"""
		if self.active_slots is None:
			y = torch.arange(sz_batch, dtype=torch.long, device=device)
			# assert False
		else:
			y = self.active_slots.to(device=device, dtype=torch.long)
		return y

class IAttention(Protocol):
	def __call__(self, x: torch.Tensor, state: DecoderState) -> torch.Tensor: ...
	
class KVCache0:
	"""	Implemented as contiguous KV Cache
	"""
	def __init__(self):
		self.k: Tensor | None = None
		self.v: Tensor | None = None
	
	def in_forward(self, q: Tensor, k: Tensor, v: Tensor,
			acts: Tensor, rope: RoPE, _):
		_, sz_seq, _ = acts.size()
		
		if self.k is not None:
			start_pos = cast(Tensor, self.k).size(2)
		else:
			start_pos = 0
		
		cos, sin = rope.get_rope_cache(start_pos, sz_seq, acts.device)
		# print(q.dtype, k.dtype, v.dtype, cos.dtype, sin.dtype)
		q, k = RoPE.apply_rotary(q, k, cos, sin)
		
		if self.k is None or self.v is None:
			self.k = k
			self.v = v
		else:
			self.k = torch.cat((self.k, k), dim=2)
			self.v = torch.cat((self.v, v), dim=2)

		# k = self.k
		# v = self.v
		return k, v
	
class KVCache1:
	"""	Implemented as slot KV Cache
		slot-based KV cache preallocates a larger tensor than the actual sequences currently need.
		this Tensor is used to determine what current writing position is.
		That said, in KV Cache, this represents the positions of true position of K and V.
		RoPE requires the current positions
		Right now every layer has its own lengths tensor,
		but the sequence length of a request is identical across all transformer layers.
		NOTE all seq in one batch shares the same capacity since the storage is still
		contiguous.
	"""
	def __init__(self, max_batch: int):
		self.k: Tensor | None = None
		self.v: Tensor | None = None
		self.max_batch = max_batch
		self.lengths = torch.zeros(max_batch, dtype=torch.long)
		# self.active = torch.zeros(max_batch, dtype=torch.bool)
	
	def clear_slot(self, slot: int):
		self.lengths[slot] = 0
		# self.active[slot] = False
	
	def in_forward(self, q: Tensor, k: Tensor, v: Tensor,
			acts: Tensor, rope: RoPE, active_slots: Tensor):
		# Ensure fetched Tensor is on the same device, or the program may crash for its sake.
		sz_batch, sz_seq, _ = acts.size()
		
		assert k.size(1) == v.size(1)
		assert active_slots.dtype == torch.long
		assert active_slots.device == acts.device
		assert active_slots.numel() == sz_batch
		assert int(active_slots.min().item()) >= 0
		assert int(active_slots.max().item()) < self.max_batch
		assert active_slots.unique().numel() == active_slots.numel()
		
		kv_head = k.size(1)
		head_dim = q.size(-1)
		
		if self.lengths.device != acts.device:
			self.lengths = self.lengths.to(acts.device)
		
		start_pos = self.lengths.index_select(0, active_slots).clone()
		"""	Please see the comments of KVCache::lengths
			it makes a start_pos (we called length in simple KV Cache) mappings.
			For example, suppose
			`state.active_slots: tensor([2, 0])` and `acts.shape: [2, 1, chan]`
			`start_pos[0] = cache.lengths[2]`
			`start_pos[1] = cache.lengths[0]`
		"""
		for row in range(sz_batch):
			pos = int(start_pos[row].item())
			cos, sin = rope.get_rope_cache(pos, sz_seq, acts.device)
			q[row:row + 1], k[row:row + 1] = (
				RoPE.apply_rotary(q[row:row + 1], k[row:row + 1], cos, sin)
			)
		
		least_cap = int((start_pos + sz_seq).max().item())
		older_cap = 0 if self.k is None else self.k.size(2)
		
		if self.k is None or self.v is None:
			# Initialization
			new_cap = max(least_cap, 16)
			self.k = k.new_zeros(
				(self.max_batch, kv_head, new_cap, head_dim))
			self.v = v.new_zeros(
				(self.max_batch, kv_head, new_cap, head_dim))
		elif older_cap < least_cap:
			# Auto growth with pre-allocation
			new_cap = max(least_cap, older_cap * 2)
			extra = new_cap - older_cap
			self.k = torch.cat((self.k,
				self.k.new_zeros(
					(self.max_batch, kv_head, extra, head_dim))
				), dim=2)
			self.v = torch.cat((self.v,
				self.v.new_zeros(
					(self.max_batch, kv_head, extra, head_dim))
				), dim=2)
		
		for row, slot_tensor in enumerate(active_slots):
			slot = int(slot_tensor.item())
			length = int(start_pos[row].item())
			self.k[slot, :, length:length + sz_seq] = k[row]
			self.v[slot, :, length:length + sz_seq] = v[row]
			self.lengths[slot] = length + sz_seq
			# cache.active[slot] = True
		
		lengths = self.lengths.index_select(0, active_slots)
		kv_len = int(lengths.max().item())
		k = self.k.index_select(0, active_slots)[:, :, :kv_len, :]
		v = self.v.index_select(0, active_slots)[:, :, :kv_len, :]
		
		key_pos = torch.arange(kv_len, device=acts.device)
		query_pos = (
			start_pos[:, None] +
			torch.arange(sz_seq, device=acts.device)[None, :]
			)
		valid_keys = key_pos[None, None, None, :] < lengths[:, None, None, None]
		"""	Element-wise comparison. Suppose, 
			key_pos = tensor([0, 1, 2, 3, 4])
			lengths = tensor([2, 4])
			lengths[:, None, None, None]
			[ [[[2]]], [[[4]]] ]
			key_pos[None, None, None, :]
			[[[[0, 1, 2, 3, 4]]]]
		"""
		causal = key_pos[None, None, None, :] <= query_pos[:, None, :, None]
		attention_mask = valid_keys & causal
		
		return q, k, v, attention_mask

class KVCache2:
	"""	Implemented as Paged KV Cache
	"""
	def __init__(self, max_batch: int, num_blocks: int, block_size: int):
		self.k_pool: Tensor | None = None
		self.v_pool: Tensor | None = None
		self.max_batch = max_batch
		self.num_blocks = num_blocks
		self.block_size = block_size
		self.lengths = torch.zeros(max_batch, dtype=torch.long)
		self.block_tables: list[list[int]] = [[] for _ in range(max_batch)]
		self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
	
	def _init_pool(self, k: Tensor):
		_, kv_head, _, head_dim = k.shape
		self.k_pool = k.new_empty((self.num_blocks, kv_head, self.block_size, head_dim))
		self.v_pool = k.new_empty((self.num_blocks, kv_head, self.block_size, head_dim))
	
	def _alloc_block(self) -> int:
		if not self.free_blocks:
			raise RuntimeError("Paged KV cache is out of physical blocks")
		return self.free_blocks.pop()
	
	def clear_slot(self, slot: int):
		self.free_blocks.extend(self.block_tables[slot])
		self.block_tables[slot].clear()
		self.lengths[slot] = 0
	
	def append(self, active_slots: Tensor, k: Tensor, v: Tensor):
		if self.k_pool is None or self.v_pool is None:
			self._init_pool(k)
		
		assert self.k_pool is not None
		assert self.v_pool is not None
		
		sz_batch, _, sz_seq, _ = k.shape
		assert active_slots.numel() == sz_batch
		
		if self.lengths.device != k.device:
			self.lengths = self.lengths.to(k.device)
		
		for row, slot_tensor in enumerate(active_slots):
			slot = int(slot_tensor.item())
			pos = int(self.lengths[slot].item())
			
			# In auto-regressive generation, we usually only append 1 unit of k and v
			# So it usually doesn't decrease the performance
			for t in range(sz_seq):
				logical_pos = pos + t
				logical_block = logical_pos // self.block_size
				offset = logical_pos % self.block_size
				
				while len(self.block_tables[slot]) <= logical_block:
					self.block_tables[slot].append(self._alloc_block())
				
				physical_block = self.block_tables[slot][logical_block]
				self.k_pool[physical_block, :, offset, :] = k[row, :, t, :]
				self.v_pool[physical_block, :, offset, :] = v[row, :, t, :]
				
			self.lengths[slot] = pos + sz_seq
		
	def gather(self, active_slots: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		assert self.k_pool is not None
		assert self.v_pool is not None
		
		active_slots = active_slots.to(device=self.lengths.device, dtype=torch.long)
		lengths = self.lengths.index_select(0, active_slots)
		kv_len = int(lengths.max().item())
		
		sz_batch = active_slots.numel()
		_, kv_head, _, head_dim = self.k_pool.shape
		k_dense = self.k_pool.new_empty((sz_batch, kv_head, kv_len, head_dim))
		v_dense = self.v_pool.new_empty((sz_batch, kv_head, kv_len, head_dim))
		
		for row, slot_tensor in enumerate(active_slots):
			slot = int(slot_tensor.item())
			length = int(lengths[row].item())
			write_pos = 0
			
			for physical_block in self.block_tables[slot]:
				if write_pos >= length:
					break
				take = min(self.block_size, length - write_pos)
				k_dense[row, :, write_pos:write_pos + take, :] = (
					self.k_pool[physical_block, :, :take, :]
					)
				v_dense[row, :, write_pos:write_pos + take, :] = (
					self.v_pool[physical_block, :, :take, :]
					)
				write_pos += take

		return k_dense, v_dense, lengths
		
	def in_forward(self, q: Tensor, k: Tensor, v: Tensor,
			acts: Tensor, rope: RoPE, active_slots: Tensor):
		"""	In pure PyTorch, KVCache1 will likely be faster than KVCache2,
			because PyTorch is good at dense tensor ops,
			but bad at Python loops over blocks/tokens.
			
			I need attention that consumes them directly, without dense gathering.
			But that usually requires a custom CUDA/Triton kernel,
			not ordinary Python-level PyTorch.
		"""
		
		# Ensure fetched Tensor is on the same device, or the program may crash for its sake.
		sz_batch, sz_seq, _ = acts.size()
		
		assert k.size(1) == v.size(1)
		assert active_slots.dtype == torch.long
		assert active_slots.device == acts.device
		assert active_slots.numel() == sz_batch
		assert int(active_slots.min().item()) >= 0
		assert int(active_slots.max().item()) < self.max_batch
		assert active_slots.unique().numel() == active_slots.numel()
		
		if self.lengths.device != acts.device:
			self.lengths = self.lengths.to(acts.device)
		
		start_pos = self.lengths.index_select(0, active_slots).clone()
		
		for row in range(sz_batch):
			pos = int(start_pos[row].item())
			cos, sin = rope.get_rope_cache(pos, sz_seq, acts.device)
			q[row:row + 1], k[row:row + 1] = (
				RoPE.apply_rotary(q[row:row + 1], k[row:row + 1], cos, sin)
			)
		
		self.append(active_slots, k, v)
		k, v, lengths = self.gather(active_slots)
		kv_len = k.size(2)
		key_pos = torch.arange(kv_len, device=acts.device)
		query_pos = start_pos[:, None] + torch.arange(sz_seq, device=acts.device)[None, :]

		valid_keys = key_pos[None, None, None, :] < lengths[:, None, None, None]
		causal = key_pos[None, None, None, :] <= query_pos[:, None, :, None]
		attention_mask = valid_keys & causal
		
		return q, k, v, attention_mask

class Norm0(nn.Module):
	""" The trend now is LayerNorm -> RMSNorm """
	def __init__(self, chan: int, eps: float, bias: bool = False):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(chan))
		self.bias = nn.Parameter(torch.zeros(chan)) if bias else None
		self.eps = eps
		
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return F.layer_norm(
			input, self.weight.shape, self.weight, self.bias, self.eps)

class Norm1(nn.Module):
	""" Implemented as RMSNorm instead """
	def __init__(self, chan: int, eps: float):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(chan))
		self.eps = eps
	
	def forward(self, acts: Tensor) -> Tensor:
		return F.rms_norm(acts, self.weight.shape, self.weight, self.eps)


class Attention2(nn.Module):
	"""	Implemented as Causal Multi-head Self-Attention
		Featured: Grouped Query Attention
	"""
	def __init__(self, opt: GPTOption, layer_id: int, sdpa: bool=True):
		super().__init__()
		assert opt.chan % opt.kv_head == 0
		assert opt.chan % opt.q_head == 0
		assert opt.q_head % opt.kv_head == 0
		assert (opt.chan // opt.q_head) % 2 == 0 # RoPE requires even head_dim
		
		self.chan = opt.chan
		self.kv_head = opt.kv_head
		self.q_head = opt.q_head
		self.drop = opt.drop
		self.rope_theta = opt.rope_theta
		self.sdpa = sdpa
		self.layer_id = layer_id
		
		self.q_proj = nn.Linear(self.chan, self.chan, False)
		self.k_proj = nn.Linear(self.chan, (self.chan // (self.q_head // self.kv_head)), False)
		self.v_proj = nn.Linear(self.chan, (self.chan // (self.q_head // self.kv_head)), False)
		self.o_proj = nn.Linear(self.chan, self.chan, False)
		
		# Deprecated, never used since F.scaled_dot_product_attention
		self.attn_dropout = nn.Dropout(self.drop) # Compatibility
		self.residual_dropout = nn.Dropout(self.drop)
		
		self.rope = RoPE(self.chan // self.q_head, self.rope_theta)
		self.register_buffer("causal_mask_cache",
			torch.empty(0, 0, dtype=torch.bool), persistent=False)
	
	def get_causal_mask(self, start_pos: int, q_len: int, kv_len: int, device) -> Tensor:
		need = max(start_pos + q_len, kv_len)
		old = self.causal_mask_cache.size(0)
		
		if self.causal_mask_cache.device != device or old < need:
			new_size = max(need, old * 2 if old > 0 else 16)
			pos_q = torch.arange(new_size, device=device)[:, None]
			pos_k = torch.arange(new_size, device=device)[None, :]
			self.causal_mask_cache = pos_k <= pos_q

		return (
			self.causal_mask_cache[start_pos:start_pos + q_len, :kv_len]
			.view(1, 1, q_len, kv_len)
		)
	
	def forward(self, acts: Tensor, state: DecoderState) -> Tensor:
		"""Forward pass for training, static generation, and slot-based continuous batching.
		"""
		sz_batch, sz_seq, sz_embd = acts.size()
		assert sz_embd == self.chan

		q: Tensor = self.q_proj(acts)
		k: Tensor = self.k_proj(acts)
		v: Tensor = self.v_proj(acts)
		
		head_dim: int = self.chan // self.q_head
		q = q.view(sz_batch, sz_seq, self.q_head, head_dim).transpose(1, 2)
		k = k.view(sz_batch, sz_seq, self.kv_head, head_dim).transpose(1, 2)
		v = v.view(sz_batch, sz_seq, self.kv_head, head_dim).transpose(1, 2)
		
		cache = None if state.kv_cache is None else state.kv_cache[self.layer_id]
		causal_mask: Tensor | None
		is_causal: bool
		
		if cache is None:
			# No KV Cache; Training
			cos, sin = self.rope.get_rope_cache(0, sz_seq, acts.device)
			q, k = RoPE.apply_rotary(q, k, cos, sin)
			causal_mask = None
			is_causal = True
		else:
			active_slots = state.get_active_slots(sz_batch, acts.device)
			assert active_slots.numel() == sz_batch
			
			q, k, v, causal_mask = (
				cache.in_forward(q, k, v, acts, self.rope, active_slots)
			)
			
			is_causal = False
		
		if self.sdpa:
			# KV Cache issue: q_len != kv_len, needs offset causal mask
			# GQA issue: q_head != kv_head, needs `enable_gqa=True` or `repeat_interleave`
			y = F.scaled_dot_product_attention(
				q, k, v,
				attn_mask=causal_mask, is_causal=is_causal,
				dropout_p=self.drop if self.training else 0.0,
				enable_gqa=True
			)
			"""	Typically, it chooses among:
				
				FlashAttention kernel (fastest, when supported)
				Memory-efficient attention (also fused, but not FlashAttention)
				Plain math implementation (fallback)
			"""
		else:
			group_size = self.q_head // self.kv_head
			q_len = q.size(2)
			kv_len = k.size(2)
			
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


class IFeedForward(Protocol):
	def __init__(self, chan: int, drop: float, mlp_mul: int): ...
	def forward(self, acts: Tensor, state: DecoderState) -> Tensor: ...
	def __call__(self, acts: Tensor, state: DecoderState) -> Tensor: ...


class FeedForward0(nn.Module):
	""" MLP GELU """
	def __init__(self, chan: int, drop: float, mlp_mul: int):
		super().__init__()
		self.c_fc = nn.Linear(chan, mlp_mul, False)
		self.gelu = nn.GELU()
		self.c_proj = nn.Linear(mlp_mul, chan, False)
		self.residual_dropout = nn.Dropout(drop)
	
	def forward(self, acts: Tensor, state: DecoderState) -> Tensor:
		y = self.residual_dropout(self.c_proj(self.gelu(self.c_fc(acts))))
		return y


class FeedForward1(nn.Module):
	""" MLP SwiGLU """
	def __init__(self, chan: int, drop: float, mlp_mul: int):
		super().__init__()
		self.gate_proj = nn.Linear(chan, mlp_mul, False)
		self.up_proj = nn.Linear(chan, mlp_mul, False)
		self.act_fn = nn.SiLU()
		self.down_proj = nn.Linear(mlp_mul, chan, False)
		self.residual_dropout = nn.Dropout(drop)

	def forward(self, acts: Tensor, state: DecoderState) -> Tensor:
		x1 = self.gate_proj(acts)
		x2 = self.up_proj(acts)
		x = self.act_fn(x1) * x2
		return self.residual_dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
	def __init__(self, norm: INorm, norm1: INorm, attn: IAttention,
			mlp: IFeedForward):
			#chan: int, drop: float, mlp_mul: int):
		super().__init__()
		# self.attn = Attention(chan, head, drop) deprecated
		self.self_attn = attn
		self.input_layernorm = norm
		self.post_attention_layernorm = norm1
		self.mlp = mlp
		# self.mlp = FeedForward(chan, drop, mlp_mul)
	
	def forward(self, acts, state: DecoderState):
		acts = acts + self.self_attn(self.input_layernorm(acts), state)
		acts = acts + self.mlp(self.post_attention_layernorm(acts), state)
		return acts


class GPT(nn.Module):
	def __init__(self, opt: GPTOption):
		super().__init__()
		assert opt.layer != None
		assert opt.vocab != None
		
		self.opt = opt
		
		self.model = nn.ModuleDict(
			dict(
				embed_tokens=nn.Embedding(opt.vocab, opt.chan),
				layers=nn.ModuleList(
					[TransformerBlock(
						Norm1(opt.chan, opt.eps), Norm1(opt.chan, opt.eps),
						Attention2(opt, layer_id), 
						FeedForward1( opt.chan, opt.drop, opt.mlp_mul) )
						for layer_id in range(opt.layer)]),
				norm=Norm1(opt.chan, opt.eps)
			)
		)
					
		self.dropout = nn.Dropout(opt.drop)
		self.lm_head = nn.Linear(opt.chan, opt.vocab, bias=False)
		cast(nn.Embedding, self.model.embed_tokens).weight = self.lm_head.weight
		
		self.apply(self._init_weights)
		
		print(f"number of parameters: {(self.get_num_params() / 1e6):.2f}M")
	
	def get_num_params(self, exclude_embeddings=True):
		# Many people have extremely weird preferences that they don't want embeddings are counted
		n_params = sum(p.numel() for p in self.parameters())
		if exclude_embeddings:
			n_params -= cast(nn.Embedding, self.model.embed_tokens).weight.numel()
		return n_params
	
	def _init_weights(self, module: Any):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
	
	def optimizer_adamw(self, weight_decay: float, learning_rate: float,
			betas: tuple[float, float], device_type: str) -> torch.optim.AdamW:
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
			f"num decayed parameter tensors: {
				len(decay_params)}, with {num_decay_params:,} parameters"
		)
		print(
			f"num non-decayed parameter tensors: {
				len(nodecay_params)}, with {num_nodecay_params:,} parameters"
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
	
	def forward(self, input: Tensor, targets: Tensor | None = None,
			state: DecoderState | None = None) -> tuple[Tensor, Tensor | None]:
		"""	DecoderState(0) is created once, then reused forever.
			If any field of state changes (and you do mutate layer_id),
				every future call shares that same object.
		"""
		if state is None:
			state = DecoderState()
		assert not (self.training and state.kv_cache is not None)
		
		# predicts = self(input)
		tok_emb: Tensor = cast(nn.Embedding, self.model.embed_tokens)(input)
		x: Tensor = cast(nn.Dropout, self.dropout)(tok_emb)
		
		for i in range(len(cast(nn.ModuleList, self.model.layers))):
			# or block in cast(nn.ModuleList, self.model.layers):
			# sstate.layer_id = i
			x = cast(nn.ModuleList, self.model.layers)[i](x, state)
		
		x = cast(Norm1, self.model.norm)(x)
		logits: torch.Tensor
		if targets is not None:
			"""	Training
				teacher-forcing: Even if the model thinks the next token should be "dog"
				instead of "cat", we don't feed "dog" back into the next position.
				We still feed the groud-truth sequence from the dataset.
			"""	
			logits = self.lm_head(x)
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
				)
			"""	cross_entropy
			logits -> [softmax] -> probabilities -> [-log(probability of correct label)] ->
			"average over all prediction positions"
			"""
		else:
			# Inference doesn't produce loss at all
			logits = self.lm_head(x[:, -1, :])
			loss = None
		return logits, loss
	"""
	@torch.no_grad
	def generate(self, idx, max_new_tokens: int, temperature: float=1.0,
			top_k: int|None=None) -> Tensor:
		state = DecoderState(
		#	[KVCache1(idx.size(0)) for _ in range(self.opt.layer)],
			[KVCache2(idx.size(0), num_blocks=1024, block_size=16) for _ in range(self.opt.layer)],
			torch.arange(idx.size(0), dtype=torch.long, device=idx.device)
		)
		
		for step in range(max_new_tokens):
			idx_cond = idx if step == 0 else idx[:, -1:]
			logits, _ = self(idx_cond, None, state)
			idx_next = self.sample_next_token(logits, temperature, top_k)
			idx = torch.cat((idx, idx_next), dim=1)
		return idx
	"""
	
	@torch.no_grad()
	def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0,
		top_k: int | None = None, eos_token_id: int | None = None) -> Tensor:
		"""Static-batch generation reference implementation."""
		
		state = DecoderState(
		#	[KVCache1(idx.size(0)) for _ in range(self.opt.layer)],
			[KVCache2(idx.size(0), num_blocks=1024, block_size=16) for _ in range(self.opt.layer)],
			torch.arange(idx.size(0), dtype=torch.long, device=idx.device)
		)

		for step in range(max_new_tokens):
			idx_cond = idx if step == 0 else idx[:, -1:]

			logits, _ = self(idx_cond, None, state)
			idx_next = self.sample_next_token(logits, temperature, top_k)

			idx = torch.cat((idx, idx_next), dim=1)

			if (eos_token_id is not None and torch.all(idx_next == eos_token_id)):
				# eos_token_id has already been appended to the result
				break

		return idx
	
	@staticmethod
	def sample_next_token(logits: Tensor, temperature: float=1.0, top_k: int|None=None) -> Tensor:
		if not torch.isfinite(logits).all():
			bad = (~torch.isfinite(logits)).sum().item()
			raise RuntimeError(
				f"logits contain {bad} non-finite values; "
				f"min={torch.nan_to_num(logits).min().item()}, "
				f"max={torch.nan_to_num(logits).max().item()}"
			)
		logits = logits / temperature
		if top_k is not None:
			v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
			logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
		probs = F.softmax(logits, dim=-1)
		return torch.multinomial(probs, num_samples=1)
	
	def get_device(self):
		return next(self.parameters()).device

