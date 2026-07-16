from typing import Any
from shared.util import notify_confirm

from shared.model import GPT, GPTOption
from dataclasses import dataclass
import tomllib
import os

import torch


def get_tokenizer_vocab_count(tokenizer_path: str) -> int:
	""" This function is only responsible to find the vocabulary count
		it's only used when Ya build a model from scratch~
		Hint: tokenizer_path should point to a directory where contains files
		like (special_tokens_map.json, tokenizer.json, tokenizer_config.json),
		you can find them in specilized tokenizer saved with
		tokenizer.save_pretrained(path) and directory of many models~
	"""
	from transformers import AutoTokenizer
	# TODO: tokenizer should be altered to my own tokenizer
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	# if tokenizer.pad_token is None:
	# 	tokenizer.pad_token = tokenizer.eos_token
	#	raise Exception(f"Invalid pad_token {tokenizer.eos_token}")

	# vocab = tokenizer.get_vocab()
	# id_to_token = {v: k for k, v in vocab.items()}
	return tokenizer.vocab_size
	


@dataclass
class trainer_options:
	model: dict[str, Any]
	train: dict[str, Any]

def model_from_scratch(opt: trainer_options) -> GPT:
	vocab_size = get_tokenizer_vocab_count(opt.train['tokenizer_path'])
	if vocab_size is None:
		print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
	
	opt_model = opt.model
	if opt_model['vocab'] == 0:
		print("since `opt_model['vocab']` is 0, using tokenizer vocabulary size by default")
		opt_model['vocab'] = vocab_size
	elif opt_model['vocab'] != vocab_size:
		notify_confirm(f"Mismatch: opt_model['vocab']: {opt_model['vocab']}, but vocal_size: {vocab_size}.\nContinue will overwrite opt_model['vocab'] with vocal_size from dataset. ")
		opt_model['vocab'] = vocab_size
	
	model_args = GPTOption(
		vocab=opt_model['vocab'],
		chan=opt_model['chan'],
		drop=opt_model['drop'],
		eps=opt_model['eps'],
		q_head=opt_model['q_head'],
		kv_head=opt_model['kv_head'],
		layer=opt_model['layer'],
		mlp_mul=opt_model['mlp_mul'],
		rope_theta=opt_model['rope_theta'],
		)
	return GPT(model_args)


def __optimizer_save_checkpoint(
		path: str, model: GPT, optimizer: torch.optim.AdamW, step: int) -> None:
	checkpoint = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'step': step,
	}
	print(f"saving checkpoint to {path}")
	torch.save(checkpoint, path)


def load_model_and_optimizer(
		meta_opt: dict[str, Any], ckpt_path: str | None, device: str):
	opt = trainer_options(meta_opt['model'], meta_opt['train'])
	model = model_from_scratch(opt)
	if model is None:
		raise RuntimeError('trainer.cased_model returned None')

	optimizer = model.optimizer_adamw(
		opt.train['adamw_weight_decay'], opt.train['optimizer_learning_rate'],
		(opt.train['adamw_beta1'], opt.train['adamw_beta2']), device
	)

	step = 0
	if ckpt_path is not None and os.path.exists(ckpt_path):
		model, optimizer, step = load_checkpoint(ckpt_path, model, optimizer)
	return model, optimizer, step


def save_checkpoint(path: str, model, optimizer, step: int) -> None:
	os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
	__optimizer_save_checkpoint(path, model, optimizer, step)


def load_checkpoint(
		ckpt_path: str, model: GPT, optimizer: torch.optim.AdamW, device: str = 'cpu'):
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
	
	checkpoint = torch.load(ckpt_path, map_location=device)

	model.load_state_dict(checkpoint['model'], strict=True)
	optimizer.load_state_dict(checkpoint['optimizer'])

	step: int = checkpoint.get('step', 0)

	print(f"resumed from step {step}")
	return model, optimizer, step


def load_model_checkpoint(model: GPT, ckpt_path: str, device: str = 'cpu') -> int:
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
	checkpoint = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(checkpoint['model'], strict=True)
	return int(checkpoint.get('step', 0))

def load_meta_dataset(path: str):
	with open(path, "rb") as f:
		return tomllib.load(f)

# shared/pretrained.py

from collections import OrderedDict
import torch


def convert_hf_llama_state_dict(
	state_dict: dict[str, torch.Tensor],
	prefix: str = "model."
) -> OrderedDict[str, torch.Tensor]:
	"""
	Convert HuggingFace LlamaForCausalLM naming
	to shark-lab Llama naming.

	Only handles key translation.
	Architecture compatibility must be checked separately.
	"""

	out = OrderedDict()

	global_map = {
		"model.embed_tokens.weight":
			"transformer.wte.weight",

		"model.norm.weight":
			"transformer.ln_f.weight",

		"lm_head.weight":
			"lm_head.weight",
	}

	for old_key, new_key in global_map.items():
		if old_key in state_dict:
			out[new_key] = state_dict[old_key]

	num_layers = 0

	for key in state_dict:
		if key.startswith("model.layers."):
			num_layers = max(
				num_layers,
				int(key.split(".")[2]) + 1
			)

	for i in range(num_layers):
		layer_map = {
			f"model.layers.{i}.input_layernorm.weight":
				f"transformer.h.{i}.ln_1.weight",

			f"model.layers.{i}.post_attention_layernorm.weight":
				f"transformer.h.{i}.ln_2.weight",

			f"model.layers.{i}.self_attn.q_proj.weight":
				f"transformer.h.{i}.attn.q_proj.weight",

			f"model.layers.{i}.self_attn.k_proj.weight":
				f"transformer.h.{i}.attn.k_proj.weight",

			f"model.layers.{i}.self_attn.v_proj.weight":
				f"transformer.h.{i}.attn.v_proj.weight",

			f"model.layers.{i}.self_attn.o_proj.weight":
				f"transformer.h.{i}.attn.o_proj.weight",

			f"model.layers.{i}.mlp.gate_proj.weight":
				f"transformer.h.{i}.mlp.c_fc1.weight",

			f"model.layers.{i}.mlp.up_proj.weight":
				f"transformer.h.{i}.mlp.c_fc2.weight",

			f"model.layers.{i}.mlp.down_proj.weight":
				f"transformer.h.{i}.mlp.c_proj.weight",
		}

		for old_key, new_key in layer_map.items():
			if old_key in state_dict:
				out[new_key] = state_dict[old_key]

	return out

def load_hf_checkpoint(model, hf_path):
	checkpoint = torch.load(hf_path, map_location="cpu")

	state_dict = convert_hf_llama_state_dict(
		checkpoint
	)

	missing, unexpected = model.load_state_dict(
		state_dict,
		strict=False
	)

	print("Missing:")
	for k in missing:
		print(k)

	print("Unexpected:")
	for k in unexpected:
		print(k)

	return model
