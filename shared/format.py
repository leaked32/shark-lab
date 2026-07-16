from typing import Any
from shared.util import notify_confirm
from collections.abc import Mapping

from shared.model import GPT, GPTOption
from dataclasses import dataclass
import tomllib
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch import Tensor as Tensor


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


def validate_state_dict_compatibility(
	model: nn.Module,
	hf_state: Mapping[str, Tensor],
) -> None:
	my_state = model.state_dict()

	missing: list[str] = []
	unexpected: list[str] = []
	shape_mismatches: list[
		tuple[str, tuple[int, ...], tuple[int, ...]]
	] = []
	dtype_mismatches: list[
		tuple[str, torch.dtype, torch.dtype]
	] = []

	for key, tensor in hf_state.items():
		if key not in my_state:
			unexpected.append(key)
			continue

		my_tensor = my_state[key]

		if tensor.shape != my_tensor.shape:
			shape_mismatches.append(
				(key, tuple(tensor.shape), tuple(my_tensor.shape))
			)

		if tensor.dtype != my_tensor.dtype:
			dtype_mismatches.append(
				(key, tensor.dtype, my_tensor.dtype)
			)

	for key in my_state:
		if key not in hf_state:
			missing.append(key)

	print(f"Hugging Face keys: {len(hf_state)}")
	print(f"Local model keys:   {len(my_state)}")

	print(f"\nMissing from HF checkpoint: {len(missing)}")
	for key in missing:
		print(f"  {key}")

	print(f"\nUnexpected HF keys: {len(unexpected)}")
	for key in unexpected:
		print(f"  {key}")

	print(f"\nShape mismatches: {len(shape_mismatches)}")
	for key, hf_shape, my_shape in shape_mismatches:
		print(
			f"  {key}: "
			f"HF={hf_shape}, local={my_shape}"
		)

	print(f"\nDtype differences: {len(dtype_mismatches)}")
	for key, hf_dtype, my_dtype in dtype_mismatches:
		print(
			f"  {key}: "
			f"HF={hf_dtype}, local={my_dtype}"
		)

	if missing or unexpected or shape_mismatches:
		raise RuntimeError(
			"Checkpoint is not structurally compatible with the model."
		)


def load_hf_state_dict(
	model: nn.Module,
	hf_state: Mapping[str, Tensor],
) -> nn.Module:
	validate_state_dict_compatibility(model, hf_state)

	my_state = model.state_dict()

	converted_state = {
		key: tensor.to(
			device=my_state[key].device,
			dtype=my_state[key].dtype,
		)
		for key, tensor in hf_state.items()
	}

	model.load_state_dict(converted_state, strict=True)

	model.lm_head.weight = model.model.embed_tokens.weight

	return model
