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
		print('using tokenizer vocabulary size as GPT vacab')
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
		)
	return GPT(model_args)


def __optimizer_save_checkpoint(path: str, model: GPT, optimizer: torch.optim.AdamW, step: int) -> None:
	checkpoint = {
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'step': step,
	}
	print(f"saving checkpoint to {path}")
	torch.save(checkpoint, path)


def load_model_and_optimizer(meta_opt: dict[str, Any], ckpt_path: str | None, device: str):
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


def load_checkpoint(ckpt_path: str, model: GPT, optimizer: torch.optim.AdamW, device: str = 'cpu'):
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
