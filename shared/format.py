from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from shared.model import GPT, GPTOption


@dataclass
class trainer_options:
	model: dict[str, Any]
	train: dict[str, Any]

def model_from_scratch(opt: trainer_options) -> GPT:
	def get_tokenizer_vocab_count(tokenizer_path: str) -> int:
		from tokenizers import Tokenizer
		tokenizer = Tokenizer.from_file(
			os.path.join(tokenizer_path, "tokenizer.json")
		)

		return tokenizer.get_vocab_size(with_added_tokens=True)
	vocab_size = get_tokenizer_vocab_count(opt.train["tokenizer_path"])

	# Copy it instead of mutating the TOML dictionary.
	model_opt = dict(opt.model)

	configured_vocab = int(model_opt.get("vocab", 0))

	if configured_vocab == 0:
		model_opt["vocab"] = vocab_size
	elif configured_vocab != vocab_size:
		raise ValueError(
			"model vocabulary does not match tokenizer: "
			f"model={configured_vocab}, tokenizer={vocab_size}"
		)

	model_args = GPTOption(
		vocab=model_opt["vocab"],
		chan=model_opt["chan"],
		drop=model_opt["drop"],
		eps=model_opt["eps"],
		q_head=model_opt["q_head"],
		kv_head=model_opt["kv_head"],
		layer=model_opt["layer"],
		mlp_mul=model_opt["mlp_mul"],
		rope_theta=model_opt["rope_theta"],
	)

	return GPT(model_args)


def _atomic_torch_save(data: dict[str, Any], path: str) -> None:
	directory = os.path.dirname(os.path.abspath(path))
	os.makedirs(directory, exist_ok=True)

	fd, temporary_path = tempfile.mkstemp(
		dir=directory,
		prefix=".checkpoint-",
		suffix=".tmp",
	)
	os.close(fd)

	try:
		torch.save(data, temporary_path)
		os.replace(temporary_path, path)
	except BaseException:
		if os.path.exists(temporary_path):
			os.remove(temporary_path)
		raise


def save_model_checkpoint(
	path: str,
	model: nn.Module,
	step: int = 0,
) -> None:
	"""Save weights for inference or pretrained-model conversion."""

	checkpoint = {
		"format_version": 1,
		"kind": "model",
		"model": model.state_dict(),
		"step": int(step),
	}

	_atomic_torch_save(checkpoint, path)
	print(f"saved model checkpoint: {path}")


def load_model_checkpoint(
	model: nn.Module,
	path: str,
	map_location: str | torch.device = "cpu",
) -> int:
	if not os.path.isfile(path):
		raise FileNotFoundError(f"checkpoint not found: {path}")

	checkpoint = torch.load(path, map_location=map_location)

	if "model" not in checkpoint:
		raise RuntimeError("checkpoint contains no model state")

	model.load_state_dict(checkpoint["model"], strict=True)

	step = int(checkpoint.get("step", 0))
	print(f"loaded model checkpoint: {path}")
	return step


def save_training_checkpoint(
	path: str,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	next_step: int,
) -> None:
	"""Save everything needed to resume training."""

	checkpoint = {
		"format_version": 1,
		"kind": "training",
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"step": int(next_step),
	}

	_atomic_torch_save(checkpoint, path)
	print(f"saved training checkpoint: {path}")


def load_training_checkpoint(
	path: str,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	map_location: str | torch.device = "cpu",
) -> int:
	if not os.path.isfile(path):
		raise FileNotFoundError(f"checkpoint not found: {path}")

	checkpoint = torch.load(path, map_location=map_location)

	if "model" not in checkpoint:
		raise RuntimeError("checkpoint contains no model state")

	if "optimizer" not in checkpoint:
		raise RuntimeError("checkpoint contains no optimizer state")

	model.load_state_dict(checkpoint["model"], strict=True)
	optimizer.load_state_dict(checkpoint["optimizer"])

	next_step = int(checkpoint.get("step", 0))
	print(f"resumed training from step {next_step}")
	return next_step


def load_meta_dataset(path: str) -> dict[str, Any]:
	with open(path, "rb") as file:
		return tomllib.load(file)


def validate_state_dict_compatibility(
	model: nn.Module,
	source_state: Mapping[str, Tensor],
) -> None:
	local_state = model.state_dict()

	missing = sorted(set(local_state) - set(source_state))
	unexpected = sorted(set(source_state) - set(local_state))

	shape_mismatches = [
		(key, tuple(source_state[key].shape), tuple(local_state[key].shape))
		for key in sorted(set(source_state) & set(local_state))
		if source_state[key].shape != local_state[key].shape
	]

	if missing:
		print(f"missing keys: {len(missing)}")
		for key in missing:
			print(f"\t{key}")

	if unexpected:
		print(f"unexpected keys: {len(unexpected)}")
		for key in unexpected:
			print(f"\t{key}")

	if shape_mismatches:
		print(f"shape mismatches: {len(shape_mismatches)}")
		for key, source_shape, local_shape in shape_mismatches:
			print(
				f"\t{key}: source={source_shape}, local={local_shape}"
			)

	if missing or unexpected or shape_mismatches:
		raise RuntimeError("model architectures are incompatible")

	print(f"state dictionary validated: {len(local_state)} tensors")


def load_hf_state_dict(
	model: nn.Module,
	hf_state: Mapping[str, Tensor],
) -> None:
	validate_state_dict_compatibility(model, hf_state)

	# load_state_dict already performs destination dtype/device conversion.
	model.load_state_dict(hf_state, strict=True)


