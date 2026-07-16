from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from shared.model import GPT, GPTOption

from tokenizers import Tokenizer

@dataclass
class trainer_options:
	model: dict[str, Any]
	train: dict[str, Any]

def model_from_scratch(opt: trainer_options) -> GPT:
	def get_tokenizer_vocab_count(tokenizer_path: str) -> int:
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


# =================================================================================================
# TOKENIZER
# =================================================================================================

def format_chat(
	messages: list[dict[str, str]],
	system_propmt: str,
	add_generation_prompt: bool = True,
) -> str:
	if not messages:
		raise ValueError("messages cannot be empty")

	valid_roles = {"system", "user", "assistant"}

	for message in messages:
		if message["role"] not in valid_roles:
			raise ValueError(
				f"unsupported role: {message['role']}"
			)

	formatted: list[str] = []

	if messages[0]["role"] != "system":
		formatted.append(
			f"<|im_start|>system\n"
			f"{system_propmt}"
			f"<|im_end|>\n"
		)

	for message in messages:
		formatted.append(
			f"<|im_start|>{message['role']}\n"
			f"{message['content']}"
			f"<|im_end|>\n"
		)

	if add_generation_prompt:
		formatted.append("<|im_start|>assistant\n")

	return "".join(formatted)

def text_idx(tokenizer, text: str, device) -> Tensor:

	ids = tokenizer.encode(
		text,
		add_special_tokens=False,
	).ids

	idx = torch.tensor(
		[ids],
		dtype=torch.long,
		device=device,
	)
	
	return idx

def idx_text(tokenizer, output, begin) -> str:
	
	generated_ids = (
		output[0, begin:]
		.detach()
		.cpu()
		.tolist()
	)

	reply = tokenizer.decode(
		generated_ids,
		skip_special_tokens=True,
	)
	
	return reply
