"""
shark-lab
shark/format.py

This little module adapts shark-lab to different purposes
"""

from __future__ import annotations

import os
import tempfile
import tomllib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from shark.model import GPT, GPTOption

from tokenizers import Tokenizer

import shark.util

@dataclass(frozen=True, slots=True)
class GeneralOption:
	"""Inference settings stored under [infer]."""
	system_prompt: str

	tokenizer_path: Path
	working_directory: Path


@dataclass(frozen=True, slots=True)
class TrainOption:
	"""Training settings stored under [train]."""

	max_steps: int
	log_interval: int
	save_interval: int
	save_independent_checkpoints: bool

	dataset_type: int
	batch_count: int

	dataset_sft_train: Path
	dataset_train: Path
	dataset_validation: Path
	corpus_block_size: int

	optimizer_learning_rate: float
	adamw_weight_decay: float
	adamw_beta1: float
	adamw_beta2: float

	def __post_init__(self) -> None:
		if self.max_steps <= 0:
			raise ValueError("train.max_steps must be > 0")
		if self.log_interval <= 0:
			raise ValueError("train.log_interval must be > 0")
		if self.save_interval <= 0:
			raise ValueError("train.save_interval must be > 0")
		if self.dataset_type not in (0, 1):
			raise ValueError(
				"train.dataset_type must be 0 (CORPUS) or 1 (SFT)"
			)
		if self.batch_count <= 0:
			raise ValueError("train.batch_count must be > 0")
		if self.corpus_block_size <= 0:
			raise ValueError("train.corpus_block_size must be > 0")
		if self.optimizer_learning_rate <= 0.0:
			raise ValueError(
				"train.optimizer_learning_rate must be > 0"
			)
		if self.adamw_weight_decay < 0.0:
			raise ValueError(
				"train.adamw_weight_decay must be >= 0"
			)
		if not 0.0 <= self.adamw_beta1 < 1.0:
			raise ValueError(
				"train.adamw_beta1 must satisfy 0 <= beta1 < 1"
			)
		if not 0.0 <= self.adamw_beta2 < 1.0:
			raise ValueError(
				"train.adamw_beta2 must satisfy 0 <= beta2 < 1"
			)

@dataclass(frozen=True, slots=True)
class SystemOption:
	"""Runtime settings stored under [system]."""

	device: str
	dtype: str

	def __post_init__(self) -> None:
		allowed_dtypes = {
			"float32",
			"float16",
			"bfloat16",
		}

		if self.dtype not in allowed_dtypes:
			raise ValueError(
				f"system.dtype must be one of {sorted(allowed_dtypes)}, "
				f"got {self.dtype!r}"
			)


@dataclass(frozen=True, slots=True)
class trainer_options:
	"""Complete configuration."""

	model: GPTOption
	general: GeneralOption
	train: TrainOption
	system: SystemOption

def require_section(
	config: dict[str, Any],
	name: str,
) -> dict[str, Any]:
	section = config.get(name)

	if not isinstance(section, dict):
		raise ValueError(f"Missing or invalid [{name}] section")

	return section


def reject_unknown_fields(
	section_name: str,
	data: dict[str, Any],
	allowed_fields: set[str],
) -> None:
	unknown_fields = set(data) - allowed_fields

	if unknown_fields:
		names = ", ".join(sorted(unknown_fields))
		raise ValueError(
			f"Unknown field(s) in [{section_name}]: {names}"
		)


def load_trainer_options(path: str | Path) -> trainer_options:
	config_path = Path(path)

	with config_path.open("rb") as file:
		raw = tomllib.load(file)

	model_raw = require_section(raw, "model")
	general_raw = require_section(raw, "general")
	train_raw = require_section(raw, "train")
	system_raw = require_section(raw, "system")

	reject_unknown_fields(
		"model",
		model_raw,
		{
			"vocab",
			"layer",
			"chan",
			"q_head",
			"kv_head",
			"mlp_mul",
			"drop",
			"eps",
			"rope_theta",
			"bias",
		},
	)

	reject_unknown_fields(
		"general",
		general_raw,
		{
			"system_prompt",
			"tokenizer_path",
			"working_directory",
		},
	)

	reject_unknown_fields(
		"train",
		train_raw,
		{
			"max_steps",
			"log_interval",
			"save_interval",
			"save_independent_checkpoints",
			"dataset_type",
			"batch_count",
			"dataset_sft_train",
			"dataset_train",
			"dataset_validation",
			"corpus_block_size",
			"optimizer_learning_rate",
			"adamw_weight_decay",
			"adamw_beta1",
			"adamw_beta2",
		},
	)

	reject_unknown_fields(
		"system",
		system_raw,
		{"device", "dtype"},
	)

	# `bias` is accepted in the TOML for compatibility but deliberately
	# not passed to GPTOption because the model does not support it yet.
	bias = model_raw.get("bias", False)
	if bias is not False:
		raise ValueError(
			"model.bias=true is unsupported by the current model"
		)

	model = GPTOption(
		vocab=int(model_raw["vocab"]),
		layer=int(model_raw["layer"]),
		chan=int(model_raw["chan"]),
		q_head=int(model_raw["q_head"]),
		kv_head=int(model_raw["kv_head"]),
		mlp_mul=int(model_raw["mlp_mul"]),
		drop=float(model_raw["drop"]),
		eps=float(model_raw["eps"]),
		rope_theta=float(model_raw["rope_theta"]),
	)

	general = GeneralOption(
		system_prompt=str(general_raw["system_prompt"]),
		tokenizer_path=Path(general_raw["tokenizer_path"]),
		working_directory=Path(general_raw["working_directory"]),
	)

	train = TrainOption(
		max_steps=int(train_raw["max_steps"]),
		log_interval=int(train_raw["log_interval"]),
		save_interval=int(train_raw["save_interval"]),
		save_independent_checkpoints=bool(train_raw["save_independent_checkpoints"]),
		dataset_type=int(train_raw["dataset_type"]),
		batch_count=int(train_raw["batch_count"]),
		dataset_sft_train=Path(train_raw["dataset_sft_train"]),
		dataset_train=Path(train_raw["dataset_train"]),
		dataset_validation=Path(train_raw["dataset_validation"]),
		corpus_block_size=int(train_raw["corpus_block_size"]),
		optimizer_learning_rate=float(
			train_raw["optimizer_learning_rate"]
		),
		adamw_weight_decay=float(
			train_raw["adamw_weight_decay"]
		),
		adamw_beta1=float(train_raw["adamw_beta1"]),
		adamw_beta2=float(train_raw["adamw_beta2"]),
	)

	system = SystemOption(
		device=str(system_raw["device"]),
		dtype=str(system_raw["dtype"]),
	)

	return trainer_options(
		model=model,
		general=general,
		train=train,
		system=system,
	)

def model_from_scratch(opt: trainer_options) -> GPT:
	def get_tokenizer_vocab_count(tokenizer_path: str | Path) -> int:
		tokenizer = Tokenizer.from_file(
			os.path.join(tokenizer_path, "tokenizer.json")
		)

		return tokenizer.get_vocab_size(with_added_tokens=True)
	vocab_size = get_tokenizer_vocab_count(opt.general.tokenizer_path)

	# Copy it instead of mutating the TOML dictionary.

	configured_vocab = opt.model.vocab

	if configured_vocab == 0:
		opt.model.vocab = vocab_size
	elif configured_vocab != vocab_size:
		raise ValueError(
			"model vocabulary does not match tokenizer: "
			f"model={configured_vocab}, tokenizer={vocab_size}"
		)


	return GPT(opt.model)


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
	else:
		model.load_state_dict(checkpoint["model"], strict=True)

	if "optimizer" not in checkpoint:
		shark.util.notify_confirm("checkpoint contains no optimizer state")
		next_step = 0
	else:
		optimizer.load_state_dict(checkpoint["optimizer"])
		next_step = int(checkpoint.get("step", 0))
	
	print(f"resumed training from step {next_step}")
	return next_step



# =================================================================================================
# TOKENIZER
# =================================================================================================

def format_chat(
	messages: list[dict[str, str]],
	system_prompt: str,
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
			f"{system_prompt}"
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

def text_ids(tokenizer, text: str) -> list[int]:
	ids = tokenizer.encode(
		text,
		add_special_tokens=False,
	).ids
	return ids

def text_idx(tokenizer, text: str, device) -> Tensor:
	ids = text_ids(tokenizer, text)
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


def get_tokenizer(tokenizer_path: str | Path) -> tuple[Tokenizer, int]:
	# tokenizer_path = meta_opt["train"]["tokenizer_path"]
	# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	
	tokenizer = Tokenizer.from_file(
		os.path.join(tokenizer_path, "tokenizer.json")
	)
	eos_token_id = tokenizer.token_to_id("<|im_end|>")

	if eos_token_id is None:
		raise ValueError("tokenizer has no <|im_end|> token")
	
	return tokenizer, eos_token_id

# =================================================================================================
# JSONL CONVESATION DATASET
# =================================================================================================

@dataclass
class JsonlMessage:
	role: str
	content: str

class JsonlDataset:

	def __init__(self, path: str | Path):
		self.path = path if isinstance(path, Path) else Path(path)
		self.items: list[list[JsonlMessage]] = []
		
		self._load()
	
	def _load(self) -> None:
		with self.path.open("r", encoding="utf-8") as f:
			for line_no, raw in enumerate(f, start=1):
				conversation: list[JsonlMessage] = []
				line = raw.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
				except json.JSONDecodeError as exc:
					raise ValueError(f"{self.path}:{line_no}: invalid JSON") from exc
				if not isinstance(obj, dict):
					raise ValueError(f"{self.path}:{line_no}: root must be object")
				
				if obj["messages"]:
					for i in obj["messages"]:
						conversation.append(JsonlMessage(i["role"], i["content"]))
				self.items.append(conversation)
	
	def to_sft_tensors(self, tokenizer: Tokenizer, ignore_index: int, conversation_index: int,
					) -> tuple[Tensor, Tensor]:
		# Python masking
		token_ids: list[int] = []
		train_mask: list[bool] = []
		
		conversation = self.items[conversation_index]
		
		for message in conversation:
			mask = message.role == 'assistant'
			prefix = f"<|im_start|>{message.role}\n"
			content = message.content
			suffix = "<|im_end|>"
			
			prefix_ids = text_ids(tokenizer, prefix)
			token_ids.extend(prefix_ids)
			train_mask.extend([False] * len(prefix_ids))
			
			content_ids = text_ids(tokenizer, content)
			token_ids.extend(content_ids)
			train_mask.extend([mask] * len(content_ids))
			
			suffix_ids = text_ids(tokenizer, suffix)
			token_ids.extend(suffix_ids)
			train_mask.extend([mask] * len(suffix_ids))
			
			newline_ids = text_ids(tokenizer, "\n")
			token_ids.extend(newline_ids)
			train_mask.extend([False] * len(newline_ids))
			
			whole = (
				f"<|im_start|>{message.role}\n"
				f"{message.content}"
				f"<|im_end|>\n"
			)
			whole_ids = text_ids(tokenizer, whole)
			
			merged = prefix_ids + content_ids + suffix_ids + newline_ids
			if whole_ids != merged:
				raise ValueError(
					"separate chat segments tokenize differently from the complete message"
				)
			
			if mask and not message.content.strip():
				raise ValueError(
					"assistant message cannot be empty"
					f"conversation: {conversation}"
					)
		
		if not any(train_mask):
			raise ValueError(
				f"conversation contains no assistant targets"
				f"conversation: {conversation}"
			)
		
		targets = [
			token_id if should_train else ignore_index
			for token_id, should_train in zip(token_ids[1:], train_mask[1:])
		]
		
		input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
		targets = torch.tensor(targets, dtype=torch.long)
		return input_ids, targets
	
	
	def validate_sft_indices(self, tokenizer) -> tuple[list[int], list[int]]:
		"""
		Check every SFT record once and return the indices that can be converted.
		This avoids infinite loops when some or all dataset records are invalid.
		
		The function returns the **valid indecies** and **the invalid**
		"""
		valid_indices: list[int] = []
		invalid_indices: list[int] = []
		item_count = len(self.items)

		if item_count == 0:
			raise RuntimeError("The SFT dataset is empty.")

		print(f"Validating {item_count} SFT records...")

		for cindex in range(item_count):
			try:
				cx, cy = self.to_sft_tensors(tokenizer, -1, cindex)

				if cx.ndim != 1 or cy.ndim != 1:
					raise ValueError(
						f"Expected one-dimensional tensors, got "
						f"x.ndim={cx.ndim}, y.ndim={cy.ndim}"
					)

				if cx.size(0) != cy.size(0):
					raise ValueError(
						f"Input and target lengths differ: "
						f"{cx.size(0)} != {cy.size(0)}"
					)

				if cx.numel() == 0:
					raise ValueError("The converted sample is empty.")

			except Exception as exc:
				print(
					f"Skipping invalid SFT record "
					f"index={cindex}: {type(exc).__name__}: {exc}"
				)
				invalid_indices.append(cindex)
				continue

			valid_indices.append(cindex)

		if not valid_indices:
			raise RuntimeError("The SFT dataset contains no valid training records.")

		skipped_count = item_count - len(valid_indices)

		print(
			f"SFT validation complete: "
			f"{len(valid_indices)} valid, "
			f"{skipped_count} skipped."
		)

		return valid_indices, invalid_indices

