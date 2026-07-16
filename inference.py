from __future__ import annotations

import argparse
import gc
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import shared.format


def set_torch_options(system_opt: dict[str, Any], device_override: str | None, dtype_override: str | None) -> tuple[str, torch.dtype]:
	dtype_name = dtype_override or system_opt.get("dtype", "float32")
	device = device_override or system_opt.get("device", "cpu")
	dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

	if dtype_name not in dtypes:
		raise ValueError(f"unsupported dtype: {dtype_name}")

	dtype = dtypes[dtype_name]
	torch.set_default_dtype(dtype)
	torch.set_default_device(device)
	return device, dtype


def encode_prompt(tokenizer: Any, prompt: str, device: str) -> Tensor:
	ids = tokenizer.encode(prompt, add_special_tokens=False)

	if not ids:
		if tokenizer.eos_token_id is None:
			raise ValueError("empty prompt and tokenizer has no eos_token_id")
		ids = [tokenizer.eos_token_id]

	return torch.tensor([ids], dtype=torch.long, device=device)


def decode_tokens(tokenizer: Any, tokens: Tensor) -> str:
	ids = tokens[0].detach().cpu().tolist()
	return tokenizer.decode(ids, skip_special_tokens=False)


def validate_state_dict(model: nn.Module, hf_state: Mapping[str, Tensor]) -> None:
	my_state = model.state_dict()

	missing = [key for key in my_state if key not in hf_state]
	unexpected = [key for key in hf_state if key not in my_state]
	shape_mismatches = [
		(key, tuple(hf_state[key].shape), tuple(my_state[key].shape))
		for key in hf_state
		if key in my_state and hf_state[key].shape != my_state[key].shape
	]

	if missing:
		print("\nMissing from Hugging Face checkpoint:")
		for key in missing:
			print(f"\t{key}")

	if unexpected:
		print("\nUnexpected Hugging Face keys:")
		for key in unexpected:
			print(f"\t{key}")

	if shape_mismatches:
		print("\nShape mismatches:")
		for key, hf_shape, my_shape in shape_mismatches:
			print(f"\t{key}: HF={hf_shape}, local={my_shape}")

	if missing or unexpected or shape_mismatches:
		raise RuntimeError("Hugging Face checkpoint is not structurally compatible")

	print(f"checkpoint validation passed: {len(my_state)} tensors")


def load_hf_weights(model: nn.Module, hf_state: Mapping[str, Tensor]) -> None:
	validate_state_dict(model, hf_state)

	# load_state_dict copies each source tensor into the existing destination tensor.
	# It also performs the required device/dtype conversion.
	incompatible = model.load_state_dict(hf_state, strict=True)

	if incompatible.missing_keys or incompatible.unexpected_keys:
		raise RuntimeError(
			f"missing={incompatible.missing_keys}, "
			f"unexpected={incompatible.unexpected_keys}"
		)

	# Preserve weight tying explicitly.
	model.lm_head.weight = model.model.embed_tokens.weight

	if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
		raise RuntimeError("lm_head and token embeddings are not tied")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate text using shark-lab with Hugging Face weights.")
	parser.add_argument("--config", default="options.toml")
	parser.add_argument("--max-new-tokens", type=int, default=100)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--device", default=None)
	parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
	args = parser.parse_args()

	if args.seed is not None:
		torch.manual_seed(args.seed)

	meta_opt = shared.format.load_meta_dataset(args.config)
	device, dtype = set_torch_options(meta_opt["system"], args.device, args.dtype)

	pretrained_path = meta_opt["train"]["tokenizer_path"]
	tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

	opt = shared.format.trainer_options(meta_opt["model"], meta_opt["train"])
	model = shared.format.model_from_scratch(opt)

	print(f"loading Hugging Face weights from: {pretrained_path}")

	# Keep the temporary Hugging Face model on CPU to avoid duplicating VRAM.
	hf_model = AutoModelForCausalLM.from_pretrained(
		pretrained_path,
		torch_dtype="auto",
		low_cpu_mem_usage=True,
		device_map="cpu",
	)

	load_hf_weights(model, hf_model.state_dict())

	del hf_model
	gc.collect()

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	model = model.to(device=device, dtype=dtype)
	model.eval()

	print("SmolLM2-360M weights loaded successfully")

	while True:
		try:
			prompt = input("prompt: ")
		except (EOFError, KeyboardInterrupt):
			print()
			break

		idx = encode_prompt(tokenizer, prompt, device)

		with torch.no_grad():
			out = model.generate(
				idx,
				args.max_new_tokens,
				temperature=args.temperature,
				top_k=args.top_k,
			)

		print(decode_tokens(tokenizer, out))


if __name__ == "__main__":
	main()
