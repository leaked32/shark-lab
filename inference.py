from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from tokenizers import Tokenizer

import shared.format



DEFAULT_SYSTEM_PROMPT = (
	"You are a helpful AI assistant named SmolLM, "
	"trained by Hugging Face"
)


def format_chat(
	messages: list[dict[str, str]],
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
			f"{DEFAULT_SYSTEM_PROMPT}"
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


def resolve_runtime(
	system_opt: dict[str, Any],
	device_override: str | None,
	dtype_override: str | None,
) -> tuple[torch.device, torch.dtype]:
	dtype_name = dtype_override or system_opt.get("dtype", "float32")
	device_name = device_override or system_opt.get("device", "cpu")

	dtypes = {
		"float32": torch.float32,
		"bfloat16": torch.bfloat16,
		"float16": torch.float16,
	}

	if dtype_name not in dtypes:
		raise ValueError(f"unsupported dtype: {dtype_name}")

	device = torch.device(device_name)

	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable")

	return device, dtypes[dtype_name]

"""
def encode_prompt1(
	tokenizer: Any,
	prompt: str,
	device: torch.device,
) -> torch.Tensor:
	ids = tokenizer.encode(prompt, add_special_tokens=False)

	if not ids:
		if tokenizer.eos_token_id is None:
			raise ValueError(
				"empty prompt and tokenizer has no eos token"
			)
		ids = [tokenizer.eos_token_id]

	return torch.tensor([ids], dtype=torch.long, device=device)

def encode_prompt(
	tokenizer: Any,
	prompt: str,
	device: torch.device,
) -> torch.Tensor:
	encoded = tokenizer.apply_chat_template(
		[
			{
				"role": "user",
				"content": prompt,
			}
		],
		tokenize=True,
		add_generation_prompt=True,
		return_tensors="pt",
	)

	return encoded["input_ids"].to(device)


def decode_tokens1(tokenizer: Any, tokens: torch.Tensor) -> str:
	return tokenizer.decode(
		tokens[0].detach().cpu().tolist(),
		skip_special_tokens=False,
	)

def decode_tokens(
	tokenizer: Any,
	tokens: torch.Tensor,
	prompt_length: int,
) -> str:
	generated = tokens[0, prompt_length:].detach().cpu().tolist()

	return tokenizer.decode(
		generated,
		skip_special_tokens=True,
	)
"""

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate text from a shark-lab model checkpoint."
	)
	parser.add_argument("--config", default="options.toml")
	parser.add_argument("--ckpt", default=None)
	parser.add_argument("--max-new-tokens", type=int, default=100)
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--device", default=None)
	parser.add_argument(
		"--dtype",
		choices=["float32", "bfloat16", "float16"],
		default=None,
	)
	args = parser.parse_args()

	meta_opt = shared.format.load_meta_dataset(args.config)
	device, dtype = resolve_runtime(
		meta_opt["system"],
		args.device,
		args.dtype,
	)

	if args.seed is not None:
		torch.manual_seed(args.seed)
		if device.type == "cuda":
			torch.cuda.manual_seed_all(args.seed)

	tokenizer_path = meta_opt["train"]["tokenizer_path"]
	# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	
	tokenizer = Tokenizer.from_file(
		os.path.join(tokenizer_path, "tokenizer.json")
	)

	opt = shared.format.trainer_options(
		meta_opt["model"],
		meta_opt["train"],
	)

	checkpoint_path = args.ckpt or os.path.join(
		opt.train["working_directory"],
		"ckpt.pt",
	)

	# Construct and load on CPU before moving to the inference device.
	torch.set_default_device("cpu")
	model = shared.format.model_from_scratch(opt)

	step = shared.format.load_model_checkpoint(
		model,
		checkpoint_path,
		map_location="cpu",
	)

	model.to(device=device, dtype=dtype)
	model.eval()

	print(f"loaded checkpoint step {step}")
	print(f"device={device}, dtype={dtype}")
	
	history: list[dict[str, str]] = []
	
	while True:
		try:
			prompt = input("prompt: ")
		except (EOFError, KeyboardInterrupt):
			print()
			break
		
		history.append(
			{
				"role": "user",
				"content": prompt,
			})
		text = format_chat(history)
		
		ids = tokenizer.encode(
			text,
			add_special_tokens=False,
		).ids

		idx = torch.tensor(
			[ids],
			dtype=torch.long,
			device=device,
		)
		
		# idx = encode_prompt(tokenizer, prompt, device)

		with torch.inference_mode():
			output = model.generate(
				idx,
				args.max_new_tokens,
				temperature=args.temperature,
				top_k=args.top_k,
			)

		generated_ids = (
			output[0, idx.shape[1]:]
			.detach()
			.cpu()
			.tolist()
		)

		reply = tokenizer.decode(
			generated_ids,
			skip_special_tokens=True,
		)

		print(reply)

		history.append({
			"role": "assistant",
			"content": reply,
		})


if __name__ == "__main__":
	main()
