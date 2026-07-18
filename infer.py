"""
shark-lab
inference.py

The file test the model by calling generate of GPT directly.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from tokenizers import Tokenizer

import shark.format
import shark.util

"""

DEFAULT_SYSTEM_PROMPT = (
	"You are a helpful AI assistant named SmolLM, "
	"trained by Hugging Face"
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

	opt = shark.format.load_trainer_options(args.config)
	device, dtype = shark.util.resolve_runtime(opt.system, args.device, args.dtype)

	if args.seed is not None:
		torch.manual_seed(args.seed)
		if device.type == "cuda":
			torch.cuda.manual_seed_all(args.seed)
	
	tokenizer, eos_token_id = shark.format.get_tokenizer(
		opt.general.tokenizer_path)
	
	system_prompt = opt.general.system_prompt

	checkpoint_path = args.ckpt or os.path.join(
		opt.general.working_directory,
		"ckpt.pt",
	)

	# Construct and load on CPU before moving to the inference device.
	torch.set_default_device("cpu")
	model = shark.format.model_from_scratch(opt)

	step = shark.format.load_model_checkpoint(
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
		text = shark.format.format_chat(history, system_prompt)
		idx = shark.format.text_idx(tokenizer, text, device)
		# idx = encode_prompt(tokenizer, prompt, device)

		with torch.inference_mode():
			output = model.generate(
				idx,
				args.max_new_tokens,
				temperature=args.temperature,
				top_k=args.top_k,
				eos_token_id=eos_token_id,
			)
		
		reply = shark.format.idx_text(tokenizer, output, idx.shape[1])

		print(f"AI: {reply}")

		history.append({
			"role": "assistant",
			"content": reply,
		})


if __name__ == "__main__":
	main()
