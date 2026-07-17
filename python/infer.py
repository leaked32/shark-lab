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

import shared.format
import shared.util

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

	meta_opt = shared.format.load_meta_dataset(args.config)
	device, dtype = shared.util.resolve_runtime(meta_opt["system"], args.device, args.dtype)

	if args.seed is not None:
		torch.manual_seed(args.seed)
		if device.type == "cuda":
			torch.cuda.manual_seed_all(args.seed)
	
	tokenizer, eos_token_id = shared.format.get_tokenizer(
		meta_opt["train"]["tokenizer_path"])
	"""
	tokenizer_path = meta_opt["train"]["tokenizer_path"]
	# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	
	tokenizer = Tokenizer.from_file(
		os.path.join(tokenizer_path, "tokenizer.json")
	)
	eos_token_id = tokenizer.token_to_id("<|im_end|>")

	if eos_token_id is None:
		raise ValueError("tokenizer has no <|im_end|> token")
	"""

	opt = shared.format.trainer_options(
		meta_opt["model"],
		meta_opt["train"],
	)
	system_prompt = meta_opt["infer"]["system_prompt"]

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
		text = shared.format.format_chat(history, system_prompt)
		idx = shared.format.text_idx(tokenizer, text, device)
		# idx = encode_prompt(tokenizer, prompt, device)

		with torch.inference_mode():
			output = model.generate(
				idx,
				args.max_new_tokens,
				temperature=args.temperature,
				top_k=args.top_k,
				eos_token_id=eos_token_id,
			)
		
		reply = shared.format.idx_text(tokenizer, output, idx.shape[1])

		print(reply)

		history.append({
			"role": "assistant",
			"content": reply,
		})


if __name__ == "__main__":
	main()
