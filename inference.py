from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from transformers import AutoTokenizer

import shared.format

def set_torch_options(system_opt: dict[str, Any], device_override: str | None, dtype_override: str | None) -> str:
	dtype_name = dtype_override or system_opt.get('dtype', 'float32')
	device = device_override or system_opt.get('device', 'cpu')
	dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
	if dtype_name not in dtypes:
		raise ValueError(f"unsupported dtype: {dtype_name}")
	torch.set_default_dtype(dtypes[dtype_name])
	torch.set_default_device(device)
	return device

def encode_prompt(tokenizer: Any, prompt: str, device: str) -> torch.Tensor:
	ids = tokenizer.encode(prompt, add_special_tokens=False)
	if len(ids) == 0:
		if tokenizer.eos_token_id is None:
			raise ValueError("empty prompt and tokenizer has no eos_token_id")
		ids = [tokenizer.eos_token_id]
	return torch.tensor([ids], dtype=torch.long, device=device)

def decode_tokens(tokenizer: Any, tokens: torch.Tensor) -> str:
	ids = tokens[0].detach().cpu().tolist()
	return tokenizer.decode(ids, skip_special_tokens=False)

def main() -> None:
	parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
	parser.add_argument('--config', default='options.toml')
	parser.add_argument('--ckpt', default=None)
	parser.add_argument('--prompt', default='')
	parser.add_argument('--max-new-tokens', type=int, default=100)
	parser.add_argument('--temperature', type=float, default=1.0)
	parser.add_argument('--top-k', type=int, default=None)
	parser.add_argument('--seed', type=int, default=None)
	parser.add_argument('--device', default=None)
	parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default=None)
	args = parser.parse_args()

	if args.seed is not None:
		torch.manual_seed(args.seed)

	meta_opt = shared.format.load_meta_dataset(args.config)
	device = set_torch_options(meta_opt['system'], args.device, args.dtype)

	tokenizer_path = meta_opt['train']['tokenizer_path']
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

	opt = shared.format.trainer_options(meta_opt['model'], meta_opt['train'])
	model = shared.format.model_from_scratch(opt)
	ckpt_path = args.ckpt or os.path.join(meta_opt['train']['working_directory'], 'ckpt.pt')
	step = shared.format.load_model_checkpoint(model, ckpt_path, device)
	model.eval()

	idx = encode_prompt(tokenizer, args.prompt, device)
	with torch.no_grad():
		out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

	print(f"loaded checkpoint step {step}")
	print(decode_tokens(tokenizer, out))

if __name__ == '__main__':
	main()
