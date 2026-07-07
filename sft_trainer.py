from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import shared.format

'''
Expected data format, JSONL:

{"prompt":"Question...","response":"Answer..."}

Run:

python sft_trainer.py --data data/sft.jsonl --max-steps 1000
'''


@dataclass
class SFTExample:
	input_ids: list[int]
	labels: list[int]


class JsonlSFTDataset(Dataset[SFTExample]):
	def __init__(self, path: str, tokenizer: Any, block_size: int, prompt_key: str, response_key: str, eos: bool = True):
		self.examples: list[SFTExample] = []
		self.tokenizer = tokenizer
		self.block_size = block_size
		self.prompt_key = prompt_key
		self.response_key = response_key
		self.eos = eos
		self._load(path)

	def _load(self, path: str) -> None:
		with open(path, 'r', encoding='utf-8') as f:
			for line_no, line in enumerate(f, 1):
				line = line.strip()
				if not line:
					continue
				obj = json.loads(line)
				prompt = str(obj[self.prompt_key])
				response = str(obj[self.response_key])
				self.examples.append(self._encode(prompt, response, line_no))
		if len(self.examples) == 0:
			raise ValueError(f'no SFT examples loaded from {path}')

	def _encode(self, prompt: str, response: str, line_no: int) -> SFTExample:
		prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
		response_ids = self.tokenizer.encode(response, add_special_tokens=False)
		if self.eos and self.tokenizer.eos_token_id is not None:
			response_ids.append(int(self.tokenizer.eos_token_id))

		ids = prompt_ids + response_ids
		labels = [-1] * len(prompt_ids) + response_ids[:]

		if len(ids) < 2:
			raise ValueError(f'line {line_no}: encoded example is too short')
		if len(ids) > self.block_size + 1:
			ids = ids[:self.block_size + 1]
			labels = labels[:self.block_size + 1]
		if all(x == -1 for x in labels[1:]):
			raise ValueError(f'line {line_no}: response was truncated away; increase block size or shorten prompt')

		return SFTExample(ids, labels)

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, idx: int) -> SFTExample:
		return self.examples[idx]


def collate_sft(batch: list[SFTExample], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
	max_len = max(len(x.input_ids) for x in batch) - 1
	xs = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
	ys = torch.full((len(batch), max_len), -1, dtype=torch.long, device=device)

	for i, ex in enumerate(batch):
		inp = ex.input_ids[:-1]
		tgt = ex.labels[1:]
		n = min(len(inp), max_len)
		xs[i, :n] = torch.tensor(inp[:n], dtype=torch.long, device=device)
		ys[i, :n] = torch.tensor(tgt[:n], dtype=torch.long, device=device)
	return xs, ys



def main() -> None:
	p = argparse.ArgumentParser(description='Response-only supervised fine-tuning for llm-lab GPT.')
	p.add_argument('--config', default='trainer.toml')
	p.add_argument('--data', required=True, help='JSONL with prompt/response fields')
	p.add_argument('--prompt-key', default='prompt')
	p.add_argument('--response-key', default='response')
	p.add_argument('--ckpt-in', default=None, help='base/pretrain checkpoint, default: working_directory/ckpt.pt')
	p.add_argument('--ckpt-out', default=None, help='SFT checkpoint output, default: working_directory/sft.pt')
	p.add_argument('--max-steps', type=int, default=None)
	p.add_argument('--batch-size', type=int, default=None)
	p.add_argument('--block-size', type=int, default=None)
	p.add_argument('--save-interval', type=int, default=None)
	p.add_argument('--log-interval', type=int, default=None)
	p.add_argument('--no-eos', action='store_true')
	args = p.parse_args()

	meta_opt = shared.format.load_meta_dataset(args.config)
	opt_sys = meta_opt['system']
	opt_train = meta_opt['train']
	device = opt_sys['device']

	torch.set_default_dtype({'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[opt_sys['dtype']])
	torch.set_default_device(device)
	torch.set_num_threads(16)
	torch.set_num_interop_threads(2)

	working_dir = opt_train['working_directory']
	ckpt_in = args.ckpt_in or os.path.join(working_dir, 'ckpt.pt')
	ckpt_out = args.ckpt_out or os.path.join(working_dir, 'sft.pt')
	max_steps = args.max_steps or opt_train['max_steps']
	batch_size = args.batch_size or opt_train['corpus_batch_size']
	block_size = args.block_size or opt_train['corpus_block_size']
	save_interval = args.save_interval or opt_train['save_interval']
	log_interval = args.log_interval or opt_train['log_interval']

	tokenizer = AutoTokenizer.from_pretrained(opt_train['tokenizer_path'])
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

	dataset = JsonlSFTDataset(args.data, tokenizer, block_size, args.prompt_key, args.response_key, eos=not args.no_eos)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_sft(b, pad_id, device))

	model, optimizer, start_step = shared.format.load_model_and_optimizer(meta_opt, ckpt_in, device)
	model.train()

	step = start_step
	while step < max_steps:
		for x, y in loader:
			if x.max().item() >= model.opt.vocab:
				raise ValueError(f'dataset token id {x.max().item()} exceeds model vocab {model.opt.vocab}')
			_, loss = model(x, y)
			if loss is None:
				raise RuntimeError('model returned loss=None')

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			if step % log_interval == 0:
				print(f'sft step {step} | loss {loss.item():.4f}')
			if step > start_step and step % save_interval == 0:
				shared.format.save_checkpoint(ckpt_out, model, optimizer, step)

			step += 1
			if step >= max_steps:
				break

	shared.format.save_checkpoint(ckpt_out, model, optimizer, step)
	print(f'SFT finished at step {step}, saved to {ckpt_out}')


if __name__ == '__main__':
	main()
