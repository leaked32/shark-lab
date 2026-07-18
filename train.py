import os
import time

import argparse
import shared.format
from shared.util import get_batch

from tokenizers import Tokenizer
import torch
from torch import Tensor

import random
from typing import cast


def enlarge_to_fit(x: Tensor, least_len: int, mm: int) -> Tensor:
	raw_len = x.size(0)
	if raw_len < least_len:
		y = x.new_full((least_len - raw_len,), mm)
		return torch.cat((x, y), dim=-1)
	return x

def main():
	parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
	parser.add_argument('--config', default='options.toml')
	args = parser.parse_args()
	
	
	meta_opt = shared.format.load_meta_dataset(args.config)
	opt_sys = meta_opt['system']
	
	torch.set_default_dtype(
		{'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
		[opt_sys['dtype']]
		)
	torch.set_default_device(opt_sys['device'])
	
	torch.set_num_threads(16)
	torch.set_num_interop_threads(2)
	
	# Trainer
	opt = shared.format.trainer_options(meta_opt['model'], meta_opt['train'])
	tokenizer_path = opt.train["tokenizer_path"]
	tokenizer, eos_token_id = shared.format.get_tokenizer(tokenizer_path)
	
	
	model_path: str = opt.train['working_directory']
	os.makedirs(model_path, exist_ok=True)
	
	model = shared.format.model_from_scratch(opt)
	if model is None:
		raise Exception("trainer.cased_model returned None")

	optimizer = model.optimizer_adamw(
		opt.train['adamw_weight_decay'],
		opt.train['optimizer_learning_rate'],
		(opt.train['adamw_beta1'], opt.train['adamw_beta2']),
		opt_sys['device']
	)

	start_step = 0
	ckpt_path = os.path.join(model_path, "ckpt.pt")

	if os.path.exists(ckpt_path):
		start_step = shared.format.load_training_checkpoint(
			ckpt_path,
			model,
			optimizer,
			map_location=opt_sys["device"],
		)
	dataset_type: int = opt.train["dataset_type"]
	batch_count: int = opt.train['corpus_batch_size']
	
	match cast(int, dataset_type):
		case 1:
			jsonl_path: str = opt.train["dataset_sft_train"]
			print(f"SFT jsonl_path: {jsonl_path}")
			dataset_context = shared.format.JsonlDataset(jsonl_path)
			comfy_arange: list[int] = [i for i in range(len(dataset_context.items))]
			puckered = comfy_arange.copy()
			random.shuffle(puckered)
	
	
	for step in range(start_step, opt.train['max_steps']):
		x: Tensor
		y: Tensor
		match cast(int, dataset_type):
			case 0:
				x, y = get_batch(opt.train['dataset_train'],
					opt.train['corpus_block_size'], batch_count)
			case 1:
				lx: list[Tensor] = []
				ly: list[Tensor] = []
				max_len = 0

				for _ in range(batch_count):
					if not puckered:
						puckered = comfy_arange.copy()
						random.shuffle(puckered)

					cindex = puckered.pop()
					cx, cy = dataset_context.to_sft_tensors(
						tokenizer,
						-1,
						cindex,
					)
					assert cx.size(0) == cy.size(0)

					lx.append(cx)
					ly.append(cy)
					max_len = max(max_len, cx.size(0))
				
				for i in range(batch_count):
					lx[i] = enlarge_to_fit(lx[i], max_len, eos_token_id)
					ly[i] = enlarge_to_fit(ly[i], max_len, -1)
				
				x = torch.stack(lx).long()
				y = torch.stack(ly).long()

		_, loss = model(x, y)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		if step % opt.train['log_interval'] == 0:
			print(f"step {step} | loss {loss.item():.4f}")
		if (step + 1) % opt.train["save_interval"] == 0:
			shared.format.save_training_checkpoint(
				ckpt_path,
				model,
				optimizer,
				next_step=step + 1,
			)

if __name__ == '__main__':
	exit(main())
