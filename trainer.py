
# import pickle # This module is considered extremely dangerous since arbitary codes can be executed by it. Please consider using safe versions.
from typing import cast, Any
from shared.util import notify_confirm, get_batch

from shared.model import GPT, GPTOption
from dataclasses import dataclass
import tomllib
import os

import torch
import torch.nn as nn

import shared.format

# Interface functions, consider build a json scheme for better management across different 


def main():
	meta_opt = load_meta_dataset('trainer.toml')
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
	
	model_path: str = opt.train['working_directory']
	os.makedirs(model_path, exist_ok=True)
	
	model = shared.format.cased_model(opt)
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
		model, optimizer, start_step = shared.format.load_checkpoint(ckpt_path, model, optimizer)
	
	for step in range(start_step, opt.train['max_steps']):
		x, y = get_batch(opt.train['dataset_train'],
			opt.train['corpus_block_size'], opt.train['corpus_batch_size'])

		_, loss = model(x, y)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		if step % opt.train['log_interval'] == 0:
			print(f"step {step} | loss {loss.item():.4f}")
		if step > start_step and step % opt.train['save_interval'] == 0:
			shared.format.save_checkpoint(os.path.join(model_path, 'ckpt.pt'), model, optimizer, step)

if __name__ == '__main__':
	exit(main())
