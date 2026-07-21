import argparse
import os
import random
from typing import cast

from tokenizers import Tokenizer
import torch
from torch import Tensor

import shark.format
import shark.util


def make_sft_batch(
	dataset_context: shark.format.JsonlDataset,
	tokenizer: Tokenizer,
	eos_token_id: int,
	batch_count: int,
	valid_indices: list[int],
	index_queue: list[int],
) -> tuple[Tensor, Tensor]:
	"""
	Create one padded SFT batch.

	If a previously valid record fails during training, it is removed from
	valid_indices. The function fails cleanly if no usable records remain.
	"""
	lx: list[Tensor] = []
	ly: list[Tensor] = []
	max_len = 0

	while len(lx) < batch_count:
		if not valid_indices:
			raise RuntimeError(
				"No valid SFT records remain while creating a batch."
			)

		if not index_queue:
			index_queue.extend(valid_indices)
			random.shuffle(index_queue)

		cindex = index_queue.pop()

		try:
			cx, cy = dataset_context.to_sft_tensors(
				tokenizer,
				-1,
				cindex,
			)

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
				f"Removing failed SFT record "
				f"index={cindex}: {type(exc).__name__}: {exc}"
			)

			try:
				valid_indices.remove(cindex)
			except ValueError:
				pass

			index_queue[:] = [
				index
				for index in index_queue
				if index != cindex
			]

			continue

		lx.append(cx)
		ly.append(cy)
		max_len = max(max_len, cx.size(0))

	if not lx:
		raise RuntimeError("Unable to construct a non-empty SFT batch.")

	for i in range(len(lx)):
		lx[i] = shark.util.enlarge_to_fit(lx[i], max_len, eos_token_id,)
		ly[i] = shark.util.enlarge_to_fit(ly[i], max_len, -1,)

	x = torch.stack(lx, dim=0).long()
	y = torch.stack(ly, dim=0).long()

	return x, y


def check_finite(name: str, x: Tensor):
	if not torch.isfinite(x).all():
		print(
			f"BAD TENSOR: {name}",
			"max=", x.max().item(),
			"min=", x.min().item(),
		)
		raise RuntimeError(f"Non-finite tensor: {name}")

def main() -> int:
	parser = argparse.ArgumentParser(
		description="Train a GPT model from a configured dataset."
	)
	parser.add_argument(
		"--config",
		default="options.toml",
		help="Path to the TOML configuration file.",
	)
	args = parser.parse_args()

	opt = shark.format.load_trainer_options(args.config)
	# opt_sys = meta_opt["system"]

	dtype_name = opt.system.dtype
	dtype_map = {
		"float32": torch.float32,
		"bfloat16": torch.bfloat16,
		"float16": torch.float16,
	}

	if dtype_name not in dtype_map:
		raise ValueError(f"Unsupported dtype: {dtype_name!r}")

	torch.set_default_dtype(dtype_map[opt.system.dtype])
	torch.set_default_device(opt.system.device)

	torch.set_num_threads(16)
	torch.set_num_interop_threads(2)
	torch.backends.cuda.enable_flash_sdp(False)
	torch.backends.cuda.enable_mem_efficient_sdp(False)
	torch.backends.cuda.enable_math_sdp(True)

	tokenizer_path = opt.general.tokenizer_path
	tokenizer, eos_token_id = shark.format.get_tokenizer(
		tokenizer_path
	)
	
	os.makedirs(opt.general.working_directory, exist_ok=True)

	model = shark.format.model_from_scratch(opt)

	if model is None:
		raise RuntimeError(
			"shark.format.model_from_scratch returned None."
		)

	optimizer = model.optimizer_adamw(
		opt.train.adamw_weight_decay,
		opt.train.optimizer_learning_rate,
		(
			opt.train.adamw_beta1,
			opt.train.adamw_beta2,
		),
		opt.system.device,
	)

	ckpt_path = os.path.join(opt.general.working_directory, "ckpt.pt")
	start_step = 0

	if os.path.exists(ckpt_path):
		start_step = shark.format.load_training_checkpoint(
			ckpt_path,
			model,
			optimizer,
			map_location=opt.system.device,
		)

		print(f"Resuming training from step {start_step}.")

	dataset_type: int = cast(int, opt.train.dataset_type)
	batch_count = cast(int, opt.train.batch_count)
	max_steps = cast(int, opt.train.max_steps)

	if batch_count <= 0:
		raise ValueError(f"batch_count must be positive, got {batch_count}.")

	if max_steps < 0:
		raise ValueError(f"max_steps must not be negative, got {max_steps}.")

	dataset_context = None
	valid_indices: list[int] = []
	index_queue: list[int] = []

	match dataset_type:
		case 0:
			dataset_train = opt.train.dataset_train

			if not os.path.exists(dataset_train):
				raise FileNotFoundError(
					f"Training dataset does not exist: "
					f"{dataset_train}"
				)

		case 1:
			jsonl_path = opt.train.dataset_sft_train

			if not os.path.exists(jsonl_path):
				raise FileNotFoundError(
					f"SFT dataset does not exist: {jsonl_path}"
				)

			print(f"SFT JSONL path: {jsonl_path}")

			dataset_context = shark.format.JsonlDataset(jsonl_path)
			valid_indices, invalid_indices = dataset_context.validate_sft_indices(tokenizer)
			
			if len(invalid_indices) > 0:
				shark.util.notify_confirm(
					"Some data in jsonl cannot be used and will be omitted. "
					f"invalid_indices {invalid_indices}"
				)
			# else:
			# 	shark.util.notify_confirm("All data in jsonl are valid.")

			index_queue = valid_indices.copy()
			random.shuffle(index_queue)

		case _:
			raise ValueError(
				f"Unsupported dataset_type: {dataset_type}"
			)

	model.train()

	for step in range(start_step, max_steps):
		x: Tensor
		y: Tensor

		match dataset_type:
			case 0:
				x, y = shark.util.get_batch(
					opt.train.dataset_train,
					opt.train.corpus_block_size,
					batch_count,
				)

			case 1:
				if dataset_context is None:
					raise RuntimeError(
						"SFT dataset was not initialized."
					)

				x, y = make_sft_batch(
					dataset_context=dataset_context,
					tokenizer=tokenizer,
					eos_token_id=eos_token_id,
					batch_count=batch_count,
					valid_indices=valid_indices,
					index_queue=index_queue,
				)

			case _:
				raise RuntimeError(
					f"Unexpected dataset_type: {dataset_type}"
				)

		optimizer.zero_grad(set_to_none=True)

		_, loss = model(x, y)

		if not torch.isfinite(loss):
			raise FloatingPointError(
				f"Non-finite loss at step {step}: "
				f"{loss.detach().item()}"
			)

		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,)
		
		optimizer.step()

		if step % opt.train.log_interval == 0:
			print(
				f"step {step} | "
				f"loss {loss.detach().item():.4f}"
			)

		if (step + 1) % opt.train.save_interval == 0:
			ckpt_path1 = os.path.join(opt.general.working_directory, f"ckpt.{step + 1}.pt")
			shark.format.save_training_checkpoint(
				ckpt_path1 if opt.train.save_independent_checkpoints else ckpt_path,
				model,
				optimizer,
				next_step=step + 1,
			)

	if max_steps > start_step:
		shark.format.save_training_checkpoint(
			ckpt_path,
			model,
			optimizer,
			next_step=max_steps,
		)

		print(f"Training complete. Final checkpoint: {ckpt_path}")
	else:
		print(
			f"No training performed: start_step={start_step}, "
			f"max_steps={max_steps}."
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
