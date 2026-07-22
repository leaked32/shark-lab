"""
shark-lab
shark/util.py

This module collects discrete shared functions
"""

from typing import Any
from pathlib import Path

import torch

import numpy as np
from torch import Tensor as Tensor

def notify_confirm(msg: str):
	print(msg)
	x = input('Do you wish to continue? Y/ Yes or No: ').lower()
	if x == 'y' or  x == 'yes':
		return
	else:
		raise InterruptedError('manual exited')
		

def get_batch(path: str | Path, block_size: int = 256, batch_size: int = 16) -> tuple[Tensor, Tensor]:
	data = np.memmap(path, dtype=np.uint16, mode='r')

	ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)

	x = []
	y = []

	for i in ix:
		x.append(torch.from_numpy(data[i:i+block_size].astype(np.int64)))
		y.append(torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)))

	x = torch.stack(x).long()
	y = torch.stack(y).long()
	
	x = x.to(device=torch.get_default_device())
	y = y.to(device=torch.get_default_device())
	
	return x, y


def resolve_runtime(
	system_opt,
	device_override: str | None = None,
	dtype_override: str | None = None,
) -> tuple[torch.device, torch.dtype]:
	dtype_name = dtype_override or system_opt.dtype
	device_name = device_override or system_opt.device

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


def enlarge_to_fit(x: Tensor, least_len: int, fill_value: int) -> Tensor:
	"""Pad a one-dimensional tensor to at least least_len."""
	raw_len = x.size(0)

	if raw_len >= least_len:
		return x

	padding = x.new_full(
		(least_len - raw_len,),
		fill_value,
	)

	return torch.cat((x, padding), dim=0)


def report_tensor(name: str, tensor: Tensor) -> None:
	value = tensor.detach()
	
	finite = torch.isfinite(value)
	finite_values = value[finite]

	print(
		f"{name}: "
		f"shape={tuple(value.shape)} "
		f"dtype={value.dtype} "
		f"nan={torch.isnan(value).sum().item()} "
		f"+inf={torch.isposinf(value).sum().item()} "
		f"-inf={torch.isneginf(value).sum().item()}",
		flush=True,
	)

	if finite_values.numel() > 0:
		print(
			f"    finite_min={finite_values.min().item():.8e} "
			f"finite_max={finite_values.max().item():.8e} "
			f"finite_abs_max={finite_values.abs().max().item():.8e} "
			f"finite_mean={finite_values.float().mean().item():.8e}",
			flush=True,
		)

def report_largest_gradient_norms(
	model: torch.nn.Module,
	*,
	top_k: int = 10,
) -> None:
	results: list[tuple[float, str, float]] = []

	for name, parameter in model.named_parameters():
		gradient = parameter.grad

		if gradient is None:
			continue

		grad_fp32 = gradient.detach().float()

		norm = torch.linalg.vector_norm(grad_fp32).item()
		abs_max = grad_fp32.abs().max().item()

		results.append((norm, name, abs_max))

	results.sort(reverse=True)

	print("Largest parameter gradients:", flush=True)

	for norm, name, abs_max in results[:top_k]:
		print(
			f"    {name}: "
			f"norm={norm:.8e} "
			f"abs_max={abs_max:.8e}",
			flush=True,
		)

def report_nonfinite_gradients(
	model: torch.nn.Module,
	*,
	step: int,
) -> None:
	""" Checker should report all bad gradients, not raise at the first parameter in model order
	"""
	
	found = False

	for name, parameter in model.named_parameters():
		gradient = parameter.grad

		if gradient is None:
			continue

		nan_count = torch.isnan(gradient).sum().item()
		posinf_count = torch.isposinf(gradient).sum().item()
		neginf_count = torch.isneginf(gradient).sum().item()

		if nan_count == 0 and posinf_count == 0 and neginf_count == 0:
			continue

		found = True

		finite = torch.isfinite(gradient)
		finite_values = gradient[finite]

		finite_abs_max = (
			finite_values.abs().max().item()
			if finite_values.numel() > 0
			else float("nan")
		)

		print(
			f"BAD RAW GRADIENT at step {step}: "
			f"{name}\n"
			f"\tshape={tuple(gradient.shape)}\n"
			f"\tdtype={gradient.dtype}\n"
			f"\tnan={nan_count}\n"
			f"\t+inf={posinf_count}\n"
			f"\t-inf={neginf_count}\n"
			f"\tfinite_abs_max={finite_abs_max:.8e}",
			flush=True,
		)

	if found:
		raise FloatingPointError(
			f"Raw backward produced non-finite gradients at step {step}"
		)

def check_gradients_finite(
	model: torch.nn.Module,
	*,
	step: int,
) -> None:
	for name, parameter in model.named_parameters():
		gradient = parameter.grad

		if gradient is None:
			continue

		finite = torch.isfinite(gradient)

		if not finite.all():
			bad = (~finite).sum().item()
			total = gradient.numel()

			finite_values = gradient[finite]
			finite_abs_max = (
				finite_values.abs().max().item()
				if finite_values.numel() > 0
				else float("nan")
			)

			raise FloatingPointError(
				f"Non-finite gradient after backward at step {step}: "
				f"{name}, bad={bad}/{total}, "
				f"largest finite magnitude={finite_abs_max:.8e}"
			)


def check_parameters_finite(
	model: torch.nn.Module,
	*,
	step: int,
	location: str,
) -> None:
	for name, parameter in model.named_parameters():
		finite = torch.isfinite(parameter)

		if not finite.all():
			bad = (~finite).sum().item()

			raise FloatingPointError(
				f"Non-finite parameter {location} at step {step}: "
				f"{name}, bad={bad}/{parameter.numel()}"
			)


def check_finite(name: str, x: Tensor):
	if not torch.isfinite(x).all():
		print(
			f"BAD TENSOR: {name}",
			"max=", x.max().item(),
			"min=", x.min().item(),
		)
		raise RuntimeError(f"Non-finite tensor: {name}")
