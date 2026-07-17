"""
shark-lab
shared/util.py

This module collects discrete shared functions
"""

from typing import Any

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
		

def get_batch(path: str, block_size: int = 256, batch_size: int = 16) -> tuple[Tensor, Tensor]:
	data = np.memmap(path, dtype=np.uint16, mode='r')

	ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)

	x = []
	y = []

	for i in ix:
		x.append(torch.from_numpy(data[i:i+block_size].astype(np.int64)))
		y.append(torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)))

	x = torch.stack(x).long()
	y = torch.stack(y).long()

	return x, y


def resolve_runtime(
	system_opt: dict[str, Any],
	device_override: str | None = None,
	dtype_override: str | None = None,
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
