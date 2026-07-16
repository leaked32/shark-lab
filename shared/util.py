from typing import Any

import torch
import numpy as np
from torch import Tensor as Tensor

from tokenizers import Tokenizer

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

# =================================================================================================
# TOKENIZER
# =================================================================================================

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



def text_idx(
	tokenizer: Tokenizer,
	text: str,
	device: torch.device,
) -> Tensor:
	ids = tokenizer.encode(
		text,
		add_special_tokens=False,
	).ids

	if not ids:
		raise ValueError("encoded prompt is empty")

	return torch.tensor(
		[ids],
		dtype=torch.long,
		device=device,
	)


def idx_text(
	tokenizer: Tokenizer,
	output: Tensor,
	prompt_length: int,
) -> str:
	generated_ids = (
		output[0, prompt_length:]
		.detach()
		.cpu()
		.tolist()
	)

	return tokenizer.decode(
		generated_ids,
		skip_special_tokens=True,
	).strip()

"""

		
		ids = tokenizer.encode(
			text,
			add_special_tokens=False,
		).ids

		idx = torch.tensor(
			[ids],
			dtype=torch.long,
			device=device,
		)
		
"""
