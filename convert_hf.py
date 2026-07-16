from __future__ import annotations

import argparse
import gc
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM

import shared.format


def resolve_dtype(name: str) -> torch.dtype:
	dtypes = {
		"float32": torch.float32,
		"bfloat16": torch.bfloat16,
		"float16": torch.float16,
	}
	return dtypes[name]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert a Hugging Face model to a shark-lab checkpoint."
	)
	parser.add_argument("--config", default="options.toml")
	parser.add_argument("--source", default=None)
	parser.add_argument("--output", default=None)
	parser.add_argument(
		"--dtype",
		choices=["float32", "bfloat16", "float16"],
		default="float32",
	)
	args = parser.parse_args()

	meta_opt = shared.format.load_meta_dataset(args.config)
	opt = shared.format.trainer_options(
		meta_opt["model"],
		meta_opt["train"],
	)

	source_path = args.source or opt.train["tokenizer_path"]
	output_path = args.output or os.path.join(
		opt.train["working_directory"],
		"pretrained.pt",
	)

	# Conversion is deliberately performed on CPU.
	torch.set_default_device("cpu")
	torch.set_default_dtype(resolve_dtype(args.dtype))

	local_model = shared.format.model_from_scratch(opt)

	print(f"loading Hugging Face model: {source_path}")

	hf_model = AutoModelForCausalLM.from_pretrained(
		source_path,
		torch_dtype="auto",
		low_cpu_mem_usage=True,
		device_map="cpu",
	)

	shared.format.load_hf_state_dict(
		local_model,
		hf_model.state_dict(),
	)

	# Only do this when your GPT architecture requires tied embeddings.
	local_model.lm_head.weight = local_model.model.embed_tokens.weight

	if (
		local_model.lm_head.weight.data_ptr()
		!= local_model.model.embed_tokens.weight.data_ptr()
	):
		raise RuntimeError("embedding weights were not tied")

	shared.format.save_model_checkpoint(
		output_path,
		local_model,
		step=0,
	)

	del hf_model
	del local_model
	gc.collect()

	print("conversion completed successfully")


if __name__ == "__main__":
	main()
