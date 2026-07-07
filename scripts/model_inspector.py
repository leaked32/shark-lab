import random
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================================================================
# INSPECTOR: TOKENIZER TOOLSET
# ================================================================================


def print_title(title: str) -> None:
	print()
	print("=" * 80)
	print(title)
	print("=" * 80)

def inspect_tokenizer(tokenizer):
	print_title("Tokenizer")

	print(type(tokenizer))
	print(f"Vocabulary: {len(tokenizer):,}")
	print(f"BOS: {tokenizer.bos_token!r}; EOS: {tokenizer.eos_token!r}; PAD: {tokenizer.pad_token!r}; UNK: {tokenizer.unk_token!r}")

def count_token_in_class(vocab):
	counter = Counter()

	leading_space = 0
	ascii_only = 0
	non_ascii = 0
	alphabetic = 0
	digit = 0
	punctuation = 0
	special = 0

	for token in vocab:
		counter[len(token)] += 1
		if token.startswith("▁") or token.startswith("Ġ"):
			leading_space += 1
		if token.startswith("<") and token.endswith(">"):
			special += 1
		if all(ord(c) < 128 for c in token):
			ascii_only += 1
		else:
			non_ascii += 1
		if any(c.isalpha() for c in token):
			alphabetic += 1
		if any(c.isdigit() for c in token):
			digit += 1
		if any(not c.isalnum() and not c.isspace() for c in token):
			punctuation += 1

	print_title("Statistics")

	print(f"ASCII only: {ascii_only:,}; Non ASCII: {non_ascii:,}; Alphabetic: {alphabetic:,}; Digit: {digit:,}; Punctuation: {punctuation:,}; Leading space: {leading_space:,}; Special: {special:,}")
	print(f"Distribution {counter}")

def sample_tokens_class(id_to_token, tokenizer, count: int):
	print_title(f"{count} Random Tokens")
	for token_id in sorted(random.sample(range(len(tokenizer)), count)):
		print(f"{token_id:5d} : {repr(id_to_token[token_id])}")


# ================================================================================
# INSPECTOR: MODEL
# ================================================================================


# ================================================================================
# SCRIPT LAUNCHER
# ================================================================================

def main() -> int:
	model_path = "/home/menv/src/trained_llm/SmolLM2-360M"

	model = AutoModelForCausalLM.from_pretrained(
		model_path, dtype=torch.bfloat16, device_map="auto")
	print(model)
	return 0
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	if tokenizer.pad_token is None:
		raise Exception("Invalid pad_token")
		tokenizer.pad_token = tokenizer.eos_token

	vocab = tokenizer.get_vocab()
	id_to_token = {v: k for k, v in vocab.items()}

	# tok = AutoTokenizer.from_pretrained("~/src/trained_llm/SmolLM2-360M")
	tokenizer.save_pretrained("~/src/tokenizers/smollm2_tokenizer")
	

if __name__ == '__main__':
	exit(main())
