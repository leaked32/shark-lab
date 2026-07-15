import torch
from torch import Tensor

def cross_entropy(logits: Tensor, targets: Tensor, ignore_index=-100):
	# logits: (B, T, V) -> (B*T, V)
	# targets: (B, T) -> (B*T)
	logits = logits.view(-1, logits.size(-1))
	targets = targets.view(-1)
	
	"""	Before the loss, the model produces:
		input_ids:
		[t0] [t1] [t2] [t3]
		The decoder predicts the next token at each position:
		logits:
		predict t1  predict t2  predict t3  predict t4
		
		So for causal LM training, we usually create:
		logits = logits[:, :-1, :]
		targets = targets[:, 1:]
		Conceptually:
		logit position 0  → target t1
		logit position 1  → target t2
		logit position 2  → target t3
		Now they align.
		
		Note, this is unchanged in SFT, please give it a shot~
	"""
	valid_mask = targets != ignore_index
	logits = logits[valid_mask]
	targets = targets[valid_mask]
	
	# Cross entropy needs competition between all tokens
	log_probs = torch.log_softmax(logits, dim=-1)

	rows = torch.arange(targets.size(0))
	correct_log_probs = log_probs[rows, targets]

	return -correct_log_probs.mean()
