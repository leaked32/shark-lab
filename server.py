from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from threading import Thread
import shared.model

def test1():
	@dataclass
	class GenerateRequest:
		id: int
		input_ids: Tensor
		max_new_tokens: int
		slot: int = -1
		prompt_len: int = 0
		generated_tokens: int = 0
		prefilled: bool = False
		finished: bool = False


	class SlotManager:
		def __init__(self, size: int):
			self.slots: list[GenerateRequest|None] = [None for _ in range(size)]

		def allocate(self, req):
			for i in range(len(self.slots)):
				if self.slots[i] is None:
					self.slots[i] = req
					req.slot = 1
					return i

			raise RuntimeError("no free slot")

		def release(self, idx):
			self.slots[idx].request = None

		def active(self) -> list[tuple[int, GenerateRequest]]:
			return [
				(i, slot)
				for i, slot in enumerate(self.slots)
				if slot is not None
			]

	'''
	@dataclass
	class GenerationContext:
		requests: list[GenerateRequest]
		state: shared.model.DecoderState
	'''

	class GenerationEngine:
		def __init__(self, model: shared.model.GPT, max_batch: int, temperature: float=1.0, top_k: int|None=None):
			self.next_id = 0
			self.temperature = temperature
			self.top_k = top_k
			
			self.model = model
			self.requests: dict[int, GenerateRequest] = {}
			self.slots = SlotManager(max_batch)
			self.state = shared.model.DecoderState([shared.model.KVCache1(max_batch) for _ in range(model.opt.layer)])

		def add(self, input_ids: Tensor, max_new_tokens: int):
			device = self.model.get_device()
			input_ids = input_ids.to(device)
			assert input_ids.dim() == 2 and input_ids.size(0) == 1
			# allows only [[101, 2054, 2003, ...]] like input_ids
			req = GenerateRequest(
				id=self.next_id,
				input_ids=input_ids,
				max_new_tokens=max_new_tokens,
				prompt_len=input_ids.size(1)
			)
			self.next_id += 1
			self.slots.allocate(req)
			self.requests[req.id] = req
			return req

		def has_active(self):
			return any(not req.finished for req in self.requests.values())

		def active_requests(self):
			return [req for _, req in self.slots.active() if not req.finished]

		def make_decode_batch(self, exclude: set[int] | None=None):
			device = self.model.get_device()
			exclude = set() if exclude is None else exclude
			active = [
				req for _, req in self.slots.active()
				if req.prefilled and not req.finished and req.id not in exclude
			]
			if not active:
				return None, [], torch.empty(0, dtype=torch.long, device=device)
			inputs = torch.cat([req.input_ids[:, -1:] for req in active], dim=0).to(device)
			active_slots = torch.tensor([req.slot for req in active], dtype=torch.long, device=device)
			return inputs, active, active_slots

		def reset_slot_cache(self, slot: int):
			if self.state.kv_cache is None:
				return
			for cache in self.state.kv_cache:
				cache.clear_slot(slot)

		def finish(self, req: GenerateRequest):
			req.finished = True
			slot = req.slot
			self.reset_slot_cache(slot)
			self.slots.release(slot)
			req.slot = -1

		@torch.no_grad
		def prefill(self, req: GenerateRequest):
			device = self.model.get_device()
			self.state.active_slots = torch.tensor([req.slot], dtype=torch.long, device=device)
			logits, _ = self.model.forward(req.input_ids.to(device), state=self.state)
			token = self.model.sample_next_token(logits, self.temperature, self.top_k)
			req.input_ids = torch.cat((req.input_ids, token.view(1, 1)), dim=1)
			req.generated_tokens += 1
			req.prefilled = True
			if req.generated_tokens >= req.max_new_tokens:
				self.finish(req)

		@torch.no_grad
		def decode(self, active: list[GenerateRequest], active_slots: Tensor):
			inputs = torch.cat([req.input_ids[:, -1:] for req in active], dim=0)
			self.state.active_slots = active_slots
			logits, _ = self.model.forward(inputs, state=self.state)
			next_tokens = self.model.sample_next_token(logits, self.temperature, self.top_k)
			for row, req in enumerate(active):
				token = next_tokens[row]
				req.input_ids = torch.cat((req.input_ids, token.view(1, 1)), dim=1)
				req.generated_tokens += 1
				if req.generated_tokens >= req.max_new_tokens:
					self.finish(req)

		@torch.no_grad
		def step(self):
			just_prefilled: set[int] = set()
			for req in list(self.active_requests()):
				if not req.prefilled:
					self.prefill(req)
					just_prefilled.add(req.id)

			_, active, active_slots = self.make_decode_batch(exclude=just_prefilled)
			"""	Why exclude just_prefilled?
				The last token of the prompt has already produced the logits for the next token.
				Decode should consume one newly generated token.
			"""
			if not active:
				return
			self.decode(active, active_slots)

		@torch.no_grad
		def run_until_done(self):
			while self.has_active():
				self.step()
			return self.requests


if __name__ == "__main__":
	test1()

