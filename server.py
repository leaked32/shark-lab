from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from threading import Thread, Lock, Event
import time
import shared.model

import argparse
import os
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import shared.format
import tkinter
import tkinter.ttk as ttk

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
	stop_reason: str | None = None

class SlotManager:
	def __init__(self, size: int):
		self.slots: list[GenerateRequest|None] = [None for _ in range(size)]

	def allocate(self, req):
		for i in range(len(self.slots)):
			if self.slots[i] is None:
				self.slots[i] = req
				req.slot = i
				return i

		raise RuntimeError("no free slot")

	def release(self, idx):
		self.slots[idx] = None

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
	def __init__(self, model: shared.model.GPT,
			tokenizer: PreTrainedTokenizerBase, max_batch: int,
			temperature: float=1.0, top_k: int|None=None,
			stop_strings: tuple[str, ...]=("\nChumbud:", "\nScenario:")):
		self.next_id = 0
		self.temperature = temperature
		self.top_k = top_k
		self.lock = Lock()

		self.model = model
		self.tokenizer = tokenizer
		self.eos_token_id = tokenizer.eos_token_id
		self.stop_sequences = {
			text: torch.tensor(
				tokenizer.encode(text, add_special_tokens=False),
				dtype=torch.long
			)
			for text in stop_strings
		}
		if any(ids.numel() == 0 for ids in self.stop_sequences.values()):
			raise ValueError("stop strings must encode to at least one token")

		self.requests: dict[int, GenerateRequest] = {}
		self.slots = SlotManager(max_batch)
		self.state = shared.model.DecoderState(
			[shared.model.KVCache1(max_batch) for _ in range(model.opt.layer)]
		)

	def add(self, input_ids: Tensor, max_new_tokens: int):
		if max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be positive")
		if input_ids.size(1) + max_new_tokens > 1024:
			raise ValueError(
				"prompt plus generated tokens exceeds KV-cache capacity"
			)

		with self.lock:
			device = self.model.get_device()
			input_ids = input_ids.to(device)
			assert input_ids.dim() == 2 and input_ids.size(0) == 1

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

	def _active_requests(self):
		return [req for _, req in self.slots.active() if not req.finished]

	def _make_decode_batch(self, exclude: set[int] | None=None):
		device = self.model.get_device()
		exclude = set() if exclude is None else exclude
		active = [
			req for _, req in self.slots.active()
			if req.prefilled and not req.finished and req.id not in exclude
		]
		if not active:
			return None, [], torch.empty(0, dtype=torch.long, device=device)

		inputs = torch.cat(
			[req.input_ids[:, -1:] for req in active], dim=0
		).to(device)
		active_slots = torch.tensor(
			[req.slot for req in active],
			dtype=torch.long, device=device
		)
		return inputs, active, active_slots

	def _reset_slot_cache(self, slot: int):
		if self.state.kv_cache is None:
			return
		for cache in self.state.kv_cache:
			cache.clear_slot(slot)

	def _finish(self, req: GenerateRequest, reason: str):
		if req.finished:
			return
		req.finished = True
		req.stop_reason = reason
		slot = req.slot
		self._reset_slot_cache(slot)
		self.slots.release(slot)
		req.slot = -1

	def _matching_stop_sequence(self, req: GenerateRequest) -> tuple[str, int] | None:
		generated = req.input_ids[0, req.prompt_len:]

		for text, ids_cpu in self.stop_sequences.items():
			size = ids_cpu.numel()
			if generated.numel() < size:
				continue

			ids = ids_cpu.to(generated.device)
			if torch.equal(generated[-size:], ids):
				return text, size

		return None

	def _accept_sampled_token(self, req: GenerateRequest, token: Tensor) -> bool:
		token_id = int(token.item())

		if self.eos_token_id is not None and token_id == self.eos_token_id:
			self._finish(req, "eos")
			return True

		req.input_ids = torch.cat(
			(req.input_ids, token.reshape(1, 1)), dim=1
		)
		req.generated_tokens += 1

		match = self._matching_stop_sequence(req)
		if match is not None:
			text, token_count = match
			req.input_ids = req.input_ids[:, :-token_count]
			req.generated_tokens -= token_count
			self._finish(req, f"stop string: {text!r}")
			return True

		if req.generated_tokens >= req.max_new_tokens:
			self._finish(req, "max_new_tokens")
			return True

		return False

	@torch.no_grad()
	def _prefill(self, req: GenerateRequest):
		device = self.model.get_device()
		self.state.active_slots = torch.tensor(
			[req.slot], dtype=torch.long, device=device
		)
		logits, _ = self.model.forward(
			req.input_ids.to(device), state=self.state
		)
		token = self.model.sample_next_token(
			logits, self.temperature, self.top_k
		)
		req.prefilled = True
		self._accept_sampled_token(req, token)

	@torch.no_grad()
	def _decode(self, active: list[GenerateRequest], active_slots: Tensor):
		inputs = torch.cat(
			[req.input_ids[:, -1:] for req in active], dim=0
		)
		self.state.active_slots = active_slots
		logits, _ = self.model.forward(inputs, state=self.state)
		next_tokens = self.model.sample_next_token(
			logits, self.temperature, self.top_k
		)

		for row, req in enumerate(active):
			self._accept_sampled_token(req, next_tokens[row])

	@torch.no_grad()
	def step(self):
		just_prefilled: set[int] = set()

		for req in list(self._active_requests()):
			if not req.prefilled:
				self._prefill(req)
				just_prefilled.add(req.id)

		_, active, active_slots = self._make_decode_batch(
			exclude=just_prefilled
		)
		if active:
			self._decode(active, active_slots)

	@torch.no_grad()
	def run_until_done(self):
		while self.has_active():
			self.step()
		return self.requests


def set_torch_options(system_opt: dict[str, Any]) -> str:
	dtype_name = system_opt.get('dtype', 'float32')
	device = system_opt.get('device', 'cpu')
	dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
	if dtype_name not in dtypes:
		raise ValueError(f"unsupported dtype: {dtype_name}")
	torch.set_default_dtype(dtypes[dtype_name])
	torch.set_default_device(device)
	return device

def encode_prompt(tokenizer: Any, prompt: str, device) -> torch.Tensor:
	ids = tokenizer.encode(prompt, add_special_tokens=False)
	if len(ids) == 0:
		if tokenizer.eos_token_id is None:
			raise ValueError("empty prompt and tokenizer has no eos_token_id")
		ids = [tokenizer.eos_token_id]
	return torch.tensor([ids], dtype=torch.long, device=device)

def decode_tokens(tokenizer: Any, tokens: torch.Tensor) -> str:
	ids = tokens[0].detach().cpu().tolist()
	return tokenizer.decode(ids, skip_special_tokens=False)

def demon_ui(engine: GenerationEngine, tokenizer: AutoTokenizer):
	root = tkinter.Tk()
	root.title("LLM Debugger")
	root.geometry("800x600")

	nbk = ttk.Notebook(root)

	def frm0():
		worker_stop = Event()

		def engine_worker():
			while not worker_stop.is_set():
				try:
					with engine.lock:
						if engine.has_active():
							engine.step()
						else:
							time.sleep(0.01)
				except Exception as exc:
					root.after(
						0,
						lambda exc=exc: generation_failed(exc)
					)
					time.sleep(0.05)

		Thread(target=engine_worker, daemon=True).start()
		
		frm = ttk.Frame(nbk, padding=8)

		requests_text = tkinter.Text(frm, height=10, wrap="none")
		response_text = tkinter.Text(frm, height=18, wrap="word")
		input_text = ttk.Entry(frm)
		status_var = tkinter.StringVar(value="Ready")

		def set_text(widget: tkinter.Text, text: str, disabled: bool=True):
			widget.configure(state="normal")
			widget.delete("1.0", tkinter.END)
			widget.insert("1.0", text)

			if disabled:
				widget.configure(state="disabled")

			widget.see(tkinter.END)

		def callback_0(event=None):
			prompt = input_text.get().strip()
			if not prompt:
				return

			if all(slot is not None for slot in engine.slots.slots):
				status_var.set("Engine queue full")
				update_submit_state()
				return

			input_text.delete(0, tkinter.END)

			try:
				req = engine.add(
					encode_prompt(
						tokenizer,
						prompt,
						engine.model.get_device()
					),
					128
				)
			except Exception as exc:
				generation_failed(exc)
				return

			status_var.set(f"Queued request {req.id}")
			update_submit_state()

		def update_submit_state():
			is_full = all(slot is not None for slot in engine.slots.slots)
			state = "disabled" if is_full else "normal"

			input_text.configure(state=state)
			submit_btn.configure(state=state)

			if is_full:
				status_var.set("Engine queue full")
			elif status_var.get() == "Engine queue full":
				status_var.set("Ready")
		
		def generation_finished(req: GenerateRequest, generated_text: str):
			generated_texts.append((req.id, generated_text))

			response_text.configure(state="normal")
			response_text.delete("1.0", tkinter.END)

			for request_id, text in generated_texts:
				response_text.insert(
					tkinter.END,
					f"[request {request_id}]\n{text}\n\n"
				)

			response_text.configure(state="disabled")
			response_text.see(tkinter.END)

			status_var.set("Ready")
			update_submit_state()
			input_text.focus_set()

		def generation_failed(exc: Exception):
			set_text(response_text, f"{type(exc).__name__}: {exc}")
			input_text.configure(state="normal")
			submit_btn.configure(state="normal")
			status_var.set("Generation failed")
			input_text.focus_set()
		
		generated_texts: list[tuple[int, str]] = []
		displayed_requests: set[int] = set()

		def refresh_requests():
			requests_text.configure(state="normal")
			requests_text.delete("1.0", tkinter.END)

			with engine.lock:
				active = engine.slots.active()
				requests_snapshot = list(engine.requests.values())

				for slot, req in active:
					requests_text.insert(
						tkinter.END,
						f"[slot {slot}] "
						f"id={req.id} "
						f"prefilled={req.prefilled} "
						f"finished={req.finished} "
						f"generated={req.generated_tokens}/{req.max_new_tokens}\n"
					)

					text = decode_tokens(tokenizer, req.input_ids)
					requests_text.insert(tkinter.END, text + "\n\n")

				for req in requests_snapshot:
					if req.finished and req.id not in displayed_requests:
						generated_ids = req.input_ids[:, req.prompt_len:]
						generated_texts.append((
							req.id,
							decode_tokens(tokenizer, generated_ids)
						))
						displayed_requests.add(req.id)

			if not active:
				requests_text.insert("1.0", "No active requests.\n")

			response_text.configure(state="normal")
			response_text.delete("1.0", tkinter.END)

			for request_id, text in generated_texts:
				response_text.insert(
					tkinter.END,
					f"[request {request_id}]\n{text}\n\n"
				)

			response_text.configure(state="disabled")
			response_text.see(tkinter.END)

			update_submit_state()
			root.after(100, refresh_requests)
		
		
		submit_btn = ttk.Button(
			frm,
			text="Submit",
			command=callback_0
		)
		status_label = ttk.Label(
			frm,
			textvariable=status_var
		)

		response_text.configure(state="disabled")

		requests_text.grid(
			row=0,
			column=0,
			columnspan=2,
			sticky="nsew",
			pady=(0, 8)
		)
		response_text.grid(
			row=1,
			column=0,
			columnspan=2,
			sticky="nsew",
			pady=(0, 8)
		)
		input_text.grid(
			row=2,
			column=0,
			sticky="ew",
			padx=(0, 8)
		)
		submit_btn.grid(
			row=2,
			column=1,
			sticky="ew"
		)
		status_label.grid(
			row=3,
			column=0,
			columnspan=2,
			sticky="w",
			pady=(6, 0)
		)

		frm.columnconfigure(0, weight=1)
		frm.rowconfigure(0, weight=1)
		frm.rowconfigure(1, weight=2)

		input_text.bind("<Return>", callback_0)
		input_text.focus_set()

		refresh_requests()

		return frm

	nbk.add(frm0(), text="LLM Debug")
	nbk.pack(fill="both", expand=True)

	root.mainloop()

def main() -> None:
	parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
	parser.add_argument('--config', default='options.toml')
	parser.add_argument('--ckpt', default=None)
	# parser.add_argument('--prompt', default='')
	parser.add_argument('--max-new-tokens', type=int, default=100)
	parser.add_argument('--temperature', type=float, default=1.0)
	parser.add_argument('--top-k', type=int, default=None)
	parser.add_argument('--seed', type=int, default=None)
	parser.add_argument('--device', default=None)
	parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default=None)
	args = parser.parse_args()

	if args.seed is not None:
		torch.manual_seed(args.seed)

	meta_opt = shared.format.load_meta_dataset(args.config)
	device = set_torch_options(meta_opt['system'])

	tokenizer_path = meta_opt['train']['tokenizer_path']
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

	opt = shared.format.trainer_options(meta_opt['model'], meta_opt['train'])
	model = shared.format.model_from_scratch(opt)
	ckpt_path = args.ckpt or os.path.join(meta_opt['train']['working_directory'], 'ckpt.pt')
	step = shared.format.load_model_checkpoint(model, ckpt_path, device)
	model.eval()
	print(f"loaded checkpoint step {step}")
	
	engine = GenerationEngine(
		model,
		tokenizer,
		max_batch=4,
		stop_strings=("\nChumbud:", "\nScenario:")
	)
	demon_ui(engine, tokenizer)
	"""
	engine.add()

	while True:
		prompt: str = input('prompt: ')
		idx = encode_prompt(tokenizer, prompt, device)
		with torch.no_grad():
			out = model.generate(idx, args.max_new_tokens,
						temperature=args.temperature, top_k=args.top_k)

		print(decode_tokens(tokenizer, out))
		"""

if __name__ == '__main__':
	main()
