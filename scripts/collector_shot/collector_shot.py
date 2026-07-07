import json
import time
from datetime import datetime
import requests
import tomllib
import re
import os

from typing import Any
from dataclasses import dataclass


# =========================
# MAIN LOOP
# =========================

WIDTH = 100
LINE = "+" + "-" * (WIDTH - 1) + "+"

def row(key: str, value: str):
	value = value[:WIDTH - 13]
	print(f"| {key:<9}: {value:<{WIDTH - 13}}|")

def print_header(scenarios: list[str], scenario_index: int, epoch: int):
	row("Epoch", str(epoch))
	# Do not change, index should align with python value for easy debugging
	row("Scenario", f"{scenario_index} / {len(scenarios) - 1}")
	row("Name", scenarios[scenario_index])
	print(LINE)

def print_result(ok: bool, ok_count: int, total: int, error: str | None = None):
	status = "PASS" if ok else "FAIL"
	row("Status", status)
	if error is not None:
		row("Error", error or "")
	row("Success", f"{ok_count} / {total}    {ok_count/max(total,1):.1%}")
	print(LINE)


def now():
	return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def load_meta_dataset(path: str):
	with open(path, "rb") as f:
		return tomllib.load(f)


# =========================
# CONFIG
# =========================

@dataclass
class collector_options:
	meta_options: dict[str, Any]
	meta_dataset: dict[str, Any] 

def find_element_in_case(x: list, case: str) -> Any:
	for xx in x:
		# print(xx)
		if xx['case'] == case and xx['enabled'] == True:
			return xx['data']
	raise Exception(f'find_element_in_case cannot find such case: {case}')

# PROMPT, BANNED_TERMS = get_prompt(mode)

# =========================
# LLM CALL
# =========================

def call_llm(options: collector_options, prompt):
	payload = {
		"model": "default",
		"messages": [{"role": "system", "content": prompt}],
		"temperature": options.meta_options['config']['temperature'],
		"max_tokens": options.meta_options['config']['max_tokens'],
	}

	r = requests.post(options.meta_options['config']['model_url'], json=payload, timeout=600)
	r.raise_for_status()
	return r.json()["choices"][0]["message"]["content"]

# =========================
# VALIDATION
# =========================


DIALOGUE_RE = re.compile(r'"([^"\n]*)"')


def raise_if_invalid_message_format(content: str):
	remaining = DIALOGUE_RE.sub("", content)

	for sym in ('"', "(", ")"):
		if sym in remaining:
			raise Exception(f"forbidden character detected: {sym}")


def merge_duplicate_roles(messages: list[dict[str, str]]) -> list[dict[str, str]]:
	merged: list[dict[str, str]] = []

	for message in messages:
		if merged and merged[-1]["role"] == message["role"]:
			merged[-1]["content"] += "\n" + message["content"]
		else:
			merged.append(message)

	return merged


def build_dataset_from_dialogue(options: collector_options, text: str) -> dict[str, list[dict[str, str]]]:
	
	MAX_ACTION_ONLY_MESSAGES: float = options.meta_options["restrictions"]["max_actions_only_messages"]
	
	if text[0] == "\"" and text[-1] == "\"":
		text = text[1:-1]
	
	lines = [line.strip() for line in text.splitlines() if ":" in line]
	mode: str = options.meta_options['config']['mode']

	action_only_count = 0
	messages: list[dict[str, str]] = []

	for line in lines:
		try:
			role, content = line.split(":", 1)
		except ValueError:
			continue

		role = role.strip().lower()
		content = content.strip()
		content_lower = content.lower()

		raise_if_invalid_message_format(content)

		dialogues = DIALOGUE_RE.findall(content)
		dialogue = " ".join(
			d.strip()
			for d in dialogues
		)

		if not dialogue:
			action_only_count += 1

		for term in find_element_in_case(options.meta_options['censored_terms'], mode):
			if term in content_lower:
				raise Exception(f"censored term: {term}")

		if role == options.meta_dataset["character"]["name"].lower():
			role = "assistant"
		elif role == options.meta_dataset["character"]["peer"].lower():
			role = "user"
		else:
			raise Exception(f"unknown role: {role}")

		messages.append({
			"role": role,
			"content": content,
		})
	
	bmle = len(messages)
	
	messages = merge_duplicate_roles(messages)
	
	if bmle == 0 or len(messages) < 5:
		raise Exception("too few turns")
	# WARNING, len(messages) must be guaranteed to be greater than 0 here!
	if (action_only_count - bmle) / bmle > MAX_ACTION_ONLY_MESSAGES:
		raise Exception(
			f"too many action-only messages ({action_only_count})"
		)

	return {"messages": messages}

def validate(options: collector_options, text: str):
	try:
		obj = build_dataset_from_dialogue(options, text)
		if obj is None:
			return False, None, "json_unrecoverable"
		if "messages" not in obj:
			return False, None, "missing_messages"
		return True, obj, None
	except Exception as e:
		return False, None, str(e)

# =========================
# LOGGING (UNIFIED)
# =========================

def save_dataset(name_prefix: str, obj):
	with open(f'{name_prefix}.jsonl', "a", encoding="utf-8") as f:
		f.write(json.dumps(obj, ensure_ascii=False) + "\n")
		f.flush()
		os.fsync(f.fileno())

def save_failed(name_prefix: str, entry, obj):
	with open(f'{name_prefix}-failed.jsonl', "a", encoding="utf-8") as f:
		f.write(json.dumps({"entry": entry, "raw": obj}, ensure_ascii=False) + "\n")
		f.flush()
		os.fsync(f.fileno())
		
def log_debug(name_prefix: str, entry):
	with open(f'{name_prefix}-log.jsonl', "a", encoding="utf-8") as f:
		f.write(json.dumps(entry, ensure_ascii=False) + "\n")
		f.flush()
		os.fsync(f.fileno())

# =========================
# ROLLOUT
# =========================

def rollout(options: collector_options, scenario: str) -> tuple[bool, str | None]:
	character_name: str = options.meta_dataset['character']['name']
	mode: str = options.meta_options['config']['mode']
	prompt = find_element_in_case(options.meta_options['prompts'], mode)
	prompt = prompt.format(
		character_card=options.meta_dataset["character"]["card"],
		character_scenario=scenario,
		character_examples=options.meta_dataset["character"]["examples"]
	)
	
	raw = call_llm(options, prompt)
	ok, data, error = validate(options, raw)
	
	entry = {
		"timestamp": now(),
		"status": "ok" if ok else "fail",
		"scenario": scenario,
	}

	if not ok:
		if error is not None:
			entry["error"] = error
		else:
			entry["error"] = 'Exception is None'
	
	log_debug(options.meta_options['config']['output_prefix'], entry)

	if ok and isinstance(data, dict):
		def inject_scenario(messages, scenario):
			return [
				{
					"role": "system",
					"content": f"Scenario: {scenario}"
				}
			] + messages

		save_dataset(options.meta_options['config']['output_prefix'],  {
			"messages": inject_scenario(data["messages"], scenario)
		})
	else:
		save_failed(options.meta_options['config']['output_prefix'], entry, raw)

	return ok, error


def generate(options: collector_options) -> int:
	def get_dataset_index(scenarios: list[str], y: int| str) -> int:
		index: int
		if isinstance(y, int):
			index = y
		elif isinstance(y, str):
			index = scenarios.index(y)
		return index
	mode: str = options.meta_options['config']['mode']
	scenarios = find_element_in_case(options.meta_dataset['scenarios'], mode)
	scenario_index: int = get_dataset_index(scenarios, options.meta_options['config']['scenario_index'])
	epoch = options.meta_options['config']['epoch']
	
	print(f'scenario count: {len(scenarios)}')
	ok_count = 0
	total = 0
	
	while True:
		print()
		print(LINE)
		
		scenario = scenarios[scenario_index]
		print_header(scenarios, scenario_index, epoch)

		ok, error = rollout(options, scenario)

		if ok:
			ok_count += 1
		
		total += 1
		print_result(ok, ok_count, total, error)
		
		time.sleep(1)
		
		scenario_index = (scenario_index + 1)
		if scenario_index == len(scenarios):
			epoch -= 1
			if epoch == 0:
				print("task completed")
				return 0
			scenario_index = 0



if __name__ == "__main__":
	# This scirpt works in current working directory by default.
	
	options = collector_options(
		load_meta_dataset("meta_options.toml"),
		load_meta_dataset("meta_dataset.toml")
		)
	exit(generate(options))
