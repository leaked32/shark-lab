import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import tomllib

# =========================
# CONSTANTS
# =========================

WIDTH = 100
LINE = "+" + "-" * (WIDTH - 2) + "+"


# =========================
# DATA STRUCTURES
# =========================


@dataclass
class GeneratorConfig:
    endpoint: str
    temperature: float
    max_tokens: int
    mode: str
    epochs: int
    scenario_index: int
    output_prefix: str
    failed_prefix: str
    debug_prompt_check: bool



@dataclass
class ValidationConfig:
    max_narration_score: int
    max_actions_density: float
    max_actions_only_ratio: float

@dataclass
class CollectorConfig:
    generator: GeneratorConfig
#     prompts: dict[str, PromptConfig]
    # censored_terms: dict[str, list[str]]
    validation: ValidationConfig


@dataclass
class CharacterConfig:
    name: str
    peer: str
    card: str


@dataclass
class DatasetCaseConfig:
    description: str
    scenarios: list[str]
    censored_terms: list[str]
    min_turns: int
    max_turns: int
    examples: str
    inject_prompts: str
    template: str


@dataclass
class DatasetConfig:
    character: CharacterConfig
    datasets: dict[str, DatasetCaseConfig]


# =========================
# IO
# =========================


def load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def now() -> str:
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


# =========================
# CONFIG PARSER
# =========================


def load_collector_config(path: str) -> CollectorConfig:
    raw = load_toml(path)
    generator = GeneratorConfig(**raw["generator"])
    # prompts = {name: PromptConfig(**value) for name, value in raw["prompts"].items()}
    # censored_terms = {
    #     name: value["terms"] for name, value in raw["censored_terms"].items() if value["enabled"]
    # }
    validation = ValidationConfig(**raw["validation"])
    return CollectorConfig(
        generator=generator, validation=validation
    )


def load_dataset_config(path: str) -> DatasetConfig:
    raw = load_toml(path)
    character = CharacterConfig(**raw["character"])
    datasets = {name: DatasetCaseConfig(**value) for name, value in raw["datasets"].items()}
    return DatasetConfig(character=character, datasets=datasets)


# =========================
# PRINT
# =========================


def row(key: str, value: str):
    value = value[: WIDTH - 14]
    print(f"| {key:<9}: {value:<{WIDTH - 14}}|")


def print_header(scenarios: list[str], scenario_index: int, epoch: int):
    print()
    print(LINE)
    row("Epoch", str(epoch))
    row("Scenario", f"{scenario_index} / {len(scenarios) - 1}")
    row("Name", scenarios[scenario_index])
    print(LINE)


def print_result(ok: bool, ok_count: int, total: int, error: str | None = None):
    row("Status", "PASS" if ok else "FAIL")

    if error:
        row("Error", error)

    row("Success", f"{ok_count} / {total} {ok_count / max(total, 1):.1%}")

    print(LINE)


def print_except(status: str, except1: str, try1: str):
    row("Status", status)
    row("Except", except1)
    row("Try", try1)
    print(LINE)


# =========================
# LLM
# =========================


def call_llm(config: CollectorConfig, prompt: str) -> str:
    retry_timeout = 90
    
    if config.generator.debug_prompt_check:
        print(prompt)
        print(len(prompt))
        exit(0)
    
    payload = {
        "model": "default",
        "messages": [{"role": "system", "content": prompt}],
        "temperature": config.generator.temperature,
        "max_tokens": config.generator.max_tokens,
    }

    while True:
        try:
            response = requests.post(config.generator.endpoint, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as e:

            print_except(
                "call_llm failed", f"{type(e).__name__}: {e}", f"retry after {retry_timeout}s"
            )

            time.sleep(retry_timeout)


# =========================
# VALIDATION
# =========================


DIALOGUE_RE = re.compile(r'"([^"\n]*)"')


def validate_message_format(content: str):
    remaining = DIALOGUE_RE.sub("", content)

    for symbol in ('"', "(", ")"):
        if symbol in remaining:
            raise Exception(f"forbidden character: {symbol}")


def merge_duplicate_roles(messages: list[dict[str, str]]):
    result = []

    for message in messages:
        if result and result[-1]["role"] == message["role"]:
            result[-1]["content"] += "\n" + message["content"]
        else:
            result.append(message)

    return result

ROLE_PATTERN = re.compile(
    r"^\s*(?:\*\*)?(?P<role>[^:*]+?)(?:\*\*)?\s*:\s*(?P<content>.*)$"
)


def split_role_content(line: str) -> tuple[str, str] | None:
    match = ROLE_PATTERN.match(line)

    if match is None:
        return None

    return (
        match.group("role").strip(),
        match.group("content").strip(),
    )

def build_dialogue_dataset(
    config: CollectorConfig, dataset: DatasetConfig, text: str, case: DatasetCaseConfig
):
    action_only = 0
    messages = []

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    for line in text.splitlines():

        if ":" not in line:
            continue

        sepa = split_role_content(line)
        if sepa is None:
            raise Exception(f"No role: {line}")
        else:
            role, content = sepa
        
        role = role.strip().lower()
        content = content.strip()

        validate_message_format(content)

        if not DIALOGUE_RE.findall(content):
            action_only += 1

        for term in case.censored_terms:
            if term.lower() in content.lower():
                raise Exception(f"censored term: {term}")

        if (
            role == dataset.character.name.lower() or
            dataset.character.name.lower().startswith(role)
            ):
            role = "assistant"
        elif role == dataset.character.peer.lower():
            role = "user"
        else:
            raise Exception(f"unknown role: {role}")

        messages.append({"role": role, "content": content})

    before_merge = len(messages)

    messages = merge_duplicate_roles(messages)

    if before_merge == 0 or len(messages) < case.min_turns or len(messages) > case.max_turns:
        raise Exception(f"invalid turn count {before_merge} {len(messages)}")

    if action_only / before_merge > config.validation.max_actions_only_ratio:
        raise Exception("too many action-only messages")

    return {"messages": messages}


# =========================
# SAVE
# =========================


def append_jsonl(path: str, obj: Any):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# GENERATION
# =========================


def rollout(
    config: CollectorConfig, dataset: DatasetConfig, case: DatasetCaseConfig, scenario: str
) -> tuple[bool, dict[str, Any]]:

    prompt = case.template.format(
        character_card=dataset.character.card,
        scenario=scenario,
        examples=case.examples,
        inject_prompts=case.inject_prompts,
    )

    raw = call_llm(config, prompt)
    try:
        result = build_dialogue_dataset(config, dataset, raw, case)
        result["messages"].insert(0, {"role": "system", "content": f"Scenario: {scenario}"})
        return True, result
    except Exception as e:
        return False, {"raw": raw, "except": str(e)}


def generate(config: CollectorConfig, dataset: DatasetConfig):
    case = dataset.datasets[config.generator.mode]
    scenarios = case.scenarios
    index = config.generator.scenario_index

    for epoch in range(config.generator.epochs):

        for scenario in scenarios[index:]:
            print_header(scenarios, index, epoch)

            x, result = rollout(config, dataset, case, scenario)
            if x:
                append_jsonl(config.generator.output_prefix + ".jsonl", result)
                print_result(True, 1, 1)
            else:
                append_jsonl(config.generator.failed_prefix + ".jsonl", result)
                print_result(False, 0, 1, result["except"])

            index += 1

            if index >= len(scenarios):
                index = 0


# =========================
# MAIN
# =========================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--options", default="options.toml")
    parser.add_argument("--dataset", default="dataset.toml")
    args = parser.parse_args()

    config = load_collector_config(args.options)
    dataset = load_dataset_config(args.dataset)
    generate(config, dataset)
