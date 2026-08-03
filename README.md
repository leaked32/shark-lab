# shark-lab

`shark-lab` is my personal LLM learning and engineering project.

`shark-lab` is an experimental LLM framework for learning and implementing
GPT/LLaMA-style language models from scratch.

The project contains model implementations, training pipelines, inference tools,
Supervised Fine-Tuning (SFT) workflows, checkpoint conversion utilities, and
ROCm/HIP experiments for GPU programming and performance exploration.

This repository documents my learning and engineering process while building
LLM systems from the ground up. The project continuously evolves as new
features, optimizations, and experiments are explored.

## Features

* GPT/LLaMA-style Transformer model implementation
* Pre-training and Supervised Fine-Tuning (SFT)
* Custom training pipeline and checkpoint handling
* Inference and evaluation tools
* Hugging Face model conversion support
* ROCm/HIP experiments and GPU kernel development

## Project structure

```text
shared/
  model.py          Model architecture
  util.py           Common utility functions
  format.py         File and checkpoint format helpers

train.py            Pre-training and supervised fine-tuning
infer.py            Inference/testing entry point
convert_hf.py       Convert compatible Hugging Face models

config/
  360m.toml         Configuration for SmolLM2-360M-Instruct compatible models
  collector_options.toml

scripts/
  collector.py      Conversation generation tools

rocm/
  shark/            ROCm-related utilities
  hip-test/         HIP kernel experiments
  chatty/           Native C++ Dear ImGui LLM chat client
```

## Usage

### Training

```bash
python train.py --config options.toml
```

### Inference

```bash
python infer.py \
  --config options.toml \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50
```

### Convert Hugging Face checkpoints

```bash
python convert_hf.py \
  --config options.toml \
  --source HuggingFaceTB/SmolLM2-360M-Instruct \
  --output checkpoints/smollm2-instruct.pt
```

## ROCm / HIP experiments

`shark-lab` contains experimental ROCm code for learning GPU programming,
kernel development, and performance optimization.

Current experiments focus on understanding:

* GPU execution models
* Memory behavior
* Kernel implementation
* Numerical stability
* Performance optimization

Run ROCm experiments:

```bash
cd rocm

cmake --preset linux-x64-debug
cmake --build build/linux-x64-debug

./build/linux-x64-debug/hip-test/hip-test
```

For ROCm training runs, debugging asynchronous GPU failures may require:

```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
python train.py --config options.safe.big.toml
```

## Design notes

The project intentionally avoids depending on large external training
frameworks for core learning purposes.

Many components are implemented directly to understand the underlying
mechanisms of LLM training systems.

External components are still used where appropriate, such as tokenization.

## Build system

The project uses CMake for building C++ and ROCm-related components.

The build setup is designed around the system-installed ROCm environment used
during development.

## Current status

`shark-lab` is under active development.

The project is primarily intended for:

* learning LLM engineering
* experimenting with model architectures
* studying training systems
* exploring GPU acceleration

APIs and internal implementations may change as the project evolves.

# chatty

A lightweight native C++ LLM chat client built with Dear ImGui.

`chatty` is designed as a practical frontend for OpenAI-compatible LLM inference
services.

## Features

* OpenAI-compatible API streaming
* Local SQLite conversation storage
* Multiple personas/peers
* Lorebook system
* Native C++ frontend

## Screenshot

<img width="1741" height="990" alt="chatty-demo-0" src="https://github.com/user-attachments/assets/9091dba2-7c52-4bf9-bc03-7724c2e0c6a5" />
<img width="1739" height="992" alt="chatty-demo-1" src="https://github.com/user-attachments/assets/57b70183-ee83-481f-b4c0-721109e40cdb" />
<img width="1915" height="1115" alt="chatty-demo-2" src="https://github.com/user-attachments/assets/6a511d31-0a36-41a3-9688-f4e3362633dc" />
<img width="1911" height="1116" alt="chatty-demo-3" src="https://github.com/user-attachments/assets/dd73cd54-bd5c-4198-94da-1432de48abf9" />

(mradermacher/MN-Crimson-Cafe-12B-i1-GGUF, EldritchLabs/MN-Crimson-Cafe-12B)

## Build

Requirements:

* C++20 compiler
* CMake
* Boost
* SQLite3
* GLFW
* OpenGL
* shark-lab
