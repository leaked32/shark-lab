# shark-lab

`shark-lab` is my personal LLM learning project.

`shark-lab` is an experimental LLM framework for learning and implementing
GPT/LLaMA-style language models from scratch.

The project contains model implementations, training pipelines, inference tools,
Supervised Fine-Tuning (SFT) workflows, checkpoint conversion utilities, and
ROCm/HIP experiments for GPU programming and performance exploration.

This repository documents my learning and engineering process while building
LLM systems from the ground up. The implementation is experimental and may
evolve as new features, optimizations, and experiments are added.

## Features

- GPT/LLaMA-style Transformer model implementation
- Pre-training and Supervised Fine-Tuning (SFT)
- Custom training pipeline and checkpoint handling
- Inference and testing tools
- Hugging Face model conversion support
- ROCm/HIP experiments and GPU kernel development

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
````

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

The current experiments focus on understanding:

* GPU execution models
* Memory behavior
* Kernel implementation
* Numerical stability
* Performance optimization

Run ROCm experiments:

```bash
cd rocm
scons

cmake --preset linux-x64-debug
cmake --build build/linux-x64-debug
./build/linux-x64-debug/hip-test/hip-test
./build/linux-x64-debug/chatty/chatty
```

For ROCm training runs, debugging asynchronous GPU failures may require:

```bash
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
python train.py --config options.safe.big.toml
```

## Design notes

The project intentionally avoids depending on large external training
frameworks for core learning purposes. Many components are implemented
directly to understand the underlying mechanisms of LLM training systems.

External components are still used where appropriate, such as tokenization.

## Build system

The ROCm experiments currently use SCons.

This choice is based on compatibility with the system-installed ROCm
environment used during development.

## Current status

`shark-lab` is under active development.

The project is primarily intended for:

* learning LLM engineering
* experimenting with model architectures
* studying training systems
* exploring GPU acceleration

APIs and internal implementations may change as the project evolves.

black --line-length 100 --skip-magic-trailing-comma .
