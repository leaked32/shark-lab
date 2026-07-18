# shark-lab

`shark-lab` is my personal LLM learning project. It contains experiments for building and training GPT/LLaMA-style language models from scratch.

This repository documents my learning process while building GPT/LLaMA-style language models from scratch. The implementation is experimental and may change as I improve it.


The project currently depends on an external tokenizer.

This repository is mainly for learning, experimentation, and engineering practice, so some parts may be incomplete or changing over time.

## Project structure

```text
shared/
  model.py      Model architecture
  util.py       Common utility functions
  format.py     File/data format helpers

trainer.py      Pre-training, Supervised fine-tuning
inference.py    Inference/testing entry point
convert_hf.py   Convert compatible models to the checkpoint.

config/
  360m.toml    Compatible configuration file for SmolLM2-360M-Instruct

scripts/
  collector.py  Conversation generation tool
  corpus/          Corpus generation utilities
  crawlers/        Local/private crawling tools, not intended for public release
```

## Notes

Examples
```bash
python inference.py \
  --config options.toml \
  --max-new-tokens 100 --temperature 0.8 --top-k 50

python convert_hf.py \
  --config options.toml \
  --source HuggingFaceTB/SmolLM2-360M-Instruct \
  --output checkpoints/smollm2-instruct.pt

python trainer.py --config options.toml

python server.py --config options.toml
```


```bash
cmake --preset linux-x64-debug
cmake --build build/linux-x64-debug
./build/linux-x64-debug/example-app
```
