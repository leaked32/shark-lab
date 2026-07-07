# shark-lab

`shark-lab` is my personal LLM learning project. It contains experiments for building and training GPT/LLaMA-style language models from scratch.

The project currently depends on an external tokenizer.

This repository is mainly for learning, experimentation, and engineering practice, so some parts may be incomplete or changing over time.

## Project structure

```text
shared/
  model.py      Model architecture
  util.py       Common utility functions
  format.py     File/data format helpers

trainer.py      Pre-training entry point
sft_trainer.py  Supervised fine-tuning entry point
inference.py    Inference/testing entry point

options.toml    Shared configuration file

scripts/
  collector_shot/  Conversation generation tool
  corpus/          Corpus generation utilities
  crawlers/        Local/private crawling tools, not intended for public release
```

## Notes

And... It's atually my personal learning project. So, I can make mistakes, it's legimate for a beginner, right?

A! Please forgive me if I make mistakes. And, it may look suspiciously similar to nanoGPT, I appreciate it.
