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

Examples
```bash
python inference.py --config ../shark-gen/options.toml --max-new-tokens 100 --temperature 0.8 --top-k 50

python trainer.py --config ../shark-gen/options.toml

python server.py --config ../shark-gen/options.toml
```


