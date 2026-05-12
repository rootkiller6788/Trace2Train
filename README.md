# Trace2Train

> Mini autograd + verifier-guided trace-to-train core.

Trace2Train does not reinvent PyTorch. It owns the **data loop**: turning execution traces (from ApeinxRT-Core) into SFT/DPO/eval datasets, then running a minimal training loop with PyTorch.

## Power

**Training data and gradient closed-loop authority** — whoever controls agent trace → SFT/DPO/eval → training loop owns the full chain from behavior to model.

## Pipeline

```
trace.jsonl (from ApeinxRT-Core)
  ↓ extract      → parse traces, extract step/token/decision sequences
  ↓ label        → verifier: success / failure / ambiguous
  ↓ sft_build    → (prompt, completion) pairs
  ↓ dpo_build    → (prompt, chosen, rejected) pairs
  ↓ eval_build   → evaluation cases
  ↓ train        → PyTorch training loop
  ↓ eval         → metrics + report
```

## Quick Start

```bash
trace2train extract traces/*.jsonl --out data/sft.jsonl
trace2train build-dpo traces/*.jsonl --out data/dpo.jsonl
trace2train build-eval traces/*.jsonl --out data/eval.jsonl
trace2train report --sft data/sft.jsonl --out reports/dataset.md
trace2train train --config examples/configs/tiny_sft.yaml
trace2train eval --config examples/configs/eval.yaml
```

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| Training backend | PyTorch |
| Data | JSONL / HuggingFace Datasets |
| Checkpoint | Safetensors |
| Package | pyproject.toml |

## What We Don't Build

- No custom autograd engine (use PyTorch's)
- No distributed training
- No RLHF / PPO
- No model serving
- No tokenizer training

## Project Structure

```
trace2train-core/
├── trace2train/
│   ├── trace/       # Schema, loader, normalizer, validator
│   ├── dataset/     # SFT/DPO/eval builders + report
│   ├── train/       # Tiny trainer, LoRA trainer, callbacks
│   ├── eval/        # Harness + metrics
│   └── cli.py       # CLI entry point
├── examples/
│   ├── traces/      # Sample trace files
│   └── configs/     # YAML configs
└── pyproject.toml
```

## Relationship to Other Apeinx Projects

```
KernelLab → (C ABI) → ApeinxRT-Core → (trace.jsonl) → Trace2Train
                          ↑
                    Apeinx-IR → (plan.json)
```

## License

TBD
