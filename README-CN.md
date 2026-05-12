# Trace2Train (训练微内核)

> Mini autograd + verifier 引导的 trace-to-train 核心。

Trace2Train 不重复造 PyTorch。它掌握的是**数据闭环**：把执行 trace（来自 ApeinxRT-Core）转化为 SFT/DPO/eval 数据集，然后用 PyTorch 跑最简训练循环。

## 底层权力

**训练数据与梯度闭环权** — 谁掌握 agent trace → SFT/DPO/eval → training loop，谁就控制了从行为到模型的完整链条。

## 管线

```
trace.jsonl（来自 ApeinxRT-Core）
  ↓ extract      → 解析 trace，提取 step/token/decision 序列
  ↓ label        → verifier 标注：success / failure / ambiguous
  ↓ sft_build    → 生成 (prompt, completion) 对
  ↓ dpo_build    → 生成 (prompt, chosen, rejected) 对
  ↓ eval_build   → 生成评估用例
  ↓ train        → PyTorch 训练循环
  ↓ eval         → 指标 + 报告
```

## 快速开始

```bash
trace2train extract traces/*.jsonl --out data/sft.jsonl
trace2train build-dpo traces/*.jsonl --out data/dpo.jsonl
trace2train build-eval traces/*.jsonl --out data/eval.jsonl
trace2train report --sft data/sft.jsonl --out reports/dataset.md
trace2train train --config examples/configs/tiny_sft.yaml
trace2train eval --config examples/configs/eval.yaml
```

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python |
| 训练后端 | PyTorch |
| 数据 | JSONL / HuggingFace Datasets |
| Checkpoint | Safetensors |
| 包管理 | pyproject.toml |

## 不做什么

- 不做自定义 autograd 引擎（用 PyTorch 的）
- 不做分布式训练
- 不做 RLHF / PPO
- 不做模型 serving
- 不做 tokenizer 训练

## 项目结构

```
trace2train-core/
├── trace2train/
│   ├── trace/       # Schema、loader、normalizer、validator
│   ├── dataset/     # SFT/DPO/eval 构建器 + 报告
│   ├── train/       # 最小训练器、LoRA 训练器、回调
│   ├── eval/        # 评估框架 + 指标
│   └── cli.py       # CLI 入口
├── examples/
│   ├── traces/      # 示例 trace 文件
│   └── configs/     # YAML 配置文件
└── pyproject.toml
```

## 与其他项目的关系

```
KernelLab → (C ABI) → ApeinxRT-Core → (trace.jsonl) → Trace2Train
                          ↑
                    Apeinx-IR → (plan.json)
```

## License

待定
