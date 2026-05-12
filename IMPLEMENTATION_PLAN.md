# Trace2Train — 落地计划

## 定位

> Mini autograd + trace-to-train 数据闭环。不重复造 PyTorch，而是做行为日志 → 训练样本 → 梯度更新 → 评估回放的闭环。

## 底层权力

**训练数据与梯度闭环权** — 谁掌握 agent trace → SFT/DPO/eval 数据 → training loop 的转换，谁就控制了从行为到模型的完整链条。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python |
| 训练后端 | PyTorch |
| 数据 | JSONL / HuggingFace Datasets |
| Checkpoint | Safetensors |
| 包管理 | pyproject.toml |

## 第一版目标

实现两条核心链路：
1. **Trace → Dataset**: execution trace 转 SFT/DPO/eval 样本
2. **Train**: 用 PyTorch 训练最小模型（MLP / tiny transformer），输出 checkpoint + 报告

## 核心链路

```
trace.jsonl (from ApeinxRT-Core)
  ↓
extract         → 解析 trace，提取 step/token/decision 序列
  ↓
label           → verifier 标注 success/failure/ambiguous
  ↓
sft_build       → 生成 (prompt, completion) 对
  ↓
dpo_build       → 生成 (prompt, chosen, rejected) 对
  ↓
eval_build      → 生成 eval 用例
  ↓
train           → PyTorch training loop
  ↓
eval            → 评估 + 报告
```

## 模块说明

```
trace2train-core/
├── trace2train/
│   ├── trace/
│   │   ├── schema.py       # TraceEvent 类型定义
│   │   ├── loader.py       # trace.jsonl 读取
│   │   ├── normalizer.py   # 归一化/去噪
│   │   └── validator.py    # 格式校验
│   ├── dataset/
│   │   ├── sft_builder.py  # trace → SFT 样本
│   │   ├── dpo_builder.py  # trace → DPO 对
│   │   ├── eval_builder.py # trace → eval 用例
│   │   └── report.py       # 数据集统计报告
│   ├── train/
│   │   ├── tiny_trainer.py # 最小训练循环
│   │   ├── lora_trainer.py # LoRA 微调 (optional)
│   │   └── callbacks.py    # 训练回调
│   ├── eval/
│   │   ├── harness.py      # 评估框架
│   │   └── metrics.py      # 指标计算
│   └── cli.py              # 命令行入口
├── examples/traces/        # 示例 trace 文件
├── examples/configs/       # 训练配置 yaml
├── tests/
└── pyproject.toml
```

## 里程碑

### M1: Trace 协议 + Loader (Week 1)
- [ ] TraceEvent schema 定义
- [ ] trace.jsonl loader + validator
- [ ] normalizer: 去重/截断/对齐
- [ ] CLI: `trace2train extract traces/*.jsonl --out data/sft.jsonl`

### M2: Dataset Builder (Week 2)
- [ ] SFT builder: trace step → (prompt, completion)
- [ ] DPO builder: success vs failure → (chosen, rejected)
- [ ] Eval builder: 从 trace 中提取验证用例
- [ ] 数据集统计报告

### M3: Training Loop (Week 3)
- [ ] Tiny trainer: MLP / char-level LM on SFT data
- [ ] Checkpoint save/load (Safetensors)
- [ ] Training 指标: loss curve, token accuracy
- [ ] LoRA trainer stub

### M4: Evaluation + Report (Week 4)
- [ ] Eval harness: 加载 checkpoint 跑 eval 用例
- [ ] Metrics: accuracy, perplexity, pass@1
- [ ] End-to-end: trace → dataset → train → eval → report

## CLI 命令

```bash
# Trace → Dataset
trace2train extract traces/*.jsonl --out data/sft.jsonl
trace2train build-dpo traces/*.jsonl --out data/dpo.jsonl
trace2train build-eval traces/*.jsonl --out data/eval.jsonl
trace2train report data/sft.jsonl --out reports/dataset.md

# Train
trace2train train --config examples/tiny_sft.yaml

# Eval
trace2train eval --config examples/eval.yaml
```

## 配置文件示例 (tiny_sft.yaml)

```yaml
data:
  sft: data/sft.jsonl
  eval: data/eval.jsonl

model:
  type: mlp
  hidden_dim: 256
  num_layers: 2

train:
  batch_size: 32
  epochs: 10
  learning_rate: 1e-4
  optimizer: adam

output:
  checkpoint_dir: checkpoints/
  report_path: reports/train_report.md
```

## 不做什么

- 不做完整 autograd 引擎（用 PyTorch 的）
- 不做分布式训练
- 不做 RLHF / PPO
- 不做 model serving
- 不做 tokenizer 训练

## 验收标准

- trace.jsonl → sft.jsonl / dpo.jsonl / eval.jsonl 格式合法
- 数据集报告包含: 样本数、avg_length、label 分布
- MLP 训练后 loss 下降明显
- 端到端闭环可复现
