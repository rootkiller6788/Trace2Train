# Trace2Train — 技术栈

## 定位

Mini autograd + verifier-guided trace-to-train core：掌握 trace → train 数据与梯度闭环。

## 语言

| 层 | 语言 |
|---|---|
| 主语言 | Python |
| 训练后端 | PyTorch |
| 底层扩展 (optional) | C extension / Rust extension (pyo3/maturin) |
| CUDA custom op (optional) | CUDA C++ |

## 为什么不用纯 C 写训练框架

训练框架如果纯 C 写，会很快被这些东西拖死：

- autograd
- optimizer
- tokenizer
- checkpoint
- LoRA
- dataset
- evaluation
- model loading
- mixed precision

所以底层权力不放在"重新写 PyTorch"，而放在：

- trace schema
- failure labeling
- SFT / DPO / eval sample generation
- verifier-guided training loop

## 依赖

| 库 | 用途 |
|---|---|
| PyTorch | 训练后端、autograd、optimizer |
| Transformers (optional) | 模型加载 |
| TRL (optional) | RLHF 管线 |
| Accelerate (optional) | 分布式训练 |
| HuggingFace Datasets | 从 CSV / JSON / TXT / Parquet 加载 trace 数据 |
| Safetensors | 安全快速的 checkpoint / tensor 存储，支持 zero-copy |
| PyO3 / Maturin (optional) | Rust 扩展绑定 |

## PyTorch 自定义算子

PyTorch 官方支持自定义 C++/CUDA operator，推荐通过 operator registration API 注册，而不是只拿 data pointer 去调用内核。这样自定义算子可以和 autograd、torch.compile、vmap 等子系统组合。

## 数据格式

| 格式 | 用途 |
|---|---|
| JSONL | trace 输入 / SFT / DPO / eval 输出 |
| Parquet (optional) | 大规模训练数据 |
| Safetensors | checkpoint / tensor artifact |

## 构建

| 工具 | 说明 |
|---|---|
| pip / uv / poetry | Python 包管理 |
| pyproject.toml | 项目元数据 |
| setuptools-rust / maturin (optional) | Rust 扩展构建 |

## 测试

| 工具 | 说明 |
|---|---|
| pytest | Python 测试 |
| CTest (optional) | C 扩展测试 |

## 第一版闭环

```bash
trace2train extract traces/*.jsonl --out data/sft.jsonl
trace2train build-dpo traces/*.jsonl --out data/dpo.jsonl
trace2train report data/sft.jsonl --out reports/dataset.md
trace2train train --config examples/tiny_sft.yaml
trace2train eval --config examples/eval.yaml
```

## 对外协议

| 输入 | 输出 |
|---|---|
| `trace.jsonl` (from ApeinxRT-Core) | `sft.jsonl` |
| execution trace / repair trajectory | `dpo.jsonl` |
| verifier labels | `eval.jsonl` |
| | `train_report.md` |
| | `checkpoint.bin` |

## 公共底座复用

| 层 | 选项 |
|---|---|
| OS | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA CUDA |
| Python | 3.10 / 3.11 |
| 报告格式 | JSON / JSONL / Markdown |

## 关键词

`trace-to-dataset` `verifier label` `failure taxonomy` `SFT` `DPO` `eval` `training report`
