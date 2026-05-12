import json
from pathlib import Path
from typing import List


def generate_report(
    sft_path: str,
    dpo_path: str,
    eval_path: str,
    output_path: str,
) -> None:
    def count(path: str) -> int:
        if not Path(path).exists():
            return 0
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())

    sft_n = count(sft_path)
    dpo_n = count(dpo_path)
    eval_n = count(eval_path)

    lines = [
        "# Dataset Report",
        "",
        "| Split | Samples |",
        "|---|---|",
        f"| SFT | {sft_n} |",
        f"| DPO | {dpo_n} |",
        f"| Eval | {eval_n} |",
        "",
        f"**Total**: {sft_n + dpo_n + eval_n} samples",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to {output_path}")
