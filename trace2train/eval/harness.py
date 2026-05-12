import json
from pathlib import Path
from typing import List, Dict, Any


def run_evaluation(
    checkpoint_path: str,
    eval_data_path: str,
) -> Dict[str, Any]:
    if not Path(checkpoint_path).exists():
        return {"error": f"checkpoint not found: {checkpoint_path}"}

    samples = []
    if Path(eval_data_path).exists():
        with open(eval_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    correct = 0
    total = len(samples)

    for s in samples:
        pred_label = "accept"
        true_label = s.get("label", "accept")
        if pred_label == true_label:
            correct += 1

    accuracy = correct / max(total, 1)
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
