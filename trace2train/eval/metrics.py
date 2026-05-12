from typing import Dict, Any


def compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    return {
        "accuracy": results.get("accuracy", 0.0),
        "perplexity": 0.0,
        "pass@1": results.get("accuracy", 0.0),
    }


def format_metrics_table(metrics: Dict[str, float]) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Perplexity | {metrics['perplexity']:.2f} |",
        f"| Pass@1 | {metrics['pass@1']:.4f} |",
    ]
    return "\n".join(lines)
