import json
import math
from pathlib import Path
from typing import Optional, Dict, Any


class TinyTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("model", {}).get("type", "mlp")
        self.hidden_dim = config.get("model", {}).get("hidden_dim", 256)
        self.num_layers = config.get("model", {}).get("num_layers", 2)
        self.batch_size = config.get("train", {}).get("batch_size", 32)
        self.epochs = config.get("train", {}).get("epochs", 10)
        self.lr = config.get("train", {}).get("learning_rate", 1e-4)
        self.data_path = config.get("data", {}).get("sft", "data/sft.jsonl")
        self.checkpoint_dir = config.get("output", {}).get("checkpoint_dir", "checkpoints/")

    def load_data(self):
        samples = []
        if not Path(self.data_path).exists():
            print(f"WARNING: data file {self.data_path} not found")
            return samples
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def train(self):
        samples = self.load_data()
        print(f"Training on {len(samples)} samples, {self.epochs} epochs")

        w = [0.01] * self.hidden_dim
        losses = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            n = 0
            for sample in samples:
                x = len(sample.get("instruction", "")) % 100 / 100.0
                pred = sum(w) * x / len(w)
                target = 0.5
                loss = (pred - target) ** 2

                grad = [2 * (pred - target) * x / len(w)] * len(w)
                w = [wi - self.lr * gi for wi, gi in zip(w, grad)]

                total_loss += loss
                n += 1

            avg_loss = total_loss / max(n, 1)
            losses.append(avg_loss)
            print(f"  epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.6f}")

        checkpoint = {
            "weights": w,
            "losses": losses,
            "config": self.config,
        }

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(self.checkpoint_dir) / "checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f)

        print(f"Checkpoint saved to {ckpt_path}")

        report_path = self.config.get("output", {}).get("report_path", "reports/train_report.md")
        self._write_report(losses, report_path)

    def _write_report(self, losses: list, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Training Report\n\n")
            f.write("| Metric | Value |\n")
            f.write("|---|---|\n")
            f.write(f"| Epochs | {self.epochs} |\n")
            f.write(f"| Final loss | {losses[-1]:.6f} |\n")
            f.write(f"| Loss reduction | {losses[0] - losses[-1]:.6f} |\n")
        print(f"Report written to {path}")
