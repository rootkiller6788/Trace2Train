from typing import Dict, Any


class LoRATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def train(self):
        print("LoRA trainer: stub — not implemented in v0.1")
