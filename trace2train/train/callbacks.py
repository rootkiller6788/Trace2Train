from typing import Dict, Any


class Callback:
    def on_epoch_start(self, epoch: int): pass
    def on_epoch_end(self, epoch: int, loss: float): pass
    def on_train_end(self, final_loss: float): pass


class LossLogger(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch: int, loss: float):
        self.losses.append(loss)
        print(f"  [callback] epoch {epoch + 1}: loss = {loss:.6f}")
