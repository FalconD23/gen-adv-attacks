from abc import ABC, abstractmethod
import torch
from src.attacks.base_attacks import BaseIterativeAttack


class BaseIterStrategy(ABC):
    def __init__(self, attack: BaseIterativeAttack):
        self.attack = attack

    @abstractmethod
    def run(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def next_epoch(self):
        pass


class TruncCurrLinClip(BaseIterStrategy):
    def __init__(
            self,
            attack: BaseIterativeAttack,
            k: int = 1,
            n_steps_start: int = 1,
    ):
        super().__init__(attack)
        self.k = k
        self.t = n_steps_start

    def next_epoch(self):
        self.t = min(
            self.t + 1,
            self.attack.n_steps
        )

    def run(self, X: torch.Tensor) -> torch.Tensor:
        X_adv = X.detach().clone()
        for i in range(self.t):
            if i < self.t - self.k:
                with torch.no_grad():
                    X_adv = self.attack.step(X_adv, None, mode="train")
            else:
                X_adv = self.attack.step(X_adv, None, mode="train")
        return X_adv


PRESETS = {
    "FullGrad": lambda attack: TruncCurrLinClip(
        attack=attack,
        k=attack.n_steps,
        n_steps_start=attack.n_steps
    ),
    "LastK": lambda attack, k: TruncCurrLinClip(
        attack=attack,
        k=k,
        n_steps_start=attack.n_steps
    )
}
