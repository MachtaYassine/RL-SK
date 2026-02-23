"""Rollout buffer with GAE for PPO training.

Separate buffers for bid and play transitions since they have
different state dimensions and action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    legal_mask: torch.Tensor
    trick_history: Optional[torch.Tensor] = None


class RolloutBuffer:
    """Stores transitions and computes GAE advantages."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, state: torch.Tensor, action: int, log_prob: float,
            value: float, reward: float, done: bool, legal_mask: torch.Tensor,
            trick_history: Optional[torch.Tensor] = None) -> None:
        self.transitions.append(Transition(
            state=state, action=action, log_prob=log_prob,
            value=value, reward=reward, done=done, legal_mask=legal_mask,
            trick_history=trick_history,
        ))

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95,
                    last_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Returns:
            (advantages, returns) both shape [N]
        """
        n = len(self.transitions)
        if n == 0:
            return torch.zeros(0), torch.zeros(0)

        advantages = torch.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            tr = self.transitions[t]
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 0.0 if tr.done else 1.0
            else:
                next_value = self.transitions[t + 1].value
                next_non_terminal = 0.0 if tr.done else 1.0

            delta = tr.reward + gamma * next_value * next_non_terminal - tr.value
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        values = torch.tensor([t.value for t in self.transitions])
        returns = advantages + values
        return advantages, returns

    def get_batches(self, advantages: torch.Tensor, returns: torch.Tensor,
                    batch_size: int) -> list[dict]:
        """Split buffer into mini-batches for PPO updates."""
        n = len(self.transitions)
        indices = torch.randperm(n)
        batches = []
        has_trick_history = self.transitions[0].trick_history is not None

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                "states": torch.stack([self.transitions[i].state for i in idx]),
                "actions": torch.tensor([self.transitions[i].action for i in idx], dtype=torch.long),
                "old_log_probs": torch.tensor([self.transitions[i].log_prob for i in idx]),
                "advantages": advantages[idx],
                "returns": returns[idx],
                "legal_masks": torch.stack([self.transitions[i].legal_mask for i in idx]),
            }
            if has_trick_history:
                batch["trick_histories"] = torch.stack(
                    [self.transitions[i].trick_history for i in idx]
                )
            batches.append(batch)

        return batches

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
