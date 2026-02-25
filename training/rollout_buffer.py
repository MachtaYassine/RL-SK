"""Rollout buffer with GAE for PPO training.

Separate buffers for bid and play transitions since they have
different state dimensions and action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch


@dataclass
class Transition:
    state: Dict[str, torch.Tensor]  # v2 encoded state dict
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    legal_mask: torch.Tensor
    trick_history: Optional[Dict[str, torch.Tensor]] = None
    belief_target: Optional[torch.Tensor] = None


class RolloutBuffer:
    """Stores transitions and computes GAE advantages."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, state: Dict[str, torch.Tensor], action: int, log_prob: float,
            value: float, reward: float, done: bool, legal_mask: torch.Tensor,
            trick_history: Optional[Dict[str, torch.Tensor]] = None,
            belief_target: Optional[torch.Tensor] = None) -> None:
        self.transitions.append(Transition(
            state=state, action=action, log_prob=log_prob,
            value=value, reward=reward, done=done, legal_mask=legal_mask,
            trick_history=trick_history, belief_target=belief_target,
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

        # Determine state dict keys from first transition
        state_keys = list(self.transitions[0].state.keys())

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            # Stack each state dict key separately
            states = {}
            for k in state_keys:
                states[k] = torch.stack([self.transitions[i].state[k] for i in idx])

            batch = {
                "states": states,
                "actions": torch.tensor([self.transitions[i].action for i in idx], dtype=torch.long),
                "old_log_probs": torch.tensor([self.transitions[i].log_prob for i in idx]),
                "advantages": advantages[idx],
                "returns": returns[idx],
                "legal_masks": torch.stack([self.transitions[i].legal_mask for i in idx]),
            }

            if has_trick_history:
                th_keys = list(self.transitions[0].trick_history.keys())
                trick_histories = {}
                for k in th_keys:
                    trick_histories[k] = torch.stack(
                        [self.transitions[i].trick_history[k] for i in idx])
                batch["trick_histories"] = trick_histories

            if self.transitions[0].belief_target is not None:
                batch["belief_targets"] = torch.stack(
                    [self.transitions[i].belief_target for i in idx])

            batches.append(batch)

        return batches

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
