"""Play actor-critic network for PPO with residual backbone and LSTM trick history."""

from __future__ import annotations

import torch
import torch.nn as nn
from networks.features import PLAY_DIM, MAX_HAND_SIZE, TRICK_FEAT_DIM, MAX_TRICKS


LSTM_HIDDEN = 128


class ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> ReLU -> Linear -> skip -> ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class PlayActorCritic(nn.Module):
    """Actor-critic for playing phase.

    Input: PLAY_DIM features + LSTM-encoded trick history
    Actor output: MAX_HAND_SIZE logits (play index 0-9)
    Critic output: scalar value
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.trick_lstm = nn.LSTM(
            input_size=TRICK_FEAT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.input_proj = nn.Sequential(
            nn.Linear(PLAY_DIM + LSTM_HIDDEN, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(4)])
        self.actor_head = nn.Linear(hidden_dim, MAX_HAND_SIZE)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _encode_tricks(self, trick_history: torch.Tensor | None) -> torch.Tensor:
        """Encode trick history through LSTM.

        Args:
            trick_history: [batch, MAX_TRICKS, TRICK_FEAT_DIM] or None

        Returns:
            [batch, LSTM_HIDDEN]
        """
        if trick_history is None:
            # Return zeros; batch size inferred later
            return None
        _, (h_n, _) = self.trick_lstm(trick_history)
        return h_n.squeeze(0)  # [batch, LSTM_HIDDEN]

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor,
                trick_history: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch, PLAY_DIM]
            legal_mask: [batch, MAX_HAND_SIZE] binary mask
            trick_history: [batch, MAX_TRICKS, TRICK_FEAT_DIM] or None

        Returns:
            (log_probs [batch, MAX_HAND_SIZE], value [batch, 1])
        """
        trick_enc = self._encode_tricks(trick_history)
        if trick_enc is None:
            trick_enc = torch.zeros(x.shape[0], LSTM_HIDDEN, device=x.device)
        h = self.input_proj(torch.cat([x, trick_enc], dim=-1))
        h = self.res_blocks(h)
        logits = self.actor_head(h)
        logits = logits + (legal_mask.log().clamp(min=-1e8))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.critic_head(h)
        return log_probs, value

    def get_action_and_value(self, x: torch.Tensor, legal_mask: torch.Tensor,
                             trick_history: torch.Tensor | None = None
                             ) -> tuple[int, float, float, float]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            x_b = x.unsqueeze(0).to(device)
            mask_b = legal_mask.unsqueeze(0).to(device)
            th_b = None
            if trick_history is not None:
                th_b = trick_history.unsqueeze(0).to(device)
            log_probs, value = self.forward(x_b, mask_b, th_b)
            log_probs = log_probs.squeeze(0)
            value = value.squeeze(0).item()
            probs = log_probs.exp()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action.item(), log_probs[action].item(), value, dist.entropy().item()
