"""PPO with clipped objective, separate bid/play optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from training.rollout_buffer import RolloutBuffer


@dataclass
class PPOStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    num_updates: int = 0


class PPO:
    """PPO trainer with separate bid and play networks."""

    def __init__(
        self,
        bid_net: BidActorCritic,
        play_net: PlayActorCritic,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.bid_net = bid_net
        self.play_net = play_net
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.bid_optimizer = torch.optim.Adam(bid_net.parameters(), lr=lr)
        self.play_optimizer = torch.optim.Adam(play_net.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer, network: str) -> PPOStats:
        """Run PPO update on a buffer.

        Args:
            buffer: Rollout buffer with transitions.
            network: "bid" or "play" to select which network to update.

        Returns:
            PPOStats with training metrics.
        """
        if len(buffer) == 0:
            return PPOStats()

        net = self.bid_net if network == "bid" else self.play_net
        optimizer = self.bid_optimizer if network == "bid" else self.play_optimizer

        advantages, returns = buffer.compute_gae(self.gamma, self.gae_lambda)
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = PPOStats()
        total_updates = 0

        device = next(net.parameters()).device

        for _ in range(self.epochs):
            for batch in buffer.get_batches(advantages, returns, self.batch_size):
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                old_log_probs = batch["old_log_probs"].to(device)
                adv = batch["advantages"].to(device)
                ret = batch["returns"].to(device)
                masks = batch["legal_masks"].to(device)

                trick_histories = batch.get("trick_histories")
                if trick_histories is not None:
                    trick_histories = trick_histories.to(device)
                    log_probs, values = net(states, masks, trick_histories)
                else:
                    log_probs, values = net(states, masks)
                values = values.squeeze(-1)

                # Gather log probs for taken actions
                action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Policy loss (clipped)
                ratio = (action_log_probs - old_log_probs).exp()
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, ret)

                # Entropy bonus
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()

                # Track stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                stats.policy_loss += policy_loss.item()
                stats.value_loss += value_loss.item()
                stats.entropy += entropy.item()
                stats.clip_fraction += clip_fraction
                total_updates += 1

        if total_updates > 0:
            stats.policy_loss /= total_updates
            stats.value_loss /= total_updates
            stats.entropy /= total_updates
            stats.clip_fraction /= total_updates
        stats.num_updates = total_updates

        return stats
