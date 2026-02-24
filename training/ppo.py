"""PPO with clipped objective, separate bid/play optimizers, and rich diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from training.rollout_buffer import RolloutBuffer


@dataclass
class PPOStats:
    # Core PPO metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    num_updates: int = 0
    # Critic health
    explained_variance: float = 0.0
    value_mae: float = 0.0
    # Policy stability
    kl_divergence: float = 0.0
    approx_kl_max: float = 0.0
    # Gradient health
    grad_norm: float = 0.0
    layer_grad_norms: Optional[Dict[str, float]] = None
    # Advantage signal quality
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_max: float = 0.0
    # Representation health
    dead_neuron_frac: float = 0.0
    effective_rank: float = 0.0
    activation_mean: float = 0.0
    activation_std: float = 0.0
    # Parameter health
    weight_norm: float = 0.0
    update_ratio: float = 0.0
    # Policy behavior
    action_diversity: float = 0.0


def _move_dict_to_device(d: dict, device: torch.device) -> dict:
    """Move all tensors in a dict to a device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


class PPO:
    """PPO trainer with separate bid and play networks."""

    def __init__(
        self,
        bid_net: BidActorCritic,
        play_net: PlayActorCritic,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
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
            PPOStats with comprehensive training diagnostics.
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

        # Accumulators for diagnostics (sampled periodically)
        kl_accum = 0.0
        kl_max_accum = 0.0
        grad_norm_accum = 0.0
        dead_neuron_accum = 0.0
        effective_rank_accum = 0.0
        act_mean_accum = 0.0
        act_std_accum = 0.0
        action_diversity_accum = 0.0
        value_mae_accum = 0.0
        diag_count = 0
        last_layer_grad_norms = {}

        for _ in range(self.epochs):
            for batch in buffer.get_batches(advantages, returns, self.batch_size):
                states = _move_dict_to_device(batch["states"], device)
                actions = batch["actions"].to(device)
                old_log_probs = batch["old_log_probs"].to(device)
                adv = batch["advantages"].to(device)
                ret = batch["returns"].to(device)
                masks = batch["legal_masks"].to(device)

                trick_histories = batch.get("trick_histories")
                if trick_histories is not None:
                    trick_histories = _move_dict_to_device(trick_histories, device)
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
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()

                # Track core stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                stats.policy_loss += policy_loss.item()
                stats.value_loss += value_loss.item()
                stats.entropy += entropy.item()
                stats.clip_fraction += clip_fraction
                total_updates += 1

                # Diagnostics every 4th batch to minimize overhead
                if total_updates % 4 == 0:
                    with torch.no_grad():
                        # KL divergence
                        kl = (old_log_probs - action_log_probs).mean().item()
                        kl_max = (old_log_probs - action_log_probs).max().item()
                        kl_accum += kl
                        kl_max_accum += kl_max

                        # Grad norm
                        grad_norm_accum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                        # Value MAE
                        value_mae_accum += (values - ret).abs().mean().item()

                        # Action diversity
                        num_possible = masks[0].sum().item()  # approximate
                        unique = actions.unique().numel()
                        action_diversity_accum += unique / max(num_possible, 1)

                        # First hidden layer activation stats
                        # Re-forward through input_proj only
                        if hasattr(net, 'input_proj'):
                            # Build the input the same way the network does
                            if network == "bid":
                                hand_emb = net.card_emb.embed_set(states["hand_ids"], states["hand_mask"])
                                seen_emb = net.card_emb.embed_set(states["seen_ids"], states["seen_mask"])
                                inp = torch.cat([hand_emb, seen_emb, states["scalars"]], dim=-1)
                            else:
                                hand_emb = net.card_emb.embed_set(states["hand_ids"], states["hand_mask"])
                                seen_emb = net.card_emb.embed_set(states["seen_ids"], states["seen_mask"])
                                trick_emb = net.card_emb.embed_set(states["trick_card_ids"], states["trick_card_mask"])
                                trick_enc = net._encode_tricks(trick_histories, device, states["scalars"].shape[0])
                                inp = torch.cat([hand_emb, seen_emb, trick_emb, states["scalars"], trick_enc], dim=-1)

                            h = net.input_proj(inp)

                            # Dead neurons: never activate across batch
                            dead_frac = (h.abs().max(dim=0).values < 1e-6).float().mean().item()
                            dead_neuron_accum += dead_frac

                            act_mean_accum += h.mean().item()
                            act_std_accum += h.std().item()

                            # Effective rank via SVD (subsample for speed)
                            h_sub = h[:min(256, h.shape[0])]
                            try:
                                S = torch.linalg.svdvals(h_sub)
                                S_norm = S / (S.sum() + 1e-8)
                                log_s = S_norm.log().clamp(min=-30)
                                eff_rank = torch.exp(-(S_norm * log_s).sum()).item()
                                effective_rank_accum += eff_rank
                            except Exception:
                                effective_rank_accum += 0.0

                        diag_count += 1

                        # Per-layer grad norms (only last diagnostic batch)
                        last_layer_grad_norms = {}
                        for name, p in net.named_parameters():
                            if p.grad is not None:
                                last_layer_grad_norms[name] = p.grad.norm().item()

        if total_updates > 0:
            stats.policy_loss /= total_updates
            stats.value_loss /= total_updates
            stats.entropy /= total_updates
            stats.clip_fraction /= total_updates

        if diag_count > 0:
            stats.kl_divergence = kl_accum / diag_count
            stats.approx_kl_max = kl_max_accum / diag_count
            stats.grad_norm = grad_norm_accum / diag_count
            stats.value_mae = value_mae_accum / diag_count
            stats.dead_neuron_frac = dead_neuron_accum / diag_count
            stats.effective_rank = effective_rank_accum / diag_count
            stats.activation_mean = act_mean_accum / diag_count
            stats.activation_std = act_std_accum / diag_count
            stats.action_diversity = action_diversity_accum / diag_count

        stats.layer_grad_norms = last_layer_grad_norms
        stats.num_updates = total_updates

        # Buffer-level stats (computed once)
        with torch.no_grad():
            all_values = torch.tensor([t.value for t in buffer.transitions])
            var_returns = returns.var()
            stats.explained_variance = (1 - (returns - all_values).var() / (var_returns + 1e-8)).item()

            stats.advantage_mean = advantages.mean().item()
            stats.advantage_std = advantages.std().item()
            stats.advantage_max = advantages.abs().max().item()

            # Weight norm
            total_wnorm = sum(p.data.norm().item() ** 2 for p in net.parameters()) ** 0.5
            stats.weight_norm = total_wnorm

            # Update ratio: how much parameters move per step relative to their size
            lr = optimizer.defaults['lr']
            stats.update_ratio = stats.grad_norm * lr / (total_wnorm + 1e-8)

        return stats
