"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_size: int = 0  # 0 = auto-scale based on GPU memory
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_dim: int = 256
    card_embed_dim: int = 16
    player_embed_dim: int = 4


@dataclass
class GameConfig:
    num_players: int = 4  # Fixed count, ignored if vary_players=True
    min_players: int = 2
    max_players: int = 8
    vary_players: bool = False


@dataclass
class TrainConfig:
    total_games: int = 100_000
    update_interval: int = 10  # PPO update every N games; 0 = auto-scale with batch
    num_workers: int = 1  # Parallel game simulations
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 1000
    pool_size: int = 20
    snapshot_interval: int = 50
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    exploration_burst_interval: int = 0  # 0 = disabled
    exploration_burst_duration: int = 0
