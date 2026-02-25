"""Hyperparameter optimization with Optuna.

Runs short training trials, each with different hyperparameters.
Optuna's Bayesian sampler learns which regions of HP space are promising.

Usage:
    python hpo.py --trials 50 --games-per-trial 10000
    # Or via SLURM: sbatch run_hpo.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil

import optuna
import torch

from config import PPOConfig, GameConfig, TrainConfig
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, args) -> float:
    """Single HPO trial: train with sampled HPs, return ELO."""

    # --- Sample hyperparameters ---
    policy_lr = trial.suggest_float("policy_lr", 3e-5, 5e-4, log=True)
    value_lr = trial.suggest_float("value_lr", 1e-4, 2e-3, log=True)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.005, 0.05, log=True)
    policy_epochs = trial.suggest_int("policy_epochs", 2, 6)
    value_epochs = trial.suggest_int("value_epochs", 4, 12)
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Ensure value_epochs >= policy_epochs
    if value_epochs < policy_epochs:
        value_epochs = policy_epochs

    # --- Build configs ---
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    ppo_cfg = PPOConfig(
        policy_lr=policy_lr,
        value_lr=value_lr,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        policy_epochs=policy_epochs,
        value_epochs=value_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
    )
    game_cfg = GameConfig(
        vary_players=True,
        min_players=2,
        max_players=8,
    )
    train_cfg = TrainConfig(
        total_games=args.games_per_trial,
        num_workers=args.workers,
        log_dir=os.path.join(trial_dir, "runs"),
        save_dir=os.path.join(trial_dir, "checkpoints"),
        save_interval=args.games_per_trial + 1,  # only save at end
        eval_interval=args.games_per_trial // 5,
    )

    # --- Train ---
    trainer = Trainer(ppo_cfg, game_cfg, train_cfg, device=args.device)
    trainer.train()

    # --- Report intermediate ELO for pruning ---
    elo = trainer.player_elo
    win_rate = sum(trainer.recent_wins) / max(len(trainer.recent_wins), 1)

    logger.info(f"Trial {trial.number}: ELO={elo:.0f}, WR={win_rate:.0%}")
    logger.info(f"  params: {trial.params}")

    # Clean up checkpoints to save disk space
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    return elo


def main():
    parser = argparse.ArgumentParser(description="HPO for Skull King RL")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--games-per-trial", type=int, default=10000,
                        help="Training games per trial (shorter = faster but noisier)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="hpo_results")
    parser.add_argument("--study-name", type=str, default="skullking-hpo")
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///hpo.db). "
                             "Enables resuming and parallel workers.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    # Storage: file-based SQLite for persistence + parallel support
    storage = args.db or f"sqlite:///{args.output_dir}/hpo.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # maximize ELO
        load_if_exists=True,   # resume if DB exists
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    logger.info(f"Starting HPO: {args.trials} trials, {args.games_per_trial} games each")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    # --- Report results ---
    print("\n" + "=" * 60)
    print("HPO RESULTS")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best ELO: {study.best_value:.0f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save best params
    import json
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved to {args.output_dir}/best_params.json")


if __name__ == "__main__":
    main()
