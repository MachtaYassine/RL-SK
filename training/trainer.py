"""Main training loop orchestration with parallel game simulation."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp

from agents.heuristic_agent import HeuristicAgent
from config import PPOConfig, GameConfig, TrainConfig
from monitoring.metrics import MetricsLogger
from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from networks.features import (
    encode_bid_state_v2, encode_play_state_v2,
    get_legal_bid_mask, get_legal_play_mask,
    encode_trick_history_v2,
    PLAY_SCALAR_DIM,
)
from skull_king.cards import SpecialType
from skull_king.game import SkullKingGame, Phase
from training.ppo import PPO, PPOStats
from training.rollout_buffer import RolloutBuffer
from training.self_play import OpponentPool
from training.worker import play_games

logger = logging.getLogger(__name__)


def _worker_init():
    """Disable grad in worker processes."""
    torch.set_grad_enabled(False)


class Trainer:
    """Orchestrates PPO self-play training with parallel game workers."""

    def __init__(
        self,
        ppo_config: PPOConfig,
        game_config: GameConfig,
        train_config: TrainConfig,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.game_config = game_config
        self.train_config = train_config
        self.ppo_config = ppo_config

        # Workers
        self.num_workers = train_config.num_workers
        if self.num_workers <= 0:
            self.num_workers = max(1, os.cpu_count() or 1)

        # Auto-scale batch size and update interval
        batch_size = ppo_config.batch_size
        if batch_size <= 0:
            batch_size = self._auto_batch_size(ppo_config.hidden_dim)
        update_interval = train_config.update_interval
        if update_interval <= 0:
            min_per_worker = 30
            min_for_ppo = max(10, batch_size * 4 // 55)
            update_interval = max(min_for_ppo, min_per_worker * self.num_workers)
        self.effective_batch_size = batch_size
        self.effective_update_interval = update_interval

        logger.info(
            f"Batch size: {batch_size}, Update interval: {update_interval}, "
            f"Workers: {self.num_workers}"
        )

        # Shared embeddings
        self.card_emb = CardEmbedding(ppo_config.card_embed_dim).to(self.device)
        self.player_emb = PlayerEmbedding(embed_dim=ppo_config.player_embed_dim).to(self.device)

        # Networks (share embedding modules)
        self.bid_net = BidActorCritic(
            self.card_emb, self.player_emb, ppo_config.hidden_dim
        ).to(self.device)
        self.play_net = PlayActorCritic(
            self.card_emb, self.player_emb, ppo_config.hidden_dim
        ).to(self.device)

        # Log param counts
        bid_params = sum(p.numel() for p in self.bid_net.parameters())
        play_params = sum(p.numel() for p in self.play_net.parameters())
        # Shared params counted in both, so unique total is less
        shared_params = sum(p.numel() for p in self.card_emb.parameters()) + \
                        sum(p.numel() for p in self.player_emb.parameters())
        logger.info(f"Bid net params: {bid_params:,} | Play net params: {play_params:,} | "
                    f"Shared embed params: {shared_params:,}")

        # PPO
        self.ppo = PPO(
            self.bid_net, self.play_net,
            lr=ppo_config.lr,
            clip_eps=ppo_config.clip_eps,
            value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            epochs=ppo_config.epochs,
            batch_size=batch_size,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )

        # Self-play
        self.opponent_pool = OpponentPool(
            max_pool_size=train_config.pool_size,
            snapshot_interval=train_config.snapshot_interval,
        )
        self.player_elo = 1200.0

        # Buffers
        self.bid_buffer = RolloutBuffer()
        self.play_buffer = RolloutBuffer()

        # Metrics
        self.metrics = MetricsLogger(train_config.log_dir)

        # Tracking
        self.total_games = 0
        self.recent_scores: List[float] = []
        self.recent_wins: List[bool] = []
        self.last_bid_stats = None
        self.last_play_stats = None

        # Exploration bursts
        self._burst_active = False
        self._burst_start_game = 0

        # Persistent worker pool
        self._pool: Optional[mp.Pool] = None

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.bid_net.load_state_dict(ckpt["bid_net"])
        self.play_net.load_state_dict(ckpt["play_net"])
        if "bid_optimizer" in ckpt:
            self.ppo.bid_optimizer.load_state_dict(ckpt["bid_optimizer"])
        if "play_optimizer" in ckpt:
            self.ppo.play_optimizer.load_state_dict(ckpt["play_optimizer"])
        self.player_elo = ckpt.get("player_elo", 1200.0)
        self.total_games = ckpt.get("total_games", 0)
        logger.info(f"Resumed from {path} at game {self.total_games} (ELO {self.player_elo:.0f})")

    def _get_pool(self) -> mp.Pool:
        if self._pool is None:
            mp.set_start_method("fork", force=True)
            self._pool = mp.Pool(self.num_workers, initializer=_worker_init)
        return self._pool

    def _auto_batch_size(self, hidden_dim: int) -> int:
        """Pick batch size based on available GPU memory."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return 256
        try:
            torch.cuda.set_device(self.device)
            free_mem = torch.cuda.mem_get_info(self.device)[0]
        except Exception:
            return 256
        bytes_per_sample = (PLAY_SCALAR_DIM + 2 * hidden_dim + 11) * 4 * 3
        target_mem = free_mem * 0.5
        batch = int(target_mem / bytes_per_sample)
        batch = max(64, min(batch, 65536))
        batch = 1 << (batch.bit_length() - 1)
        logger.info(f"Auto batch size: {batch} (GPU free: {free_mem / 1e9:.1f} GB)")
        return batch

    def train(self) -> None:
        """Main training loop: collect games in parallel, update on GPU."""
        logger.info(f"Starting training for {self.train_config.total_games} games")
        start_time = time.time()

        games_remaining = self.train_config.total_games - self.total_games
        while games_remaining > 0:
            games_this_round = min(self.effective_update_interval, games_remaining)
            self._collect_games(games_this_round)
            self.total_games += games_this_round
            games_remaining -= games_this_round

            # Exploration burst logic (disabled by default, set interval > 0 to enable)
            burst_interval = self.train_config.exploration_burst_interval
            burst_duration = self.train_config.exploration_burst_duration
            if burst_interval > 0:
                if not self._burst_active and self.total_games % burst_interval < self.effective_update_interval:
                    self._start_burst()
                elif self._burst_active and self.total_games - self._burst_start_game >= burst_duration:
                    self._end_burst()

            # PPO update
            self._update_networks()

            # Snapshot for self-play pool
            self.opponent_pool.maybe_snapshot(self.bid_net, self.play_net)

            # Logging
            elapsed = time.time() - start_time
            games_per_sec = self.total_games / max(elapsed, 1)
            avg_score = sum(self.recent_scores) / max(len(self.recent_scores), 1)
            win_rate = sum(self.recent_wins) / max(len(self.recent_wins), 1)
            parts = [
                f"Game {self.total_games}/{self.train_config.total_games}",
                f"ELO {self.player_elo:.0f}",
                f"WR {win_rate:.0%}",
                f"AvgScore {avg_score:.0f}",
            ]
            if self.last_bid_stats:
                parts.append(f"BidLoss {self.last_bid_stats.policy_loss:.3f}")
                parts.append(f"BidEnt {self.last_bid_stats.entropy:.2f}")
            if self.last_play_stats:
                parts.append(f"PlayLoss {self.last_play_stats.policy_loss:.3f}")
                parts.append(f"PlayEnt {self.last_play_stats.entropy:.2f}")
            parts.append(f"{games_per_sec:.1f} g/s")
            logger.info(" | ".join(parts))

            # Save checkpoint
            if self.total_games % self.train_config.save_interval < self.effective_update_interval:
                self._save_checkpoint()

            # Evaluation
            if self.total_games % self.train_config.eval_interval < self.effective_update_interval:
                self._evaluate()

        if self._pool:
            self._pool.terminate()
            self._pool = None
        self._save_checkpoint()
        logger.info("Training complete")

    def _start_burst(self) -> None:
        """Start an exploration burst: inject noise and boost entropy."""
        self._burst_active = True
        self._burst_start_game = self.total_games
        with torch.no_grad():
            for name, param in self.bid_net.named_parameters():
                if "actor" in name:
                    param.data += torch.randn_like(param) * 0.1
            for name, param in self.play_net.named_parameters():
                if "actor" in name:
                    param.data += torch.randn_like(param) * 0.1
        self.ppo.entropy_coef = 0.5
        logger.info(f"Exploration burst started at game {self.total_games}")

    def _end_burst(self) -> None:
        """End an exploration burst: restore entropy coefficient."""
        self._burst_active = False
        self.ppo.entropy_coef = self.ppo_config.entropy_coef
        logger.info(f"Exploration burst ended at game {self.total_games}")

    def _collect_games(self, num_games: int) -> None:
        """Collect transitions from num_games, using parallel workers if available."""
        bid_sd = {k: v.cpu() for k, v in self.bid_net.state_dict().items()}
        play_sd = {k: v.cpu() for k, v in self.play_net.state_dict().items()}

        # Use opponent pool to select opponent type per worker
        opp_type = self.opponent_pool.sample_opponent_type()
        use_heuristic = (opp_type == "heuristic")
        opp_bid_sd = None
        opp_play_sd = None
        if opp_type == "pool":
            entry = self.opponent_pool.get_pool_snapshot()
            if entry is not None:
                opp_bid_sd = {k: v.cpu() if hasattr(v, 'cpu') else v
                              for k, v in entry.bid_state.items()}
                opp_play_sd = {k: v.cpu() if hasattr(v, 'cpu') else v
                               for k, v in entry.play_state.items()}

        common_args = dict(
            bid_state_dict=bid_sd,
            play_state_dict=play_sd,
            hidden_dim=self.ppo_config.hidden_dim,
            num_players=self.game_config.num_players,
            vary_players=self.game_config.vary_players,
            min_players=self.game_config.min_players,
            max_players=self.game_config.max_players,
            use_heuristic_opp=use_heuristic,
            opp_bid_state_dict=opp_bid_sd,
            opp_play_state_dict=opp_play_sd,
            card_embed_dim=self.ppo_config.card_embed_dim,
            player_embed_dim=self.ppo_config.player_embed_dim,
        )

        if self.num_workers > 1:
            pool = self._get_pool()
            games_per_worker = num_games // self.num_workers
            remainder = num_games % self.num_workers
            args_list = []
            for w in range(self.num_workers):
                n = games_per_worker + (1 if w < remainder else 0)
                if n == 0:
                    continue
                args_list.append(dict(
                    num_games=n,
                    seed=random.randint(0, 2**31),
                    **common_args,
                ))
            results = pool.starmap(play_games, [
                (a["num_games"], a["bid_state_dict"], a["play_state_dict"],
                 a["hidden_dim"], a["num_players"], a["vary_players"],
                 a["min_players"], a["max_players"], a["use_heuristic_opp"],
                 a["seed"], a["opp_bid_state_dict"], a["opp_play_state_dict"],
                 a["card_embed_dim"], a["player_embed_dim"])
                for a in args_list
            ])
            for result in results:
                self._ingest_worker_result(result)
        else:
            result = play_games(
                num_games=num_games,
                seed=random.randint(0, 2**31),
                **common_args,
            )
            self._ingest_worker_result(result)

    def _ingest_worker_result(self, result: Dict) -> None:
        """Load worker transitions into buffers and update tracking."""
        for t in result["bid_transitions"]:
            self.bid_buffer.add(
                t["state"], t["action"], t["log_prob"],
                t["value"], t["reward"], t["done"], t["legal_mask"],
            )
        for t in result["play_transitions"]:
            self.play_buffer.add(
                t["state"], t["action"], t["log_prob"],
                t["value"], t["reward"], t["done"], t["legal_mask"],
                t.get("trick_history"),
            )

        for s in result["scores"]:
            self.recent_scores.append(s)
            self.metrics.log_scalar("Train/Reward/Total", s, self.total_games)
        for w in result["wins"]:
            self.recent_wins.append(w)

        self.recent_scores = self.recent_scores[-100:]
        self.recent_wins = self.recent_wins[-100:]

        if result["scores"]:
            wr = sum(result["wins"]) / len(result["wins"])
            expected = 1.0 / (1.0 + 10.0 ** ((1200.0 - self.player_elo) / 400.0))
            self.player_elo += 32.0 * (wr - expected)
            self.metrics.log_scalar("Eval/ELO", self.player_elo, self.total_games)

    def _update_networks(self) -> None:
        """Run PPO updates on both networks and log all diagnostics."""
        if len(self.bid_buffer) > 0:
            # Log bid distribution before clearing
            bid_actions = torch.tensor([t.action for t in self.bid_buffer.transitions])
            self.metrics.log_histogram("Diagnostics/BidDistribution", bid_actions, self.total_games)

            self.last_bid_stats = self.ppo.update(self.bid_buffer, "bid")
            self._log_stats(self.last_bid_stats, "Bid")
            self.bid_buffer.clear()

        if len(self.play_buffer) > 0:
            # Log play action distribution before clearing
            play_actions = torch.tensor([t.action for t in self.play_buffer.transitions])
            self.metrics.log_histogram("Diagnostics/PlayActionDistribution", play_actions, self.total_games)

            self.last_play_stats = self.ppo.update(self.play_buffer, "play")
            self._log_stats(self.last_play_stats, "Play")
            self.play_buffer.clear()

    def _log_stats(self, s: PPOStats, prefix: str) -> None:
        """Log all PPO stats to TensorBoard."""
        step = self.total_games

        # Core PPO metrics
        self.metrics.log_scalar(f"Loss/Policy/{prefix}", s.policy_loss, step)
        self.metrics.log_scalar(f"Loss/Value/{prefix}", s.value_loss, step)
        self.metrics.log_scalar(f"Policy/Entropy/{prefix}", s.entropy, step)
        self.metrics.log_scalar(f"Policy/ClipFraction/{prefix}", s.clip_fraction, step)

        # Critic health
        self.metrics.log_scalar(f"Diagnostics/ExplainedVariance/{prefix}", s.explained_variance, step)
        self.metrics.log_scalar(f"Diagnostics/ValueMAE/{prefix}", s.value_mae, step)

        # Policy stability
        self.metrics.log_scalar(f"Diagnostics/KL/{prefix}", s.kl_divergence, step)
        self.metrics.log_scalar(f"Diagnostics/KLMax/{prefix}", s.approx_kl_max, step)

        # Gradient health
        self.metrics.log_scalar(f"Diagnostics/GradNorm/{prefix}", s.grad_norm, step)

        # Advantage signal quality
        self.metrics.log_scalar(f"Advantage/Mean/{prefix}", s.advantage_mean, step)
        self.metrics.log_scalar(f"Advantage/Std/{prefix}", s.advantage_std, step)
        self.metrics.log_scalar(f"Advantage/Max/{prefix}", s.advantage_max, step)

        # Representation health
        self.metrics.log_scalar(f"Diagnostics/DeadNeurons/{prefix}", s.dead_neuron_frac, step)
        self.metrics.log_scalar(f"Diagnostics/EffectiveRank/{prefix}", s.effective_rank, step)
        self.metrics.log_scalar(f"Diagnostics/ActivationMean/{prefix}", s.activation_mean, step)
        self.metrics.log_scalar(f"Diagnostics/ActivationStd/{prefix}", s.activation_std, step)

        # Parameter health
        self.metrics.log_scalar(f"Diagnostics/WeightNorm/{prefix}", s.weight_norm, step)
        self.metrics.log_scalar(f"Diagnostics/UpdateRatio/{prefix}", s.update_ratio, step)

        # Policy behavior
        self.metrics.log_scalar(f"Diagnostics/ActionDiversity/{prefix}", s.action_diversity, step)

        # Per-layer grad norms
        if s.layer_grad_norms:
            for name, norm in s.layer_grad_norms.items():
                safe_name = name.replace(".", "/")
                self.metrics.log_scalar(f"GradNorm/{prefix}/{safe_name}", norm, step)

    def _evaluate(self) -> None:
        """Evaluate against heuristic opponents."""
        wins = 0
        total_score = 0
        num_eval = 50

        self.bid_net.eval()
        self.play_net.eval()

        for _ in range(num_eval):
            np_ = (random.randint(self.game_config.min_players, self.game_config.max_players)
                   if self.game_config.vary_players else self.game_config.num_players)
            game = SkullKingGame(num_players=np_)
            game.reset()
            heuristic = HeuristicAgent()

            # Trick history tracking for LSTM
            trick_history_list = []
            current_trick_cards = []
            current_trick_players = []
            prev_round = game.round_number

            while not game.is_game_over():
                pid = game.get_current_player()
                state = game.get_state(pid)

                # Reset trick history on new round
                if game.round_number != prev_round:
                    prev_round = game.round_number
                    trick_history_list = []
                    current_trick_cards = []
                    current_trick_players = []

                if game.phase == Phase.BIDDING:
                    if pid == 0:
                        features = encode_bid_state_v2(state)
                        mask = get_legal_bid_mask(state)
                        action, _, _, _ = self.bid_net.get_action_and_value(features, mask)
                        game.step_bid(pid, action)
                    else:
                        game.step_bid(pid, heuristic.choose_bid(state))

                elif game.phase == Phase.PLAYING:
                    if pid == 0:
                        features = encode_play_state_v2(state)
                        mask = get_legal_play_mask(state)
                        trick_hist = encode_trick_history_v2(trick_history_list, np_)
                        action, _, _, _ = self.play_net.get_action_and_value(
                            features, mask, trick_hist)
                        tigress = None
                        if action < len(state.hand) and state.hand[action].special == SpecialType.TIGRESS:
                            tigress = state.all_bids[0] > state.all_tricks_won[0]
                        played_card = state.hand[action] if action < len(state.hand) else state.hand[0]
                        current_trick_cards.append(played_card.card_id)
                        current_trick_players.append(pid)
                        result = game.step_play(pid, action, tigress)
                    else:
                        hi, tig = heuristic.choose_play(state)
                        if hi < len(state.hand):
                            current_trick_cards.append(state.hand[hi].card_id)
                        current_trick_players.append(pid)
                        result = game.step_play(pid, hi, tig)

                    # Record completed trick
                    if result is not None:
                        winner_idx = result.winner_index
                        trick_history_list.append({
                            "card_ids": list(current_trick_cards),
                            "player_ids": list(current_trick_players),
                            "winner_id": current_trick_players[winner_idx]
                                if winner_idx < len(current_trick_players) else 0,
                        })
                        current_trick_cards = []
                        current_trick_players = []

            if game.get_winner() == 0:
                wins += 1
            total_score += game.players[0].score

        self.bid_net.train()
        self.play_net.train()

        win_rate = wins / num_eval
        avg_score = total_score / num_eval
        self.metrics.log_scalar("Eval/WinRate", win_rate, self.total_games)
        self.metrics.log_scalar("Eval/AvgScore", avg_score, self.total_games)
        logger.info(f"Eval: WinRate={win_rate:.2%}, AvgScore={avg_score:.1f}")

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        path = Path(self.train_config.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "bid_net": self.bid_net.state_dict(),
            "play_net": self.play_net.state_dict(),
            "bid_optimizer": self.ppo.bid_optimizer.state_dict(),
            "play_optimizer": self.ppo.play_optimizer.state_dict(),
            "player_elo": self.player_elo,
            "total_games": self.total_games,
        }, path / f"checkpoint_{self.total_games}.pt")
        torch.save({
            "bid_net": self.bid_net.state_dict(),
            "play_net": self.play_net.state_dict(),
            "player_elo": self.player_elo,
            "total_games": self.total_games,
        }, path / "latest.pt")
        logger.info(f"Saved checkpoint at game {self.total_games}")
