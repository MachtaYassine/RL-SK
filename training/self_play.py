"""Self-play opponent pool with ELO tracking."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from agents.base import Agent
from agents.heuristic_agent import HeuristicAgent
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic


@dataclass
class PoolEntry:
    bid_state: dict
    play_state: dict
    elo: float = 1200.0
    games_played: int = 0


class OpponentPool:
    """Manages a pool of past agent snapshots for self-play.

    Opponent selection: 50% latest, 30% pool sample, 20% heuristic.
    """

    def __init__(self, max_pool_size: int = 20, snapshot_interval: int = 50):
        self.pool: List[PoolEntry] = []
        self.max_pool_size = max_pool_size
        self.snapshot_interval = snapshot_interval
        self.games_since_snapshot = 0
        self.heuristic = HeuristicAgent()

    def maybe_snapshot(self, bid_net: BidActorCritic, play_net: PlayActorCritic) -> None:
        """Snapshot current networks if interval reached."""
        self.games_since_snapshot += 1
        if self.games_since_snapshot >= self.snapshot_interval:
            self.games_since_snapshot = 0
            entry = PoolEntry(
                bid_state=copy.deepcopy(bid_net.state_dict()),
                play_state=copy.deepcopy(play_net.state_dict()),
            )
            self.pool.append(entry)
            if len(self.pool) > self.max_pool_size:
                self.pool.pop(0)  # Remove oldest

    def sample_opponent_type(self) -> str:
        """Sample opponent type: 'latest', 'pool', or 'heuristic'."""
        r = random.random()
        if r < 0.5:
            return "latest"
        elif r < 0.8 and len(self.pool) > 0:
            return "pool"
        else:
            return "heuristic"

    def get_pool_snapshot(self) -> Optional[PoolEntry]:
        """Get a random snapshot from the pool."""
        if not self.pool:
            return None
        return random.choice(self.pool)

    def update_elo(self, player_score: float, opponent_score: float,
                   opponent_entry: Optional[PoolEntry], player_elo: float,
                   k: float = 32.0) -> float:
        """Update ELO ratings after a game.

        Returns updated player ELO.
        """
        opponent_elo = opponent_entry.elo if opponent_entry else 1200.0

        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))
        actual = 1.0 if player_score > opponent_score else (0.5 if player_score == opponent_score else 0.0)
        new_player_elo = player_elo + k * (actual - expected)

        if opponent_entry:
            opp_expected = 1.0 - expected
            opp_actual = 1.0 - actual
            opponent_entry.elo += k * (opp_actual - opp_expected)
            opponent_entry.games_played += 1

        return new_player_elo
