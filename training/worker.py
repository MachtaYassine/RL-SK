"""Game simulation worker for parallel data collection.

Each worker runs games on CPU using copies of the networks,
collects transitions, and returns them to the main process.

Reward shaping:
- Each bid transition gets the normalized round score for that round.
- Each play transition gets a small trick reward (+0.1 if trick won helps
  toward bid, -0.1 if trick won hurts) plus the normalized round score
  at the end of the round.
- The last transition of the game also gets a game-end bonus based on
  final placement (win=+1, relative score otherwise).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch

from agents.heuristic_agent import HeuristicAgent
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from networks.features import (
    encode_bid_state, encode_play_state,
    get_legal_bid_mask, get_legal_play_mask,
)
from skull_king.cards import SpecialType
from skull_king.game import SkullKingGame, Phase
from skull_king.scoring import score_round


def play_games(
    num_games: int,
    bid_state_dict: dict,
    play_state_dict: dict,
    hidden_dim: int,
    num_players: int,
    vary_players: bool,
    min_players: int,
    max_players: int,
    use_heuristic_opp: bool,
    seed: int,
) -> Dict:
    """Play multiple games and return collected transitions + stats."""
    rng = random.Random(seed)

    bid_net = BidActorCritic(hidden_dim)
    bid_net.load_state_dict(bid_state_dict)
    bid_net.eval()

    play_net = PlayActorCritic(hidden_dim)
    play_net.load_state_dict(play_state_dict)
    play_net.eval()

    heuristic = HeuristicAgent()

    bid_transitions: List[dict] = []
    play_transitions: List[dict] = []
    scores: List[float] = []
    wins: List[bool] = []

    for g in range(num_games):
        np_ = rng.randint(min_players, max_players) if vary_players else num_players
        game = SkullKingGame(num_players=np_, seed=rng.randint(0, 2**31))
        game.reset()

        # Per-round transition tracking
        round_bid_trans: List[dict] = []   # bid transitions for current round
        round_play_trans: List[dict] = []  # play transitions for current round
        all_bid_trans: List[dict] = []     # all bid transitions for the game
        all_play_trans: List[dict] = []    # all play transitions for the game

        prev_round = game.round_number
        prev_tricks_won = 0  # player 0's tricks before this play action
        player_bid = 0       # player 0's bid for current round

        while not game.is_game_over():
            pid = game.get_current_player()
            cur_round = game.round_number

            # Detect round boundary — assign round rewards to buffered transitions
            if cur_round != prev_round and prev_round in game.round_scores:
                _assign_round_rewards(
                    round_bid_trans, round_play_trans,
                    game.round_scores[prev_round][0], prev_round,
                )
                all_bid_trans.extend(round_bid_trans)
                all_play_trans.extend(round_play_trans)
                round_bid_trans = []
                round_play_trans = []
                prev_round = cur_round
                prev_tricks_won = 0

            state = game.get_state(pid)

            if game.phase == Phase.BIDDING:
                if pid == 0:
                    features = encode_bid_state(state)
                    mask = get_legal_bid_mask(state)
                    with torch.no_grad():
                        action, log_prob, value, entropy = bid_net.get_action_and_value(features, mask)
                    round_bid_trans.append({
                        "state": features, "action": action, "log_prob": log_prob,
                        "value": value, "reward": 0.0, "done": False, "legal_mask": mask,
                    })
                    player_bid = action
                    game.step_bid(pid, action)
                else:
                    if use_heuristic_opp:
                        bid = heuristic.choose_bid(state)
                    else:
                        features = encode_bid_state(state)
                        mask = get_legal_bid_mask(state)
                        action, _, _, _ = bid_net.get_action_and_value(features, mask)
                        bid = action
                    game.step_bid(pid, bid)

            elif game.phase == Phase.PLAYING:
                if pid == 0:
                    features = encode_play_state(state)
                    mask = get_legal_play_mask(state)
                    with torch.no_grad():
                        action, log_prob, value, entropy = play_net.get_action_and_value(features, mask)
                    tigress = None
                    if action < len(state.hand) and state.hand[action].special == SpecialType.TIGRESS:
                        tigress = state.all_bids[pid] > state.all_tricks_won[pid]

                    prev_tricks_won = game.players[0].tricks_won
                    result = game.step_play(pid, action, tigress)

                    # Intermediate trick reward
                    trick_reward = 0.0
                    if result is not None:
                        new_tricks = game.players[0].tricks_won
                        won_trick = new_tricks > prev_tricks_won
                        if won_trick:
                            # Won a trick: good if under/at bid, bad if over bid
                            if new_tricks <= player_bid:
                                trick_reward = 0.1
                            else:
                                trick_reward = -0.1
                        else:
                            # Didn't win: good if already at bid, bad if under bid
                            if prev_tricks_won < player_bid:
                                trick_reward = -0.05
                            # else: neutral

                    round_play_trans.append({
                        "state": features, "action": action, "log_prob": log_prob,
                        "value": value, "reward": trick_reward, "done": False, "legal_mask": mask,
                    })
                else:
                    if use_heuristic_opp:
                        hi, tig = heuristic.choose_play(state)
                    else:
                        features = encode_play_state(state)
                        mask = get_legal_play_mask(state)
                        action, _, _, _ = play_net.get_action_and_value(features, mask)
                        hi = action
                        tig = None
                        if hi < len(state.hand) and state.hand[hi].special == SpecialType.TIGRESS:
                            tig = True
                    game.step_play(pid, hi, tig)

        # Flush last round's transitions
        if prev_round in game.round_scores:
            _assign_round_rewards(
                round_bid_trans, round_play_trans,
                game.round_scores[prev_round][0], prev_round,
            )
        all_bid_trans.extend(round_bid_trans)
        all_play_trans.extend(round_play_trans)

        # Game-end bonus: placement reward
        player_score = game.players[0].score
        all_scores = [game.players[i].score for i in range(np_)]
        rank = sum(1 for s in all_scores if s > player_score)  # 0 = first place
        placement_reward = (np_ - 1 - 2 * rank) / max(np_ - 1, 1)  # +1 for 1st, -1 for last

        # Add placement bonus to last transitions
        if all_bid_trans:
            all_bid_trans[-1]["reward"] += placement_reward
            all_bid_trans[-1]["done"] = True
        if all_play_trans:
            all_play_trans[-1]["reward"] += placement_reward
            all_play_trans[-1]["done"] = True

        bid_transitions.extend(all_bid_trans)
        play_transitions.extend(all_play_trans)
        scores.append(player_score)
        wins.append(game.get_winner() == 0)

    return {
        "bid_transitions": bid_transitions,
        "play_transitions": play_transitions,
        "scores": scores,
        "wins": wins,
    }


def _assign_round_rewards(
    bid_trans: List[dict],
    play_trans: List[dict],
    round_score: int,
    round_number: int,
) -> None:
    """Assign normalized round score to transitions within this round.

    The bid transition gets the full round score signal since bidding
    directly determines the scoring function.
    The last play transition of the round gets the round score on top
    of any trick rewards.
    """
    # Normalize: typical round scores range from -100 to +100
    normalized = round_score / 100.0

    # Bid: the bid directly caused this score
    if bid_trans:
        bid_trans[-1]["reward"] = normalized
        bid_trans[-1]["done"] = True  # Episode boundary for GAE

    # Play: last play transition gets the round-end signal
    if play_trans:
        play_trans[-1]["reward"] += normalized
        play_trans[-1]["done"] = True  # Episode boundary for GAE
