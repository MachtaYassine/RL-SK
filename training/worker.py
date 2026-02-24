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
from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from networks.features import (
    encode_bid_state_v2, encode_play_state_v2,
    get_legal_bid_mask, get_legal_play_mask,
    encode_trick_history_v2,
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
    opp_bid_state_dict: dict = None,
    opp_play_state_dict: dict = None,
    card_embed_dim: int = 16,
    player_embed_dim: int = 4,
) -> Dict:
    """Play multiple games and return collected transitions + stats."""
    rng = random.Random(seed)

    # Reconstruct networks with shared embeddings
    card_emb = CardEmbedding(card_embed_dim)
    player_emb = PlayerEmbedding(embed_dim=player_embed_dim)

    bid_net = BidActorCritic(card_emb, player_emb, hidden_dim)
    bid_net.load_state_dict(bid_state_dict)
    bid_net.eval()

    play_net = PlayActorCritic(card_emb, player_emb, hidden_dim)
    play_net.load_state_dict(play_state_dict)
    play_net.eval()

    # Opponent network (pool snapshot or current self)
    if opp_bid_state_dict is not None and not use_heuristic_opp:
        opp_card_emb = CardEmbedding(card_embed_dim)
        opp_player_emb = PlayerEmbedding(embed_dim=player_embed_dim)
        opp_bid_net = BidActorCritic(opp_card_emb, opp_player_emb, hidden_dim)
        opp_bid_net.load_state_dict(opp_bid_state_dict)
        opp_bid_net.eval()
        opp_play_net = PlayActorCritic(opp_card_emb, opp_player_emb, hidden_dim)
        opp_play_net.load_state_dict(opp_play_state_dict)
        opp_play_net.eval()
    else:
        opp_bid_net = bid_net
        opp_play_net = play_net

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
        round_bid_trans: List[dict] = []
        round_play_trans: List[dict] = []
        all_bid_trans: List[dict] = []
        all_play_trans: List[dict] = []

        prev_round = game.round_number
        prev_tricks_won = 0
        player_bid = 0

        # Trick history tracking for LSTM
        trick_history_list: List[dict] = []
        current_trick_card_ids: List[int] = []
        current_trick_player_ids: List[int] = []

        while not game.is_game_over():
            pid = game.get_current_player()
            cur_round = game.round_number

            # Detect round boundary — assign round rewards
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
                trick_history_list = []
                current_trick_card_ids = []
                current_trick_player_ids = []

            state = game.get_state(pid)

            if game.phase == Phase.BIDDING:
                if pid == 0:
                    features = encode_bid_state_v2(state)
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
                        features = encode_bid_state_v2(state)
                        mask = get_legal_bid_mask(state)
                        action, _, _, _ = opp_bid_net.get_action_and_value(features, mask)
                        bid = action
                    game.step_bid(pid, bid)

            elif game.phase == Phase.PLAYING:
                if pid == 0:
                    features = encode_play_state_v2(state)
                    mask = get_legal_play_mask(state)
                    trick_hist = encode_trick_history_v2(trick_history_list, np_)
                    with torch.no_grad():
                        action, log_prob, value, entropy = play_net.get_action_and_value(
                            features, mask, trick_hist)
                    tigress = None
                    if action < len(state.hand) and state.hand[action].special == SpecialType.TIGRESS:
                        tigress = state.all_bids[pid] > state.all_tricks_won[pid]

                    # Track the card being played for trick history
                    played_card = state.hand[action] if action < len(state.hand) else state.hand[0]
                    current_trick_card_ids.append(played_card.card_id)
                    current_trick_player_ids.append(pid)

                    prev_tricks_won = game.players[0].tricks_won
                    result = game.step_play(pid, action, tigress)

                    # Intermediate trick reward
                    trick_reward = 0.0
                    if result is not None:
                        new_tricks = game.players[0].tricks_won
                        won_trick = new_tricks > prev_tricks_won
                        if won_trick:
                            if new_tricks <= player_bid:
                                trick_reward = 0.1
                            else:
                                trick_reward = -0.1
                        else:
                            if prev_tricks_won < player_bid:
                                trick_reward = -0.05

                        # Record completed trick
                        winner_idx = result.winner_index
                        trick_history_list.append({
                            "card_ids": list(current_trick_card_ids),
                            "player_ids": list(current_trick_player_ids),
                            "winner_id": current_trick_player_ids[winner_idx]
                                if winner_idx < len(current_trick_player_ids) else 0,
                        })
                        current_trick_card_ids = []
                        current_trick_player_ids = []

                    round_play_trans.append({
                        "state": features, "action": action, "log_prob": log_prob,
                        "value": value, "reward": trick_reward, "done": False,
                        "legal_mask": mask, "trick_history": trick_hist,
                    })
                else:
                    if use_heuristic_opp:
                        hi, tig = heuristic.choose_play(state)
                    else:
                        features = encode_play_state_v2(state)
                        mask = get_legal_play_mask(state)
                        action, _, _, _ = opp_play_net.get_action_and_value(features, mask)
                        hi = action
                        tig = None
                        if hi < len(state.hand) and state.hand[hi].special == SpecialType.TIGRESS:
                            tig = True

                    # Track opponent's card for trick history
                    if hi < len(state.hand):
                        current_trick_card_ids.append(state.hand[hi].card_id)
                    current_trick_player_ids.append(pid)

                    result = game.step_play(pid, hi, tig)

                    # Record completed trick from opponent's play
                    if result is not None:
                        winner_idx = result.winner_index
                        trick_history_list.append({
                            "card_ids": list(current_trick_card_ids),
                            "player_ids": list(current_trick_player_ids),
                            "winner_id": current_trick_player_ids[winner_idx]
                                if winner_idx < len(current_trick_player_ids) else 0,
                        })
                        current_trick_card_ids = []
                        current_trick_player_ids = []

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
        rank = sum(1 for s in all_scores if s > player_score)
        placement_reward = (np_ - 1 - 2 * rank) / max(np_ - 1, 1)

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
    """Assign normalized round score to transitions within this round."""
    normalized = round_score / 100.0

    if bid_trans:
        bid_trans[-1]["reward"] = normalized
        # done=False: GAE bootstraps through rounds for long-horizon learning
        # done=True is only set at actual game end

    if play_trans:
        play_trans[-1]["reward"] += normalized
        # done=False: GAE bootstraps through rounds
