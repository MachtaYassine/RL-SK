"""Feature extraction: GameState -> tensor for bid and play networks.

Bid features (159):
    hand_presence[70] + cards_seen[70] + round[1] + num_players[1] +
    bid_position[1] + other_bids[8] + scores[8]

Play features (195):
    hand_presence[70] + cards_seen[70] + legal_mask[10] + trick_card_ids[8] +
    trick_player_ids[8] + bid[1] + tricks_won[1] + all_bids[8] +
    all_tricks_won[8] + round[1] + num_players[1] + tricks_remaining[1] +
    position_in_trick[1] + winner_id[1] + winning_card_id[1] + lead_suit[5]
"""

from __future__ import annotations

import torch
from skull_king.cards import NUM_CARDS
from skull_king.game import GameState, Phase

BID_DIM = 159
PLAY_DIM = 195
MAX_HAND_SIZE = 10
MAX_PLAYERS = 8


def encode_bid_state(state: GameState) -> torch.Tensor:
    """Encode a bidding-phase game state into a tensor."""
    assert state.phase == Phase.BIDDING
    features = torch.zeros(BID_DIM)
    idx = 0

    for card in state.hand:
        features[idx + card.card_id] = 1.0
    idx += NUM_CARDS

    for cid in state.cards_seen:
        features[idx + cid] = 1.0
    idx += NUM_CARDS

    features[idx] = state.round_number / 10.0
    idx += 1

    features[idx] = state.num_players / 8.0
    idx += 1

    features[idx] = state.bid_position / max(state.num_players - 1, 1)
    idx += 1

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = max(state.all_bids[i], 0) / max(state.round_number, 1)
    idx += MAX_PLAYERS

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = state.all_scores[i] / 500.0
    idx += MAX_PLAYERS

    assert idx == BID_DIM
    return features


def encode_play_state(state: GameState) -> torch.Tensor:
    """Encode a playing-phase game state into a tensor."""
    assert state.phase == Phase.PLAYING
    features = torch.zeros(PLAY_DIM)
    idx = 0

    for card in state.hand:
        features[idx + card.card_id] = 1.0
    idx += NUM_CARDS

    for cid in state.cards_seen:
        features[idx + cid] = 1.0
    idx += NUM_CARDS

    for action in state.legal_actions:
        if action < MAX_HAND_SIZE:
            features[idx + action] = 1.0
    idx += MAX_HAND_SIZE

    for i in range(MAX_PLAYERS):
        if i < len(state.current_trick_cards):
            features[idx + i] = state.current_trick_cards[i].card_id / NUM_CARDS
    idx += MAX_PLAYERS

    for i in range(MAX_PLAYERS):
        if i < len(state.current_trick_players):
            features[idx + i] = state.current_trick_players[i] / max(state.num_players - 1, 1)
    idx += MAX_PLAYERS

    features[idx] = state.all_bids[state.player_id] / max(state.round_number, 1)
    idx += 1

    features[idx] = state.all_tricks_won[state.player_id] / max(state.round_number, 1)
    idx += 1

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = max(state.all_bids[i], 0) / max(state.round_number, 1)
    idx += MAX_PLAYERS

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = state.all_tricks_won[i] / max(state.round_number, 1)
    idx += MAX_PLAYERS

    features[idx] = state.round_number / 10.0
    idx += 1

    features[idx] = state.num_players / 8.0
    idx += 1

    features[idx] = state.tricks_remaining / max(state.round_number, 1)
    idx += 1

    features[idx] = state.position_in_trick / max(state.num_players - 1, 1)
    idx += 1

    features[idx] = (state.current_winner_id + 1) / (state.num_players + 1)
    idx += 1

    features[idx] = (state.current_winning_card_id + 1) / (NUM_CARDS + 1)
    idx += 1

    if state.lead_suit is not None:
        features[idx + state.lead_suit.value] = 1.0
    else:
        features[idx + 4] = 1.0
    idx += 5

    assert idx == PLAY_DIM, f"Expected {PLAY_DIM}, got {idx}"
    return features


def get_legal_bid_mask(state: GameState) -> torch.Tensor:
    """Return mask of legal bid values. Shape: [11] (bids 0-10)."""
    mask = torch.zeros(11)
    for a in state.legal_actions:
        if a < 11:
            mask[a] = 1.0
    return mask


def get_legal_play_mask(state: GameState) -> torch.Tensor:
    """Return mask of legal play indices. Shape: [10]."""
    mask = torch.zeros(MAX_HAND_SIZE)
    for a in state.legal_actions:
        if a < MAX_HAND_SIZE:
            mask[a] = 1.0
    return mask
