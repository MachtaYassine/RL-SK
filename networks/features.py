"""Feature extraction: GameState -> tensors for bid and play networks.

V2 functions return dicts of typed tensors (card IDs as LongTensor for
embedding lookup, scalar context as float). This replaces the v1 flat
one-hot encoding which was sparse and lost card structure.

Bid scalar context (19):
    round(1) + num_players(1) + bid_position(1) + other_bids(8) + scores(8)

Play scalar context (37):
    legal_mask(10) + bid(1) + tricks_won(1) + all_bids(8) + all_tricks_won(8)
    + round(1) + num_players(1) + tricks_remaining(1) + position_in_trick(1)
    + lead_suit(5)
"""

from __future__ import annotations

import torch
from skull_king.cards import NUM_CARDS
from skull_king.game import GameState, Phase

MAX_HAND_SIZE = 10
MAX_PLAYERS = 8
MAX_TRICKS = 10

BID_SCALAR_DIM = 19
PLAY_SCALAR_DIM = 37

# Legacy constants (kept for reference, no longer used by networks)
BID_DIM = 159
PLAY_DIM = 195
TRICK_FEAT_DIM = 17


# ---------------------------------------------------------------------------
# V2 encode functions — return dicts of typed tensors
# ---------------------------------------------------------------------------

def encode_bid_state_v2(state: GameState) -> dict:
    """Encode bidding-phase state into dict of tensors.

    Returns:
        dict with keys:
            hand_ids:  LongTensor[MAX_HAND_SIZE]   card IDs (0-padded)
            hand_mask: Tensor[MAX_HAND_SIZE]        1.0 where card exists
            seen_ids:  LongTensor[NUM_CARDS]        card IDs of seen cards (0-padded)
            seen_mask: Tensor[NUM_CARDS]            1.0 where seen
            scalars:   Tensor[BID_SCALAR_DIM]       normalized scalar features
    """
    assert state.phase == Phase.BIDDING

    hand_ids = torch.zeros(MAX_HAND_SIZE, dtype=torch.long)
    hand_mask = torch.zeros(MAX_HAND_SIZE)
    for i, card in enumerate(state.hand):
        if i < MAX_HAND_SIZE:
            hand_ids[i] = card.card_id
            hand_mask[i] = 1.0

    seen_ids = torch.zeros(NUM_CARDS, dtype=torch.long)
    seen_mask = torch.zeros(NUM_CARDS)
    for j, cid in enumerate(state.cards_seen):
        if j < NUM_CARDS:
            seen_ids[j] = cid
            seen_mask[j] = 1.0

    scalars = torch.zeros(BID_SCALAR_DIM)
    idx = 0
    scalars[idx] = state.round_number / 10.0; idx += 1
    scalars[idx] = state.num_players / 8.0; idx += 1
    scalars[idx] = state.bid_position / max(state.num_players - 1, 1); idx += 1
    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            scalars[idx + i] = max(state.all_bids[i], 0) / max(state.round_number, 1)
    idx += MAX_PLAYERS
    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            scalars[idx + i] = state.all_scores[i] / 500.0
    idx += MAX_PLAYERS
    assert idx == BID_SCALAR_DIM

    return {
        "hand_ids": hand_ids,
        "hand_mask": hand_mask,
        "seen_ids": seen_ids,
        "seen_mask": seen_mask,
        "scalars": scalars,
    }


def encode_play_state_v2(state: GameState) -> dict:
    """Encode playing-phase state into dict of tensors.

    Returns:
        dict with keys:
            hand_ids, hand_mask:                 as bid
            seen_ids, seen_mask:                 as bid
            trick_card_ids:  LongTensor[MAX_PLAYERS]   current trick card IDs
            trick_card_mask: Tensor[MAX_PLAYERS]        1.0 where card played
            trick_player_ids: LongTensor[MAX_PLAYERS]   who played each card
            trick_player_mask: Tensor[MAX_PLAYERS]
            scalars:         Tensor[PLAY_SCALAR_DIM]
    """
    assert state.phase == Phase.PLAYING

    hand_ids = torch.zeros(MAX_HAND_SIZE, dtype=torch.long)
    hand_mask = torch.zeros(MAX_HAND_SIZE)
    for i, card in enumerate(state.hand):
        if i < MAX_HAND_SIZE:
            hand_ids[i] = card.card_id
            hand_mask[i] = 1.0

    seen_ids = torch.zeros(NUM_CARDS, dtype=torch.long)
    seen_mask = torch.zeros(NUM_CARDS)
    for j, cid in enumerate(state.cards_seen):
        if j < NUM_CARDS:
            seen_ids[j] = cid
            seen_mask[j] = 1.0

    trick_card_ids = torch.zeros(MAX_PLAYERS, dtype=torch.long)
    trick_card_mask = torch.zeros(MAX_PLAYERS)
    for i, card in enumerate(state.current_trick_cards):
        if i < MAX_PLAYERS:
            trick_card_ids[i] = card.card_id
            trick_card_mask[i] = 1.0

    trick_player_ids = torch.zeros(MAX_PLAYERS, dtype=torch.long)
    trick_player_mask = torch.zeros(MAX_PLAYERS)
    for i, pid in enumerate(state.current_trick_players):
        if i < MAX_PLAYERS:
            trick_player_ids[i] = pid
            trick_player_mask[i] = 1.0

    scalars = torch.zeros(PLAY_SCALAR_DIM)
    idx = 0

    # Legal mask (10)
    for action in state.legal_actions:
        if action < MAX_HAND_SIZE:
            scalars[idx + action] = 1.0
    idx += MAX_HAND_SIZE

    scalars[idx] = state.all_bids[state.player_id] / max(state.round_number, 1); idx += 1
    scalars[idx] = state.all_tricks_won[state.player_id] / max(state.round_number, 1); idx += 1

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            scalars[idx + i] = max(state.all_bids[i], 0) / max(state.round_number, 1)
    idx += MAX_PLAYERS

    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            scalars[idx + i] = state.all_tricks_won[i] / max(state.round_number, 1)
    idx += MAX_PLAYERS

    scalars[idx] = state.round_number / 10.0; idx += 1
    scalars[idx] = state.num_players / 8.0; idx += 1
    scalars[idx] = state.tricks_remaining / max(state.round_number, 1); idx += 1
    scalars[idx] = state.position_in_trick / max(state.num_players - 1, 1); idx += 1

    # Lead suit one-hot (5: 4 suits + "no lead")
    if state.lead_suit is not None:
        scalars[idx + state.lead_suit.value] = 1.0
    else:
        scalars[idx + 4] = 1.0
    idx += 5

    assert idx == PLAY_SCALAR_DIM, f"Expected {PLAY_SCALAR_DIM}, got {idx}"

    return {
        "hand_ids": hand_ids,
        "hand_mask": hand_mask,
        "seen_ids": seen_ids,
        "seen_mask": seen_mask,
        "trick_card_ids": trick_card_ids,
        "trick_card_mask": trick_card_mask,
        "trick_player_ids": trick_player_ids,
        "trick_player_mask": trick_player_mask,
        "scalars": scalars,
    }


def encode_trick_history_v2(trick_history: list, num_players: int) -> dict:
    """Encode completed tricks as structured tensors for embedding lookup.

    Args:
        trick_history: List of dicts with 'card_ids', 'player_ids', 'winner_id'.
        num_players: Number of players in the game.

    Returns:
        dict with keys:
            card_ids:     LongTensor[MAX_TRICKS, MAX_PLAYERS]
            card_masks:   Tensor[MAX_TRICKS, MAX_PLAYERS]
            player_ids:   LongTensor[MAX_TRICKS, MAX_PLAYERS]
            player_masks: Tensor[MAX_TRICKS, MAX_PLAYERS]
            winner_ids:   LongTensor[MAX_TRICKS]
            trick_mask:   Tensor[MAX_TRICKS]   1.0 for tricks that happened
    """
    card_ids = torch.zeros(MAX_TRICKS, MAX_PLAYERS, dtype=torch.long)
    card_masks = torch.zeros(MAX_TRICKS, MAX_PLAYERS)
    player_ids = torch.zeros(MAX_TRICKS, MAX_PLAYERS, dtype=torch.long)
    player_masks = torch.zeros(MAX_TRICKS, MAX_PLAYERS)
    winner_ids = torch.zeros(MAX_TRICKS, dtype=torch.long)
    trick_mask = torch.zeros(MAX_TRICKS)

    for t, trick in enumerate(trick_history[:MAX_TRICKS]):
        trick_mask[t] = 1.0
        winner_ids[t] = trick["winner_id"]
        for i in range(min(len(trick["card_ids"]), MAX_PLAYERS)):
            card_ids[t, i] = trick["card_ids"][i]
            card_masks[t, i] = 1.0
        for i in range(min(len(trick["player_ids"]), MAX_PLAYERS)):
            player_ids[t, i] = trick["player_ids"][i]
            player_masks[t, i] = 1.0

    return {
        "card_ids": card_ids,
        "card_masks": card_masks,
        "player_ids": player_ids,
        "player_masks": player_masks,
        "winner_ids": winner_ids,
        "trick_mask": trick_mask,
    }


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


# ---------------------------------------------------------------------------
# Legacy V1 functions (kept for reference, no longer called by networks)
# ---------------------------------------------------------------------------

def encode_bid_state(state: GameState) -> torch.Tensor:
    """V1: Encode a bidding-phase game state into a flat tensor."""
    assert state.phase == Phase.BIDDING
    features = torch.zeros(BID_DIM)
    idx = 0
    for card in state.hand:
        features[idx + card.card_id] = 1.0
    idx += NUM_CARDS
    for cid in state.cards_seen:
        features[idx + cid] = 1.0
    idx += NUM_CARDS
    features[idx] = state.round_number / 10.0; idx += 1
    features[idx] = state.num_players / 8.0; idx += 1
    features[idx] = state.bid_position / max(state.num_players - 1, 1); idx += 1
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
    """V1: Encode a playing-phase game state into a flat tensor."""
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
    features[idx] = state.all_bids[state.player_id] / max(state.round_number, 1); idx += 1
    features[idx] = state.all_tricks_won[state.player_id] / max(state.round_number, 1); idx += 1
    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = max(state.all_bids[i], 0) / max(state.round_number, 1)
    idx += MAX_PLAYERS
    for i in range(MAX_PLAYERS):
        if i < state.num_players:
            features[idx + i] = state.all_tricks_won[i] / max(state.round_number, 1)
    idx += MAX_PLAYERS
    features[idx] = state.round_number / 10.0; idx += 1
    features[idx] = state.num_players / 8.0; idx += 1
    features[idx] = state.tricks_remaining / max(state.round_number, 1); idx += 1
    features[idx] = state.position_in_trick / max(state.num_players - 1, 1); idx += 1
    features[idx] = (state.current_winner_id + 1) / (state.num_players + 1); idx += 1
    features[idx] = (state.current_winning_card_id + 1) / (NUM_CARDS + 1); idx += 1
    if state.lead_suit is not None:
        features[idx + state.lead_suit.value] = 1.0
    else:
        features[idx + 4] = 1.0
    idx += 5
    assert idx == PLAY_DIM, f"Expected {PLAY_DIM}, got {idx}"
    return features


def encode_trick_history(trick_history: list, num_players: int) -> torch.Tensor:
    """V1: Encode completed tricks into a fixed-size tensor."""
    result = torch.zeros(MAX_TRICKS, TRICK_FEAT_DIM)
    for t, trick in enumerate(trick_history[:MAX_TRICKS]):
        idx = 0
        for i in range(MAX_PLAYERS):
            if i < len(trick["card_ids"]):
                result[t, idx + i] = trick["card_ids"][i] / NUM_CARDS
        idx += MAX_PLAYERS
        for i in range(MAX_PLAYERS):
            if i < len(trick["player_ids"]):
                result[t, idx + i] = trick["player_ids"][i] / max(num_players - 1, 1)
        idx += MAX_PLAYERS
        result[t, idx] = trick["winner_id"] / max(num_players - 1, 1)
    return result
