import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .SimpleAgent import SkullKingAgent

# Helper: process hand information
def process_hand(hand, hand_size, suit_mapping, legal_actions=None):
    hand_vec = []
    for card in hand:
        # If the card is not legal, replace with [0, 0, 0, 0, 0]
        if legal_actions and card not in legal_actions:
            hand_vec.extend([0, 0, 0, 0, 0])
            continue
        
        # One-hot encoding for the suit
        one_hot = [0, 0, 0, 0]
        if card[0] in suit_mapping:
            one_hot[suit_mapping[card[0]]] = 1
        
        # Convert rank to float
        try:
            rank = float(card[1])
        except (ValueError, TypeError):
            rank = 0.0
        
        # Append one-hot suit encoding + rank
        hand_vec.extend(one_hot + [rank])
    
    # Ensure fixed hand size by padding
    while len(hand_vec) < hand_size * 5:
        hand_vec.extend([0] * 5)
    
    return hand_vec


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(PolicyNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LearningSkullKingAgent(SkullKingAgent):
    def __init__(self, num_players, hand_size=10, learning_rate=1e-3, shared_bid_net=None, shared_play_net=None,
                 shared_trick_win_predictor=None,   # NEW parameter
                 suit_count=4, max_rank=14, max_players=5, bid_dimension_vars=None, play_dimension_vars=None,
                 bid_hidden_dims=None, play_hidden_dims=None):
        super().__init__(num_players)
        self.hand_size = hand_size
        self.suit_count = suit_count
        self.max_rank = max_rank
        self.max_players = max_players
        self.bid_dimension_vars = bid_dimension_vars or []
        self.play_dimension_vars = play_dimension_vars or []
        self.log_probs = []
        # NEW: Use hardcoded extra features dimensions:
        bid_input_dim = hand_size * 5 + 4 + self.max_players   # 10 from total_scores + 4 extra scalars: position_in_order, player_id, round_number, bidding_phase
        play_input_dim = hand_size * 5 + self.max_players * 5 +3+ 1 + 1 +self.max_players+ self.max_players +1 +5 +2   # Computed as: hand_vec + 3+40 (trick_vec) +2 ([personal_bid, tricks_wincount]) + 10 +10 + 1 +4 +1+2
        if shared_bid_net is not None and shared_play_net is not None:
            self.bid_net = shared_bid_net
            self.play_net = shared_play_net
            # NEW: use shared trick predictor if provided, else create one.
            if shared_trick_win_predictor is not None:
                self.trick_win_predictor = shared_trick_win_predictor
            else:
                self.trick_win_predictor = nn.Linear(hidden_dim, 11)
        else:
            self.bid_net = PolicyNetwork(bid_input_dim, output_dim=11, hidden_dims=bid_hidden_dims)
            self.play_net = PolicyNetwork(play_input_dim, output_dim=hand_size, hidden_dims=play_hidden_dims)
            self.trick_win_predictor = nn.Linear(hidden_dim, 11)
        assert self.bid_net.model[0].in_features == bid_input_dim, f"Bid net input dimension mismatch: expected {bid_input_dim}, got {self.bid_net.model[0].in_features}"
        assert self.play_net.model[0].in_features == play_input_dim, f"Play net input dimension mismatch: expected {play_input_dim}, got {self.play_net.model[0].in_features}"
        hidden_dim = play_hidden_dims[-1] if play_hidden_dims is not None else 64
        self.optimizer_bid = optim.Adam(self.bid_net.parameters(), lr=learning_rate)
        # Note: In shared mode, each agent’s play optimizer will be built from the shared play_net plus the shared trick predictor.
        self.optimizer_play = optim.Adam(
            list(self.play_net.parameters()) + list(self.trick_win_predictor.parameters()),
            lr=learning_rate
        )
        self.trick_win_predictons = []

    def _process_bid_observation(self, observation):
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        hand_vec = process_hand(observation['hand'], self.hand_size, suit_mapping)
        round_number = observation.get('round_number', 0)
        bidding_phase = observation.get('bidding_phase', 1)
        player_id = observation.get('player_id', 0)
        position_in_order = observation.get('position_in_playing_order', 0)
        total_scores = observation.get('total_scores', [0] * 10)
        input_vec = hand_vec + total_scores + [position_in_order, player_id, round_number, bidding_phase]
        expected_dim = self.bid_net.model[0].in_features
        assert len(input_vec) == expected_dim, f"Bid input vector length mismatch: expected {expected_dim}, got {len(input_vec)}"
        return torch.tensor(input_vec, dtype=torch.float32)

    def _process_play_observation(self, observation):
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        legal_actions = observation.get("legal_actions", list(range(len(observation['hand']))))
        hand_vec = process_hand(observation['hand'], self.hand_size, suit_mapping,legal_actions)
        current_trick = observation.get('current_trick', [])
        trick_vec = []
        ranks = []
        for card in current_trick:
            # One-hot encoding for the suit
            one_hot = [0, 0, 0, 0]
            if card[0] in suit_mapping:
                one_hot[suit_mapping[card[0]]] = 1

            # Convert rank to float and store for statistics
            try:
                rank = float(card[1])
            except (ValueError, TypeError):
                rank = 0.0
            ranks.append(rank)

            # Append encoding for this card
            trick_vec.extend(one_hot + [rank])
        
        # Ensure fixed trick size by padding
        while len(trick_vec) < self.max_players * 5:
            trick_vec.extend([0] * 5)

        # Compute statistics: count, average rank, max rank
        count = len(ranks)
        avg_rank = np.mean(ranks) if ranks else 0.0
        max_rank = np.max(ranks) if ranks else 0.0

        # Append trick summary statistics
        trick_vec.extend([count, avg_rank, max_rank])
        personal_bid = observation.get('personal_bid', 0)
        tricks_wincount = observation.get('personnal_tricks_wincount', 0)
        all_bids = observation.get('all_bids', [0] * 10)
        all_tricks_won = observation.get('all_tricks_won', [0] * 10)
        player_id = observation.get('player_id', 0)
        winning_card = observation.get('winning_card', (-1, -1))
        win_one_hot = [0, 0, 0, 0]
        if winning_card[0] in suit_mapping:
            win_one_hot[suit_mapping[winning_card[0]]] = 1
        try:
            win_rank = float(winning_card[1])
        except (ValueError, TypeError):
            win_rank = 0.0
        current_winner = observation.get('current_winner', -1)
        position_in_order = observation.get('position_in_playing_order', 0)
        input_vec = (hand_vec + trick_vec + [personal_bid, tricks_wincount] +
                     all_bids + all_tricks_won + [player_id] + win_one_hot +
                     [win_rank] + [current_winner, position_in_order])
        expected_dim = self.play_net.model[0].in_features
        assert len(input_vec) == expected_dim, f"Play input vector length mismatch: expected {expected_dim}, got {len(input_vec)}"
        return torch.tensor(input_vec, dtype=torch.float32)

    def bid(self, observation):
        input_tensor = self._process_bid_observation(observation)
        logits = self.bid_net(input_tensor)
        effective_actions = int(observation['round_number']) + 1
        indices = torch.arange(logits.shape[0], device=logits.device)
        mask = torch.where(indices < effective_actions,
                           torch.zeros_like(logits),
                           torch.full_like(logits, -float('inf')))
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action_tensor = m.sample()  
        log_prob = m.log_prob(action_tensor)
        self.log_probs.append(log_prob)
        # NEW: Instead of storing the integer action, store the raw predicted tensor.
        self.bid_prediction = probs  
        return action_tensor.item()

    def play_card(self, observation):
        input_tensor = self._process_play_observation(observation)
        layers = list(self.play_net.model.children())
        hidden_model = torch.nn.Sequential(*layers[:-1])
        hidden = hidden_model(input_tensor)
        logits = layers[-1](hidden)
        trick_win_logits = self.trick_win_predictor(hidden)
        
        legal_actions = observation.get("legal_actions", list(range(len(observation['hand']))))
        indices = torch.arange(logits.shape[0], device=logits.device)
        legal_mask = torch.tensor([i in legal_actions for i in indices.tolist()], device=logits.device)
        mask = torch.where(legal_mask, torch.zeros_like(logits), torch.full_like(logits, -float('inf')))
        masked_logits = logits + mask

        indices_trick_win = torch.arange(trick_win_logits.shape[0], device=trick_win_logits.device)
        mask_trick_win = torch.where(
            indices_trick_win < int(observation['round_number']) + 1,
            torch.zeros_like(trick_win_logits),
            torch.full_like(trick_win_logits, -float('inf'))
        )
        trick_win_logits_masked = trick_win_logits + mask_trick_win
        # NEW: Instead of sampling, use the softmax directly as the prediction.
        trick_win_probs = F.softmax(trick_win_logits_masked, dim=-1)
        # Store these probabilities so that loss computed on them is differentiable.
        self.trick_win_predictons.append(trick_win_probs)
        
        # Continue with action selection for play:
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action_tensor = m.sample()
        log_prob = m.log_prob(action_tensor)
        self.log_probs.append(log_prob)
        return action_tensor.item()

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)