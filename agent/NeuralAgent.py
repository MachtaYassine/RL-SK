import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .SimpleAgent import SkullKingAgent

# Helper: process hand information
def process_hand(hand, hand_size, suit_mapping):
    hand_vec = []
    for card in hand:
        one_hot = [0, 0, 0, 0]
        if card[0] in suit_mapping:
            one_hot[suit_mapping[card[0]]] = 1
        try:
            rank = float(card[1])
        except (ValueError, TypeError):
            rank = 0.0
        hand_vec.extend(one_hot + [rank])
    while len(hand_vec) < hand_size * 5:
        hand_vec.extend([0] * 5)
    return hand_vec

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LearningSkullKingAgent(SkullKingAgent):
    def __init__(self, num_players, hand_size=10, learning_rate=1e-3, shared_bid_net=None, shared_play_net=None,
                 suit_count=4, max_rank=14, max_players=10, bid_dimension_vars=None, play_dimension_vars=None,
                 fixed_bid_features=4, fixed_play_features=33):
        super().__init__(num_players)
        self.hand_size = hand_size
        self.suit_count = suit_count
        self.max_rank = max_rank
        self.max_players = max_players
        self.bid_dimension_vars = bid_dimension_vars or []
        self.play_dimension_vars = play_dimension_vars or []
        self.fixed_bid_features = fixed_bid_features
        self.fixed_play_features = fixed_play_features
        self.log_probs = []
        if shared_bid_net is not None and shared_play_net is not None:
            self.bid_net = shared_bid_net
            self.play_net = shared_play_net
        else:
            self.bid_net = PolicyNetwork(self._bid_input_dim(), output_dim=11)
            self.play_net = PolicyNetwork(self._play_input_dim(), output_dim=hand_size)
        self.trick_win_predictor = nn.Linear(64, 11)
        self.optimizer_bid = optim.Adam(self.bid_net.parameters(), lr=learning_rate)
        self.optimizer_play = optim.Adam(
            list(self.play_net.parameters()) + list(self.trick_win_predictor.parameters()),
            lr=learning_rate
        )
        self.trick_win_predictons = []

    def _bid_input_dim(self):
        dim = 0
        dim += self.hand_size * self.suit_count
        dim += self.max_players + self.fixed_bid_features
        return dim

    def _play_input_dim(self):
        dim = 0
        dim += self.hand_size * self.suit_count
        dim += self.fixed_play_features
        return dim

    def _process_bid_observation(self, observation):
        # Process hand as before.
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        hand_vec = process_hand(observation['hand'], self.hand_size, suit_mapping)
        # Extract required scalars from observation.
        round_number = observation.get('round_number', 0)
        bidding_phase = observation.get('bidding_phase', 1)
        player_id = observation.get('player_id', 0)
        position_in_order = observation.get('position_in_playing_order', 0)
        # total_scores is expected to be padded to fixed MAX_PLAYERS (10)
        total_scores = observation.get('total_scores', [0] * 10)
        input_vec = hand_vec + total_scores + [position_in_order, player_id, round_number, bidding_phase]
        return torch.tensor(input_vec, dtype=torch.float32)

    def _process_play_observation(self, observation):
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        hand_vec = process_hand(observation['hand'], self.hand_size, suit_mapping)
        current_trick = observation.get('current_trick', [])
        trick_vec = []
        if current_trick:
            ranks = [float(card[1]) for _, card in current_trick if isinstance(card[1], (int, float))]
            count = len(ranks)
            avg_rank = np.mean(ranks) if ranks else 0.0
            max_rank = np.max(ranks) if ranks else 0.0
        else:
            count, avg_rank, max_rank = 0, 0.0, 0.0
        trick_vec.extend([count, avg_rank, max_rank])
        while len(trick_vec) < 3:
            trick_vec.extend([0] * 3)
        personal_bid = observation.get('personal_bid', 0)
        tricks_wincount = observation['personnal_tricks_wincount']
        all_bids = observation.get('all_bids', [0] * 10)
        all_tricks_won = observation.get('all_tricks_won', [0] * 10)
        player_id = observation.get('player_id', 0)
        winning_card = observation['winning_card']
        win_one_hot = [0, 0, 0, 0]
        if winning_card[0] in suit_mapping:
            win_one_hot[suit_mapping[winning_card[0]]] = 1
        try:
            win_rank = float(winning_card[1])
        except (ValueError, TypeError):
            win_rank = 0.0
        current_winner = observation['current_winner']
        position_in_order = observation.get('position_in_playing_order', 0)
        input_vec = (hand_vec + trick_vec + [personal_bid, tricks_wincount] + all_bids + all_tricks_won +
                     [player_id] + win_one_hot + [win_rank] + [current_winner, position_in_order])
        # print(input_vec)
        return torch.tensor(input_vec, dtype=torch.float32)

    def bid(self, observation):
        input_tensor = self._process_bid_observation(observation)
        logits = self.bid_net(input_tensor)
        effective_actions = int(observation['round_number']) + 1  # valid bids: 0 to round_number inclusive
        indices = torch.arange(logits.shape[0], device=logits.device)
        # NEW: Build mask without in-place modifications.
        mask = torch.where(
            indices < effective_actions,
            torch.zeros_like(logits),
            torch.full_like(logits, -float('inf'))
        )
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        log_prob = m.log_prob(torch.tensor(action))
        self.log_probs.append(log_prob)
        return action

    def play_card(self, observation):
        input_tensor = self._process_play_observation(observation)
        # Use play_net’s hidden layers to derive both logits and round score prediction.
        x = F.relu(self.play_net.fc1(input_tensor))
        hidden = F.relu(self.play_net.fc2(x))
        logits = self.play_net.fc3(hidden)
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
        trick_win_logits_masked_according_to_round = trick_win_logits+ mask_trick_win
        trick_win_probs = F.softmax(trick_win_logits_masked_according_to_round, dim=-1)
        d = torch.distributions.Categorical(trick_win_probs)
        predicted_number_of_tricks_won = d.sample().item()


        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        log_prob = m.log_prob(torch.tensor(action))
        self.log_probs.append(log_prob)
        self.trick_win_predictons.append(predicted_number_of_tricks_won)
        return action

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)