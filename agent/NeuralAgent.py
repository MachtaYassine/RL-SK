import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .SimpleAgent import SkullKingAgent

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
    def __init__(self, num_players, hand_size=10, learning_rate=1e-3):
        super().__init__(num_players)
        self.hand_size = hand_size  # maximum expected hand size (for playing)
        self.log_probs = []  # new: buffer for log probabilities
        # For bidding, maximum valid actions are 0..10 (for round 10) i.e. 11 actions.
        self.bid_net = PolicyNetwork(input_dim=self._bid_input_dim(), output_dim=11)
        self.play_net = PolicyNetwork(input_dim=self._play_input_dim(), output_dim=hand_size)
        self.optimizer_bid = optim.Adam(self.bid_net.parameters(), lr=learning_rate)
        self.optimizer_play = optim.Adam(self.play_net.parameters(), lr=learning_rate)

    # Each card now is encoded with 4 numbers (one-hot suit) and 1 for rank.
    def _bid_input_dim(self):
        # New dimension:
        # hand encoding: hand_size * 5
        # round_number, tricks_won, personal_bid: 3
        # all_bids: num_players
        # player_id: 1
        # winning_card: 4 (one-hot) + 1 (rank) = 5
        # current_winner: 1
        return self.hand_size * 5 + 3 + self.num_players + 1 + 5 + 1  # = self.hand_size*5 + num_players + 10

    def _play_input_dim(self):
        # New dimension:
        # hand encoding: hand_size * 5
        # current trick encoding: num_players * 5
        # personal_bid, tricks_won: 2
        # player_id: 1
        # winning_card: 5
        # current_winner: 1
        return self.hand_size * 5 + self.num_players * 5 + 2 + 1 + 5 + 1  # = self.hand_size*5 + self.num_players*5 + 10

    def _process_bid_observation(self, observation):
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        hand = observation['hand']
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
        while len(hand_vec) < self.hand_size * 5:
            hand_vec.extend([0] * 5)
        round_number = observation['round_number']
        tricks_won = observation['tricks_won']
        personal_bid = observation.get('personal_bid', 0)
        all_bids = observation.get('all_bids', [0] * self.num_players)
        player_id = observation.get('player_id', 0)
        winning_card = observation['winning_card']  # Directly use winning_card
        win_one_hot = [0, 0, 0, 0]
        if winning_card[0] in suit_mapping:
            win_one_hot[suit_mapping[winning_card[0]]] = 1
        try:
            win_rank = float(winning_card[1])
        except (ValueError, TypeError):
            win_rank = 0.0
        current_winner = observation['current_winner']
        input_vec = hand_vec + [round_number, tricks_won, personal_bid] + all_bids + [player_id] + win_one_hot + [win_rank] + [current_winner]
        # print(input_vec)
        return torch.tensor(input_vec, dtype=torch.float32)

    def _process_play_observation(self, observation):
        suit_mapping = {"Parrot": 0, "Treasure Chest": 1, "Treasure Map": 2, "Jolly Roger": 3}
        hand = observation['hand']
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
        while len(hand_vec) < self.hand_size * 5:
            hand_vec.extend([0] * 5)
        current_trick = observation.get('current_trick', [])
        trick_vec = []
        for play in current_trick:
            # Assuming play is a tuple: (player_id, card)
            _, card = play
            one_hot = [0, 0, 0, 0]
            if card[0] in suit_mapping:
                one_hot[suit_mapping[card[0]]] = 1
            try:
                rank = float(card[1])
            except (ValueError, TypeError):
                rank = 0.0
            trick_vec.extend(one_hot + [rank])
        while len(trick_vec) < self.num_players * 5:
            trick_vec.extend([0] * 5)
        personal_bid = observation.get('personal_bid', 0)
        tricks_won = observation.get('tricks_won', 0)
        player_id = observation.get('player_id', 0)
        winning_card = observation['winning_card']  # Directly use winning_card
        win_one_hot = [0, 0, 0, 0]
        if winning_card[0] in suit_mapping:
            win_one_hot[suit_mapping[winning_card[0]]] = 1
        try:
            win_rank = float(winning_card[1])
        except (ValueError, TypeError):
            win_rank = 0.0
        current_winner = observation['current_winner']
        input_vec = hand_vec + trick_vec + [personal_bid, tricks_won, player_id] + win_one_hot + [win_rank] + [current_winner]
        # print(input_vec)
        return torch.tensor(input_vec, dtype=torch.float32)

    def bid(self, observation):
        input_tensor = self._process_bid_observation(observation)
        logits = self.bid_net(input_tensor)
        # Create a mask so that only bets 0..current_round are allowed.
        effective_actions = int(observation['round_number']) + 1  # valid bids: 0 to round_number inclusive
        mask = torch.full_like(logits, 0.0)
        if effective_actions < logits.shape[0]:
            mask[effective_actions:] = -float('inf')
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        log_prob = m.log_prob(torch.tensor(action))
        self.log_probs.append(log_prob)  # record log probability
        return action

    def play_card(self, observation):
        input_tensor = self._process_play_observation(observation)
        logits = self.play_net(input_tensor)
        # Determine valid actions based on current hand size (before padding).
        effective_actions = len(observation['hand'])
        mask = torch.full_like(logits, 0.0)
        if effective_actions < logits.shape[0]:
            mask[effective_actions:] = -float('inf')
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        log_prob = m.log_prob(torch.tensor(action))
        self.log_probs.append(log_prob)  # record log probability
        return action

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)