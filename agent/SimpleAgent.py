import numpy as np
import random

class SkullKingAgent:
    def __init__(self, num_players):
        self.num_players = num_players
        self.bids = [0] * num_players  # Initialize bids for each player

    def bid(self, observation):
        # Simple bidding strategy: bid half of the cards in hand
        hand_size = len(observation['hand'])
        return max(0, min(hand_size, hand_size // 2))

    def play_card(self, observation):
        hand = observation['hand']
        if hand:
            legal_actions = observation.get("legal_actions", list(range(len(hand))))
            return random.choice(legal_actions)  # use legal moves
        return 0  # Default action if no cards are available

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)