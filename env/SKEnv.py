import gym
from gym import spaces
import numpy as np
import random

class SkullKingEnv(gym.Env):
    def __init__(self, num_players=3):
        super(SkullKingEnv, self).__init__()
        self.num_players = num_players
        self.round_number = 1
        self.max_rounds = 10
        self.deck = self.create_deck()
        self.hands = [[] for _ in range(num_players)]
        self.bids = [None] * num_players
        self.tricks_won = [0] * num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0  # newly added for multi-agent symmetry
        
        # Observation space
        max_hand_size = 10  # Maximum cards in a hand (round 10)
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([4, 15] * max_hand_size),  # (suit, rank) pairs
            "round_number": spaces.Discrete(11),
            "bidding_phase": spaces.Discrete(2),
            "tricks_won": spaces.Discrete(11),
            "current_trick": spaces.MultiDiscrete([4, 15] * num_players)
        })
        
        # Action space
        self.action_space = spaces.Discrete(self.round_number + 1)  # Bidding action space

    def create_deck(self):
        suits = ["Parrot", "Treasure Chest", "Treasure Map", "Jolly Roger"]
        special_cards = ["Escape", "Pirate", "Mermaid", "Skull King", "Tigress", "Kraken", "White Whale", "Loot"]
        deck = [(suit, rank) for suit in suits for rank in range(1, 15)]
        deck += [(special, None) for special in special_cards for _ in range(2)]
        random.shuffle(deck)
        return deck

    def deal_cards(self):
        random.shuffle(self.deck)
        for i in range(self.num_players):
            self.hands[i] = [self.deck.pop() for _ in range(self.round_number)]

    def reset(self):
        self.round_number = 1
        self.deck = self.create_deck()
        self.deal_cards()
        self.bids = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0  # newly added for multi-agent symmetry
        return self._get_observation()

    def _get_observation(self):
        return {
            "hand": self.hands[self.current_player],  # changed from self.hands[0]
            "round_number": self.round_number,
            "bidding_phase": int(self.bidding_phase),
            "tricks_won": self.tricks_won[self.current_player],  # changed from self.tricks_won[0]
            "current_trick": self.current_trick
        }

    def step(self, action):
        if self.bidding_phase:
            self.bids[self.current_player] = action  # use current_player instead of fixed index
            self.current_player = (self.current_player + 1) % self.num_players
            if all(b is not None for b in self.bids):
                self.bidding_phase = False
                self.action_space = spaces.Discrete(len(self.hands[self.current_player]))
            return self._get_observation(), 0, False, {}
        else:
            played_card = self.hands[self.current_player].pop(action)
            self.current_trick.append(played_card)
            if len(self.current_trick) == self.num_players:
                winner = self.resolve_trick()
                self.tricks_won[winner] += 1
                self.current_trick = []
            done = self.round_number == self.max_rounds and all(len(h) == 0 for h in self.hands)
            reward = self.calculate_reward() if done else 0
            self.current_player = (self.current_player + 1) % self.num_players
            return self._get_observation(), reward, done, {}

    def resolve_trick(self):
        # Determine winner based on Skull King rules
        return random.randint(0, self.num_players - 1)  # Placeholder

    def calculate_reward(self):
        if self.bids[0] == self.tricks_won[0]:
            return 20 * self.bids[0]
        elif self.bids[0] == 0 and self.tricks_won[0] == 0:
            return 10 * self.round_number
        else:
            return -10 * abs(self.tricks_won[0] - self.bids[0])

