import gym
from gym import spaces
import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


class SkullKingEnvNoSpecials(gym.Env):
    def __init__(self, num_players=3):
        super(SkullKingEnvNoSpecials, self).__init__()
        self.num_players = num_players
        self.round_number = 1
        self.max_rounds = 10
        self.deck = self.create_deck()
        self.hands = [[] for _ in range(num_players)]
        self.bids = [None] * num_players
        self.tricks_won = [0] * num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0  # Track the current player
        self.total_scores = [0] * self.num_players  # NEW: track accumulated round rewards
        
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
        deck = [(suit, rank) for suit in suits for rank in range(1, 15)]
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
        self.current_player = 0
        self.total_scores = [0] * self.num_players  # NEW: reset total scores at game start
        obs = self._get_observation()
        logger.debug(f"Environment reset: round={self.round_number}, hand={self.hands[self.current_player]}, current_trick={self.current_trick}",
                     color="magenta")
        return obs

    def _get_observation(self):
        return {
            "hand": self.hands[self.current_player],  # changed from self.hands[0]
            "round_number": self.round_number,
            "bidding_phase": int(self.bidding_phase),
            "tricks_won": self.tricks_won[self.current_player],  # changed from self.tricks_won[0]
            "current_trick": self.current_trick,
            "total_scores": self.total_scores
        }

    def step(self, action):
        if self.bidding_phase:
            logger.debug(f"Bidding phase: player {self.current_player} bids {action}", color="green")
            self.bids[self.current_player] = action  # Current player places bid
            self.current_player = (self.current_player + 1) % self.num_players
            if all(b is not None for b in self.bids):  # If all players have bid
                self.bidding_phase = False
                self.action_space = spaces.Discrete(len(self.hands[0]))  # Now select card
                logger.debug("All bids received. Transitioning to play phase.", color="green")
            obs = self._get_observation()
            logger.debug(f"Observation after bidding: {obs}", color="magenta")
            return obs, 0, False, {}
        else:
            logger.debug(f"Play phase: player {self.current_player} action {action}", color="green")
            played_card = self.hands[self.current_player].pop(action)  # Play selected card
            self.current_trick.append((self.current_player, played_card))
            logger.debug(f"Updated current trick: {self.current_trick}", color="magenta")
            self.current_player = (self.current_player + 1) % self.num_players
            if len(self.current_trick) == self.num_players:  # If trick is complete
                winner = self.resolve_trick()
                self.tricks_won[winner] += 1
                logger.debug(f"Trick complete. Winner: player {winner} with trick: {self.current_trick}", color="green")
                self.current_trick = []
                # NEW: check if round is complete (all hands empty)
                if all(len(h) == 0 for h in self.hands):
                    round_rewards = self.calculate_reward()
                    # Update total scores
                    self.total_scores = [ts + r for ts, r in zip(self.total_scores, round_rewards)]
                    current_round = self.round_number  # capture current round before increment
                    if current_round < self.max_rounds:
                        logger.info(f"Round {current_round} complete: Round rewards: {round_rewards}, Total scores: {self.total_scores}")
                        obs = self.next_round()
                        return obs, round_rewards, False, {}
                    else:
                        logger.info(f"Final round {current_round} complete: Round rewards: {round_rewards}, Total scores: {self.total_scores}")
                        done = True
                        return self._get_observation(), round_rewards, done, {}
                rewards = self.calculate_reward()  # Calculate rewards after each trick
                logger.debug(f"Rewards: {rewards}, done: False", color="magenta")
                return self._get_observation(), rewards, False, {}
            obs = self._get_observation()
            logger.debug(f"Observation after play action: {obs}", color="yellow")
            return obs, [0] * self.num_players, False, {}

    def step_with_agent(self, agent):
        """
        Delegates action selection to the agent via its bid/play_card methods.
        """
        obs = self._get_observation()
        if self.bidding_phase and hasattr(agent, 'bid'):
            action = agent.bid(obs)
        elif not self.bidding_phase and hasattr(agent, 'play_card'):
            action = agent.play_card(obs)
        else:
            action = agent.act(obs, self.bidding_phase)
        return self.step(action)

    def resolve_trick(self):
        # Determine winner based on Skull King rules
        lead_suit = self.current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        winning_card = self.current_trick[0][1]
        winner = self.current_trick[0][0]
        for player, card in self.current_trick[1:]:
            if card[0] == trump_suit and winning_card[0] != trump_suit:
                winning_card = card
                winner = player
            elif card[0] == lead_suit and card[1] > winning_card[1]:
                winning_card = card
                winner = player
        return winner

    def calculate_reward(self):
        rewards = [0] * self.num_players
        for i in range(self.num_players):
            if self.bids[i] == self.tricks_won[i]:
                rewards[i] = 20 * self.bids[i]
            elif self.bids[i] == 0 and self.tricks_won[i] == 0:
                rewards[i] = 10 * self.round_number
            else:
                rewards[i] = -10 * abs(self.tricks_won[i] - self.bids[i])
        return rewards

    def next_round(self):
        self.round_number += 1
        self.deck = self.create_deck()
        self.deal_cards()
        self.bids = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0
        self.action_space = spaces.Discrete(self.round_number + 1)
        # Note: total_scores remains unchanged
        return self._get_observation()