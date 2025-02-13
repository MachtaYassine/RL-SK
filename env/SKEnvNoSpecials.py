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
        self.total_scores = [0] * self.num_players 
        self.current_winner = -1
        self.winning_card = (-1, -1)
        self.player_0_always_starts = True 
        
        # Observation space
        max_hand_size = 10  # Maximum cards in a hand (round 10)
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([4, 15] * max_hand_size),  # (suit, rank) pairs
            "round_number": spaces.Discrete(11),
            "bidding_phase": spaces.Discrete(2),
            "tricks_won": spaces.Discrete(11),
            "current_trick": spaces.MultiDiscrete([4, 15] * num_players),
            "all_bids": spaces.MultiDiscrete([11] * num_players),  # bids from 0 to 10
            "personal_bid": spaces.Discrete(11),
            "player_id": spaces.Discrete(num_players),
            "winning_card": spaces.MultiDiscrete([4, 15]),  # (suit, rank) pair
            "current_winner": spaces.Discrete(num_players),
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
    
    def _update_current_trick_winner(self, latest_player, latest_card):
        """
        Update the current trick winner given the latest played card.
        """
        trump_suit = "Jolly Roger"
        # If winning_card is still the default, update it.
        if self.winning_card == (-1, -1):
            self.current_winner = latest_player
            self.winning_card = latest_card
            return

        lead_suit = self.current_trick[0][1][0]
        # Update winner if latest card is trump and the current winning card isn't trump.
        if latest_card[0] == trump_suit and self.winning_card[0] != trump_suit:
            self.current_winner = latest_player
            self.winning_card = latest_card
        # Else if both cards are of lead suit and the latest card has a higher rank.
        elif latest_card[0] == lead_suit and self.winning_card[0] == lead_suit and latest_card[1] > self.winning_card[1]:
            self.current_winner = latest_player
            self.winning_card = latest_card
        # Otherwise, leave the winner unchanged.

    def _get_observation(self):
        obs = {
            "player_id": self.current_player,
            "hand": self.hands[self.current_player],
            "round_number": self.round_number,
            "bidding_phase": int(self.bidding_phase),
            "tricks_won": self.tricks_won[self.current_player],
            "personal_bid": self.bids[self.current_player] if self.bids[self.current_player] is not None else 0,
            "current_trick": self.current_trick,
            "all_bids": [bid if bid is not None else 0 for bid in self.bids],
            "current_winner": self.current_winner,
            "winning_card": self.winning_card
        }
        return obs

    def _process_bidding_step(self, action):
        logger.debug(f"Bidding phase: player {self.current_player} bids {action}", color="green")
        self.bids[self.current_player] = action
        self.current_player = (self.current_player + 1) % self.num_players
        if all(b is not None for b in self.bids):
            self.bidding_phase = False
            # Now selecting a card, so change the action space accordingly.
            self.action_space = spaces.Discrete(len(self.hands[0]))
            logger.debug("All bids received. Transitioning to play phase.", color="green")
        obs = self._get_observation()
        logger.debug(f"Observation after bidding: {obs}", color="magenta")
        return obs, 0, False, {}

    def _process_play_step(self, action):
        logger.debug(f"Play phase: player {self.current_player} action {action}", color="green")
        played_card = self.hands[self.current_player].pop(action)
        self.current_trick.append((self.current_player, played_card))
        logger.debug(f"Updated current trick: {self.current_trick}", color="magenta")

        # Update winning info.
        if len(self.current_trick) == 1:
            self.current_winner = self.current_player
            self.winning_card = played_card
        else:
            self._update_current_trick_winner(self.current_player, played_card)

        self.current_player = (self.current_player + 1) % self.num_players

        if len(self.current_trick) == self.num_players:
            winner = self.resolve_trick()
            self.tricks_won[winner] += 1
            logger.debug(f"Trick complete. Winner: player {winner} with trick: {self.current_trick}", color="green")
            self.current_trick = []
            self.current_winner = -1
            self.winning_card = (-1, -1)
            if all(len(h) == 0 for h in self.hands): # if all the hands are empty !
                round_rewards = self.calculate_reward()
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
            # The hands are not empty, so continue playing.
            rewards = self.calculate_reward()
            logger.debug(f"Rewards: {rewards}, done: False", color="magenta")
            return self._get_observation(), rewards, False, {}
        # Trick not complete, continue playing.
        obs = self._get_observation()
        logger.debug(f"Observation after play action: {obs}", color="yellow")
        return obs, [0] * self.num_players, False, {}

    def step(self, action):
        """
        Unified step method that delegates to bidding or play phase methods.
        """
        if self.bidding_phase:
            return self._process_bidding_step(action)
        else:
            return self._process_play_step(action)

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
            if self.bids[i] > 0:
                if self.bids[i] == self.tricks_won[i]:
                    rewards[i] = 20 * self.bids[i]
                else:
                    rewards[i] = -10 * abs(self.tricks_won[i] - self.bids[i])
            elif self.bids[i] == 0 :
                if self.tricks_won[i] == 0:
                    rewards[i] = 10 * self.round_number
                else:
                    rewards[i] = -10 * self.round_number
        return rewards

    def next_round(self):
        self.round_number += 1
        self.deck = self.create_deck()
        self.deal_cards()
        self.bids = [None] * self.num_players   # reset bids between rounds
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0 + (self.round_number-1) % self.num_players if not self.player_0_always_starts else 0
        self.action_space = spaces.Discrete(self.round_number + 1)
        return self._get_observation()