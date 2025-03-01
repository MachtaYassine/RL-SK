import gym
import numpy as np
import random
from typing import List

class SkullKingEnvNoSpecials(gym.Env):
    def __init__(self, num_players=3, logger=None):
        super(SkullKingEnvNoSpecials, self).__init__()
        self.num_players = num_players
        self.max_train_players = num_players  # training player count may differ from max supported
        self.max_players = 5  # Fixed maximum number of players for observation dims
        self.round_number = 1
        self.max_rounds = 10
        self.deck = self.create_deck()
        self.hands = [[] for _ in range(num_players)]
        self.bids = [None] * self.max_players
        self.tricks_won = [0] * self.max_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = 0  # Track the current player
        self.total_scores = [0] * self.max_players
        self.current_winner = -1
        self.winning_card = (-1, -1)
        self.player_0_always_starts = True 
        self.leading_player_index = None
        self.leading_suit = None
        self.logger = logger
        self.suit_count = 4
        self.max_rank = 14
        self.hand_size = 10
        self.bid_dimension_vars = ["hand_size", "suit_count", "max_players"]
        self.play_dimension_vars = ["hand_size", "suit_count"]
        self.bid_feature_vars = ["total_scores", "position_in_order", "player_id", "round_number", "bidding_phase"]
        self.play_feature_vars = ["trick_vec", "personal_bid", "tricks_wincount", "all_bids", "all_tricks_won",
                                  "player_id", "win_one_hot", "win_rank", "current_winner", "position_in_order"]
        self.fixed_bid_features = self._calculate_feature_count(self.bid_feature_vars)
        self.fixed_play_features = self._calculate_feature_count(self.play_feature_vars)
        
        # Observation space
        max_hand_size = 10  # Maximum cards in a hand (round 10)
        
        
        # Action space

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
        self.logger.debug(f"Environment reset: round={self.round_number}, hand={self.hands[self.current_player]}, current_trick={self.current_trick}",
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

    def get_legal_actions(self, player_index: int) -> List[int]:
        """Return all legal actions (indices in hand) for the given player.
        If the trick has started, players must follow the leading suit if possible.
        """
        hand = self.hands[player_index]
        if self.current_trick:
            leading_suit = self.current_trick[0][1][0]
            highest_card_of_leading_suit = max(
                (card for _, card in self.current_trick if card[0] == leading_suit),
                key=lambda x: x[1]
            )
            #Card of the same suit and higher rank
            actions = [i for i, card in enumerate(hand) if card[0] == leading_suit and card[1] > highest_card_of_leading_suit[1]]
            if not actions:
                #Card of the same suit since no higher card
                actions = [i for i, card in enumerate(hand) if card[0] != leading_suit]
            if not actions:
                #Any card if no card of the same suit
                actions = list(range(len(hand)))
        else:
            actions = list(range(len(hand)))
        return actions

    def _get_observation(self):
        bids = [bid if bid is not None else 0 for bid in self.bids]
        if len(bids) < self.max_players:
            bids_padded = bids + [-1] * (self.max_players - len(bids))
        else:
            bids_padded = bids[:self.max_players]
        if len(self.tricks_won) < self.max_players:
            tricks_won_padded = self.tricks_won + [-1] * (self.max_players - len(self.tricks_won))
        else:
            tricks_won_padded = self.tricks_won[:self.max_players]
        if len(self.total_scores) < self.max_players:
            total_scores_padded = self.total_scores + [-100000] * (self.max_players - len(self.total_scores))
        else:
            total_scores_padded = self.total_scores[:self.max_players]
        obs = {
            "player_id": self.current_player,
            "hand": self.hands[self.current_player],
            "round_number": self.round_number,
            "bidding_phase": int(self.bidding_phase),
            "personnal_tricks_wincount": self.tricks_won[self.current_player],
            "personal_bid": self.bids[self.current_player] if self.bids[self.current_player] is not None else 0,
            "current_trick": self.current_trick,
            "all_bids": bids_padded,
            "all_tricks_won": tricks_won_padded,
            "total_scores": total_scores_padded,
            "current_winner": self.current_winner,
            "winning_card": self.winning_card,
            "position_in_playing_order": self.current_player if self.player_0_always_starts 
                                         else (self.current_player - self.round_number + 1) % self.num_players,
            "legal_actions": self.get_legal_actions(self.current_player),
        }
        return obs

    def _process_bidding_step(self, action):
        """
        Process the bidding action for the current player.
        
        Logs the bid, updates the current player's bid, and if all bids are received,
        transitions to the play phase by adjusting the action space.
        
        Returns:
            tuple: (updated observation, reward (0), done flag (False), empty info dict)
        """
        self.logger.debug(f"Bidding phase: player {self.current_player} bids {action}", color="green")
        self.bids[self.current_player] = action
        self.current_player = (self.current_player + 1) % self.num_players
        if all(b is not None for b in self.bids):
            self.bidding_phase = False
            # Now selecting a card, so change the action space accordingly.
            self.logger.debug("All bids received. Transitioning to play phase.", color="green")
        obs = self._get_observation()
        self.logger.debug(f"Observation after bidding: {obs}", color="magenta")
        return obs, 0, False, {}

    def _process_play_step(self, action):
        """
        Process the play action for the current player.
        
        Logs the play action, updates the trick with the played card, and if the trick is complete,
        resolves it (updating scores and handling round transitions). Returns the updated observation,
        round rewards if applicable, a done flag, and an info dict.
        
        Returns:
            tuple: (updated observation, round rewards (or [0]*num_players), done flag, info dict)
        """
        self.logger.debug(f"Play phase: player {self.current_player} action {action}", color="green")
        played_card = self.hands[self.current_player].pop(action)
        self.current_trick.append((self.current_player, played_card))
        self.logger.debug(f"Updated current trick: {self.current_trick}", color="magenta")

        # NEW: If this is the first card of the trick, set leader info.
        if len(self.current_trick) == 1:
            self.leading_player_index = self.current_player
            self.leading_suit = played_card[0]
            self.current_winner = self.current_player
            self.winning_card = played_card
        else:
            self._update_current_trick_winner(self.current_player, played_card)

        self.current_player = (self.current_player + 1) % self.num_players

        if len(self.current_trick) == self.num_players:
            # Store trick info before clearing
            trick_info = {"trick_cards": self.current_trick.copy()}
            winner = self.resolve_trick()
            trick_info["trick_winner"] = winner
            self.tricks_won[winner] += 1
            self.logger.debug(f"Trick complete. Winner: player {winner} with trick: {self.current_trick}", color="green")
            # NEW: Start next trick with the trick winner.
            self.current_player = winner
            self.current_trick = []
            self.current_winner = -1
            self.winning_card = (-1, -1)
            # Reset leader info after trick completion
            self.leading_player_index = None
            self.leading_suit = None
            if all(len(h) == 0 for h in self.hands): # if all the hands are empty !
                round_rewards = self.calculate_reward()
                self.total_scores = [ts + r for ts, r in zip(self.total_scores, round_rewards)]
                current_round = self.round_number  # capture current round before increment
                info = {"trick_info": trick_info, "round_bids": self.bids, "round_tricks_won": self.tricks_won}
                if current_round < self.max_rounds:
                    self.logger.info(f"Round {current_round} complete: Round rewards: {round_rewards}, Total scores: {self.total_scores}")
                    obs = self.next_round()
                    return obs, round_rewards, False, info
                else:
                    self.logger.info(f"Final round {current_round} complete: Round rewards: {round_rewards}, Total scores: {self.total_scores}")
                    done = True
                    return self._get_observation(), round_rewards, done, info
            # The hands are not empty, so continue playing.
            return self._get_observation(), [0] * self.num_players, False, {"trick_info": trick_info}
        # Trick not complete, continue playing.
        obs = self._get_observation()
        self.logger.debug(f"Observation after play action: {obs}", color="yellow")
        return obs, [0] * self.num_players, False, {}

    def step(self, action):
        """
        Unified step method that delegates to bidding or play phase methods.
        """
        return self._process_bidding_step(action) if self.bidding_phase else self._process_play_step(action)

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
        lead_suit = self.current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        winning_card = self.current_trick[0][1]
        winner = self.current_trick[0][0]
        for player, card in self.current_trick[1:]:
            # If the new card is trump.
            if card[0] == trump_suit:
                # If current winning card is not trump or is trump with lower rank.
                if winning_card[0] != trump_suit or (winning_card[0] == trump_suit and card[1] > winning_card[1]):
                    winning_card = card
                    winner = player
            # Else if no trump has been played, compare lead suit cards.
            elif winning_card[0] != trump_suit and card[0] == lead_suit and card[1] > winning_card[1]:
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
        # NEW: Store tricks won from this round before resetting.
        self.last_round_tricks = self.tricks_won.copy()
        self.deck = self.create_deck()
        self.deal_cards()
        self.bids = [None] * self.num_players   # reset bids between rounds
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.bidding_phase = True
        # NEW: Randomize the starting player for this round.
        self.current_player = random.randint(0, self.num_players - 1)
        self.logger.info(f"Next round {self.round_number} starts with player {self.current_player} as leader.", color="cyan")
        return self._get_observation()

    def _calculate_feature_count(self, feature_vars):
        count = 0
        if "total_scores" in feature_vars:
            count += self.max_players  # Padded length
        if "position_in_order" in feature_vars:
            count += 1
        if "player_id" in feature_vars:
            count += 1
        if "round_number" in feature_vars:
            count += 1
        if "bidding_phase" in feature_vars:
            count += 1
        if "trick_vec" in feature_vars:
            count += 3
        if "personal_bid" in feature_vars:
            count += 1
        if "tricks_wincount" in feature_vars:
            count += 1
        if "all_bids" in feature_vars:
            count += self.max_players  # Padded length
        if "all_tricks_won" in feature_vars:
            count += self.max_players  # Padded length
        if "win_one_hot" in feature_vars:
            count += 4
        if "win_rank" in feature_vars:
            count += 1
        if "current_winner" in feature_vars:
            count += 1
        return count

    def get_env_properties(self):
        return {
            "suit_count": self.suit_count,
            "max_rank": self.max_rank,
            "max_players": self.max_players,
            "hand_size": self.hand_size,
            "bid_dimension_vars": self.bid_dimension_vars,
            "play_dimension_vars": self.play_dimension_vars,
            "fixed_bid_features": self.fixed_bid_features,
            "fixed_play_features": self.fixed_play_features
        }