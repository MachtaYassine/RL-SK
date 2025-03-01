import random
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials

class SubEpisodeEnv(SkullKingEnvNoSpecials):
    def __init__(self, num_players, logger=None, training_mode=True):
        super().__init__(num_players=num_players, logger=logger)
        self.training_mode = training_mode
        self.last_bids = None

    def reset(self):
        self.hands = [[] for _ in range(self.num_players)]
        self.bids = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.total_scores = [0] * self.num_players

        self.deck = self.create_deck()
        max_possible_round = len(self.deck) // self.num_players
        self.round_number = random.randint(1, min(self.max_rounds, max_possible_round))
        self.deal_cards()
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = random.randint(0, self.num_players - 1)
        obs = self._get_observation()
        if self.logger:
            self.logger.debug(f"SubEpisodeEnv reset: round={self.round_number}, starting player={self.current_player}", color="magenta")
        return obs

    def next_round(self):
        self.round_number += 1
        self.last_round_tricks = self.tricks_won.copy()
        self.last_bids = self.bids.copy()
        self.deck = self.create_deck()
        max_possible_round = len(self.deck) // self.num_players
        self.round_number = random.randint(1, min(self.max_rounds, max_possible_round))
        self.hands = [[] for _ in range(self.num_players)]
        self.deal_cards()
        self.bids = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.bidding_phase = True
        self.current_player = random.randint(0, self.num_players - 1)
        self.logger.info(f"SubEpisodeEnv: Next round {self.round_number} starts with player {self.current_player}.", color="cyan")
        return self._get_observation()

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
                    obs = self.next_round() # Because the game never really ends, just sample a new round ! 
                    done=True
                    return self._get_observation(), round_rewards, done, info
            # The hands are not empty, so continue playing.
            return self._get_observation(), [0] * self.num_players, False, {"trick_info": trick_info}
        # Trick not complete, continue playing.
        obs = self._get_observation()
        self.logger.debug(f"Observation after play action: {obs}", color="yellow")
        return obs, [0] * self.num_players, False, {}