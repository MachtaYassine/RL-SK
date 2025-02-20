from gym import spaces
import random
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials

class SubEpisodeEnv(SkullKingEnvNoSpecials):
    def reset(self):
        # Update lists based on current num_players.
        self.hands = [[] for _ in range(self.num_players)]
        self.bids = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.total_scores = [0] * self.num_players

        self.deck = self.create_deck()
        # Constrain round_number so there are enough cards.
        max_possible_round = len(self.deck) // self.num_players
        self.round_number = random.randint(1, min(self.max_rounds, max_possible_round))
        # Deal cards using the randomized round_number.
        self.deal_cards()
        self.current_trick = []
        self.bidding_phase = True
        # Randomize the starting player.
        self.current_player = random.randint(0, self.num_players - 1)
        obs = self._get_observation()
        if self.logger:
            self.logger.debug(f"SubEpisodeEnv reset: round={self.round_number}, starting player={self.current_player}", color="magenta")
        return obs

    def next_round(self):
        self.last_bids = self.bids.copy()
        self.last_round_tricks = self.tricks_won.copy()
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
        self.action_space = spaces.Discrete(self.round_number + 1)
        # Removed verbose detail:
        self.logger.info(f"SubEpisodeEnv: Next round {self.round_number} starts with player {self.current_player}.", color="cyan")
        return self._get_observation()
