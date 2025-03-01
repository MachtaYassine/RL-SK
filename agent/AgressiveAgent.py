from .SimpleAgent import SkullKingAgent

class AggressiveSkullKingAgent(SkullKingAgent):
    """
    Tends to bid high if holding many strong cards and plays highest cards
    to aggressively win tricks.
    """
    def bid(self, observation):
        hand = observation['hand']
        # Count very strong cards (rank >= 12) as aggressive indicator.
        count_strong = sum(1 for card in hand if card[1] >= 12)
        # Aggressive bid: try to win maximum possible tricks if hand is strong.
        bid = min(len(hand), count_strong + 1)
        return bid

    def play_card(self, observation):
        hand = observation['hand']
        if not hand:
            return 0
        legal_actions = observation.get("legal_actions", list(range(len(hand))))
        current_trick = observation.get('current_trick', [])
        if len(current_trick) == 0:
            # Choose highest card among legal moves.
            highest_index = max(legal_actions, key=lambda i: hand[i][1])
            return highest_index
        lead_suit = current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        candidates = [(i, card) for i, card in enumerate(hand) if i in legal_actions and (card[0] == lead_suit or card[0] == trump_suit)]
        if candidates:
            best = max(candidates, key=lambda x: x[1][1])
            return best[0]
        # Fallback: play highest card among legal moves.
        highest_index = max(legal_actions, key=lambda i: hand[i][1])
        return highest_index

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)