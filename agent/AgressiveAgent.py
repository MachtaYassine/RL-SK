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

        current_trick = observation.get('current_trick', [])
        # If leading, play the highest card.
        if len(current_trick) == 0:
            highest_index = max(range(len(hand)), key=lambda i: hand[i][1])
            return highest_index

        # Otherwise, try to win: play the highest card from those following lead or trump.
        lead_suit = current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        candidates = [(i, card) for i, card in enumerate(hand) if card[0] == lead_suit or card[0] == trump_suit]
        if candidates:
            # Choose highest card among candidates
            best = max(candidates, key=lambda x: x[1][1])
            return best[0]
        # Else play the highest card available.
        highest_index = max(range(len(hand)), key=lambda i: hand[i][1])
        return highest_index

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)