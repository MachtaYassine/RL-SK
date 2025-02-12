from .SimpleAgent import SkullKingAgent


class IntermediateSkullKingAgent(SkullKingAgent):
    """
    Uses hand heuristics to bid based on the number of high cards
    and plays cards based on the current trick state.
    """
    def __init__(self, num_players, high_card_threshold=10):
        super().__init__(num_players)
        self.high_card_threshold = high_card_threshold

    def bid(self, observation):
        # Bid as the count of cards exceeding the threshold rank.
        hand = observation['hand']
        count_high_cards = sum(1 for card in hand if card[1] >= self.high_card_threshold)
        # Sometimes adjust the bid conservatively in lower rounds.
        bid = min(len(hand), count_high_cards)
        return bid

    def play_card(self, observation):
        hand = observation['hand']
        current_trick = observation.get('current_trick', [])
        if not hand:
            return 0

        # If leading the trick, play a mid-range card. 
        if len(current_trick) == 0:
            # Sort hand by rank and select the median card.
            sorted_indices = sorted(range(len(hand)), key=lambda i: hand[i][1])
            median_index = sorted_indices[len(sorted_indices)//2]
            return median_index

        # Else, try to beat the current winning card with the smallest winning card.
        lead_suit = current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        # Determine current winning rank & suit.
        winning_card = current_trick[0][1]
        for player, card in current_trick[1:]:
            if card[0] == trump_suit and winning_card[0] != trump_suit:
                winning_card = card
            elif card[0] == lead_suit and winning_card[0] == lead_suit and card[1] > winning_card[1]:
                winning_card = card

        # Find candidate cards from hand that follow suit or trump if needed.
        candidates = [(i, card) for i, card in enumerate(hand) if card[0] == lead_suit]
        # If player has no cards that follow lead, check for trump.
        if not candidates:
            candidates = [(i, card) for i, card in enumerate(hand) if card[0] == trump_suit]
        # If any candidate can beat current winning card, select the smallest winning card.
        winning_candidates = []
        for i, card in candidates:
            # If trick is trumped, only trump cards can win.
            if winning_card[0] == trump_suit and card[0] == trump_suit and card[1] > winning_card[1]:
                winning_candidates.append((i, card))
            # Otherwise, follow suit.
            elif winning_card[0] != trump_suit and card[0] == lead_suit and card[1] > winning_card[1]:
                winning_candidates.append((i, card))
        if winning_candidates:
            # Select candidate with smallest rank that still wins
            best = min(winning_candidates, key=lambda x: x[1][1])
            return best[0]
        # Otherwise, discard lowest card.
        discard_index = min(range(len(hand)), key=lambda i: hand[i][1])
        return discard_index

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)