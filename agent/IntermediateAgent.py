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
        if not hand:
            return 0
        legal_actions = observation.get("legal_actions", list(range(len(hand))))
        current_trick = observation.get('current_trick', [])
        if len(current_trick) == 0:
            # Sort only legal indices by card rank and select the median move.
            sorted_indices = sorted(legal_actions, key=lambda i: hand[i][1])
            return sorted_indices[len(sorted_indices) // 2]
        lead_suit = current_trick[0][1][0]
        trump_suit = "Jolly Roger"
        winning_card = current_trick[0][1]
        for player, card in current_trick[1:]:
            if card[0] == trump_suit and winning_card[0] != trump_suit:
                winning_card = card
            elif card[0] == lead_suit and winning_card[0] == lead_suit and card[1] > winning_card[1]:
                winning_card = card
        candidates = [(i, card) for i, card in enumerate(hand) if i in legal_actions and card[0] == lead_suit]
        if not candidates:
            candidates = [(i, card) for i, card in enumerate(hand) if i in legal_actions and card[0] == trump_suit]
        winning_candidates = []
        for i, card in candidates:
            if winning_card[0] == trump_suit and card[0] == trump_suit and card[1] > winning_card[1]:
                winning_candidates.append((i, card))
            elif winning_card[0] != trump_suit and card[0] == lead_suit and card[1] > winning_card[1]:
                winning_candidates.append((i, card))
        if winning_candidates:
            best = min(winning_candidates, key=lambda x: x[1][1])
            return best[0]
        # Discard lowest card among legal moves.
        discard_index = min(legal_actions, key=lambda i: hand[i][1])
        return discard_index

    def act(self, observation, bidding_phase):
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)