class HumanAgent:
    def __init__(self, num_players):
        self.num_players = num_players

    def bid(self, observation):
        hand = observation['hand']
        print("Your hand:")
        for i, card in enumerate(hand):
            print(f"  {i}: {card}")
        max_bid = len(hand)
        while True:
            try:
                bid = int(input(f"Enter your bid (integer between 0 and {max_bid}): "))
                if 0 <= bid <= max_bid:
                    return bid
                else:
                    print(f"Invalid bid! Please enter a number between 0 and {max_bid}.")
            except ValueError:
                print("Invalid input! Please enter an integer.")

    def play_card(self, observation):
        hand = observation['hand']
        legal_actions = observation.get("legal_actions", list(range(len(hand))))
        print("Your hand:")
        for i, card in enumerate(hand):
            indicator = " (LEGAL)" if i in legal_actions else ""
            print(f"  {i}: {card}{indicator}")
        while True:
            try:
                card_index = int(input(f"Enter card index to play from legal actions {legal_actions}: "))
                if card_index in legal_actions:
                    return card_index
                else:
                    print(f"Invalid card index! Please enter one of {legal_actions}.")
            except ValueError:
                print("Invalid input! Please enter an integer.")

    def act(self, observation, bidding_phase):
        return self.bid(observation) if bidding_phase else self.play_card(observation)
