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
        print("Your hand:")
        for i, card in enumerate(hand):
            print(f"  {i}: {card}")
        while True:
            try:
                card_index = int(input(f"Enter card index to play (integer between 0 and {len(hand)-1}): "))
                if 0 <= card_index < len(hand):
                    return card_index
                else:
                    print(f"Invalid card index! Please enter a number between 0 and {len(hand)-1}.")
            except ValueError:
                print("Invalid input! Please enter an integer.")

    def act(self, observation, bidding_phase):
        return self.bid(observation) if bidding_phase else self.play_card(observation)
