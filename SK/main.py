import random
from collections import defaultdict

# Constants for the deck
SUITS = ["Parrot", "Treasure Chest", "Pirate Map", "Jolly Roger"]  # Jolly Roger is trump

SPECIAL_CARDS = {
    "Tigress": 1,
    "Skull King": 1,
    "Mermaid": 2,
    "Escape": 5,
    "Loot": 2,
    "Kraken": 1,
    "White Whale": 1,
}

# Add advanced pirate abilities
PIRATE_ABILITIES = {
    "Rosie D’ Laney": "Choose a player to lead the next trick.",
    "Will the Bandit": "Draw 2 cards from the deck and discard 2.",
    "Rascal of Roatan": "Bet 0, 10, or 20 points. Earn/lose based on bid success.",
    "Juanita Jade": "Privately look through any undealt cards for the round.",
    "Harry the Giant": "Change your bid by +/-1 or leave it the same.",
}

suit_mapper = {"P": "Parrot", "T": "Treasure Chest", "M": "Pirate Map", "J": "Jolly Roger", "Tig": "Tigress", "SK": "Skull King", 
               "Mer": "Mermaid", "Esc": "Escape", "L": "Loot", "K": "Kraken", "WW": "White Whale", "RD": "Rosie D’ Laney", 
               "BB": "Will the Bandit", "RR": "Rascal of Roatan", "JJ": "Juanita Jade", "HG": "Harry the Giant"}

# Combine pirates with special cards
SPECIAL_CARDS.update({pirate: 1 for pirate in PIRATE_ABILITIES.keys()})

NUMERIC_CARDS = {suit: list(range(1, 15)) for suit in SUITS}  

# Function to create a deck
def create_deck():
    deck = []
    # Add numeric cards
    for suit, numbers in NUMERIC_CARDS.items():
        for number in numbers:
            deck.append((suit, number))
    # Add special cards
    for card, count in SPECIAL_CARDS.items():
        deck.extend([(card, None)] * count)
    return deck

class SkullKing:
    def __init__(self, num_players):
        self.num_players = num_players
        self.deck = create_deck()
        self.players = [f"Player {i+1}" for i in range(num_players)]
        self.hands = defaultdict(list)
        self.scores = {player: 0 for player in self.players}
        self.current_trick = []
        self.round_number = 1
        self.undealt_cards = []  # To track cards not dealt in the round

    def shuffle_and_deal(self):
        random.shuffle(self.deck)
        num_cards = self.round_number
        self.undealt_cards = self.deck[:]
        for player in self.players:
            self.hands[player] = [self.deck.pop() for _ in range(num_cards)]
        self.undealt_cards = self.deck[:]

    def play_trick(self):
        for player in self.players:
            print(f"Current trick: {self.current_trick}")
            card = self.play_card(player)
            self.current_trick.append((player, card))
        self.resolve_trick()

    def play_card(self, player):
        print(f"{player}'s hand: {self.hands[player]}")
        
        card = input(f"{player}, enter the card you want to play: ")
        #transform input to tuple
        suit, value = card.split(',')
        card = (suit_mapper[suit], int(value) if value.isdigit() else None)
        while card not in self.hands[player]:
            card = input(f"{player}, you don't have that card. Enter a valid card: ")
            suit, value = card.split(',')
            card = (suit_mapper[suit], int(value) if value.isdigit() else None)
        self.hands[player].remove(card)
        
        return card

    def resolve_trick(self):
        leading_suit = None
        highest_card = None
        winner = None

        for player, card in self.current_trick:
            suit, value = card
            if leading_suit is None and suit in NUMERIC_CARDS:
                leading_suit = suit  # Set leading suit

            if self.is_higher_card(leading_suit, card, highest_card):
                highest_card = card
                winner = player

        print(f"{winner} wins the trick with {highest_card}!")
        self.trick_win_count[self.players.index(winner)] += 1
        self.current_trick = []

    def is_higher_card(self, leading_suit, card, current_high):
        if current_high is None:
            return True

        suit, value = card
        current_suit, current_value = current_high

        # Special cards (Pirates, Skull King, etc.)
        if suit in SPECIAL_CARDS:
            return True  # Simplified, add detailed rules here

        # Check suits and trump
        if suit == "Jolly Roger" and current_suit != "Jolly Roger":
            return True  # Trump beats other suits
        if suit == leading_suit and value > current_value:
            return True
        return False

    def apply_ability(self, player, card):
        suit, value = card
        if suit in PIRATE_ABILITIES:
            print(f"{player} activates the ability of {suit}: {PIRATE_ABILITIES[suit]}")
            if suit == "Rosie D’ Laney":
                self.choose_next_leader(player)
            elif suit == "Will the Bandit":
                self.draw_and_discard(player)
            elif suit == "Rascal of Roatan":
                self.rascal_bet(player)
            elif suit == "Juanita Jade":
                self.reveal_undealt_cards(player)
            elif suit == "Harry the Giant":
                self.change_bid(player)

    def choose_next_leader(self, player):
        print(f"{player} chooses the next leader.")

    def draw_and_discard(self, player):
        print(f"{player} draws 2 cards from the deck.")
        for _ in range(2):
            if self.deck:
                self.hands[player].append(self.deck.pop())
        print(f"{player}'s new hand: {self.hands[player]}")
        # Placeholder for discard logic
        

    def rascal_bet(self, player):
        print(f"{player} places a special bet (0, 10, or 20 points).")

    def reveal_undealt_cards(self, player):
        print(f"{player} looks at the undealt cards: {self.undealt_cards}")

    def change_bid(self, player):
        print(f"{player} adjusts their bid.")

    def score_round(self,round_number):
        for i, player in enumerate(self.players):
            if self.n_tricks[i] == 0:
                if self.trick_win_count[i] == 0:
                    self.scores[player] += 10 * round_number
                else:
                    self.scores[player] -= 10 * round_number
            else:
                if self.n_tricks[i] == self.trick_win_count[i]:
                    self.scores[player] += 20 + 10 * self.n_tricks[i]
                else:
                    self.scores[player] -= 10 * abs(self.n_tricks[i] - self.trick_win_count[i])
        print(f"Scores after round {round_number}:")
        for player, score in self.scores.items():
            print(f"    {player}: {score}")

    def play_round(self):
        self.shuffle_and_deal()
        self.bet_on_tricks()
        self.trick_win_count=[0]*self.num_players
        for _ in range(self.round_number):  # Number of tricks equals cards dealt
            self.play_trick()
        self.score_round(self.round_number)
        self.round_number += 1

    def play_game(self):
        for _ in range(10):  # 10 rounds
            print(f"Starting round {self.round_number}")
            self.play_round()
        print("Game over! Final scores:")
        print(self.scores)
        
    def bet_on_tricks(self):
        self.n_tricks = []
        for player in self.players:
            print(f"{player} is betting on the number of tricks they will win.")
            print(f"{player}'s hand: {self.hands[player]}")
            n_tricks = int(input("Enter your bet: "))
            self.n_tricks.append(n_tricks)

import argparse
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--players", type=int, default=4, help="Number of players")
    args = parser.parse_args()
    
    
    game = SkullKing(args.players)
    game.play_game()
