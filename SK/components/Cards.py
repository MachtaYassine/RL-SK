import random
from collections import defaultdict

# Constants
SUITS = ["Parrot", "Treasure Chest", "Pirate Map", "Jolly Roger"]  # Jolly Roger is trump
SPECIAL_CARDS = {
    "Tigress": 1, "Skull King": 1, "Mermaid": 2, "Escape": 5, "Loot": 2,
    "Kraken": 1, "White Whale": 1
}
PIRATE_ABILITIES = {
    "Tigress": "Choose to play as a Pirate or as an Escape card.",
    "Rosie D’ Laney": "Choose a player to lead the next trick.",
    "Will the Bandit": "Draw 2 cards from the deck and discard 2.",
    "Rascal of Roatan": "Bet 0, 10, or 20 points. Earn/lose based on bid success.",
    "Juanita Jade": "Privately look through any undealt cards for the round.",
    "Harry the Giant": "Change your bid by +/-1 or leave it the same.",
}

SKIP_CARDS = {'Escape', 'Loot', 'Kraken', 'White Whale'}

SPECIAL_CARDS_WINNING_COMBINATIONS = { '(Skull King,Tigress)': True, 
                           '(Skull King,Rascal of Roatan)': True,
                            '(Skull King,Rosie D’ Laney)': True,
                            '(Skull King,Juanita Jade)': True,
                            '(Skull King,Harry the Giant)': True,
                            '(Skull King,Will the Bandit)': True,
                           '(Mermaid,Skull King)': True, 
                            '(Tigress,Mermaid)': True,
                            '(Rascal of Roatan,Mermaid)': True,
                            '(Rosie D’ Laney,Mermaid)': True,
                            '(Juanita Jade,Mermaid)': True,
                            '(Harry the Giant,Mermaid)': True,
                            '(Will the Bandit,Mermaid)': True,}
                            

SPECIAL_CARDS.update({pirate: 1 for pirate in PIRATE_ABILITIES.keys()})
NUMERIC_CARDS = {suit: list(range(1, 2)) for suit in SUITS}

suit_mapper = {"P": "Parrot", "T": "Treasure Chest", "M": "Pirate Map", "J": "Jolly Roger", "Tig": "Tigress", "SK": "Skull King", 
               "Mer": "Mermaid", "Esc": "Escape", "L": "Loot", "K": "Kraken", "WW": "White Whale", "RD": "Rosie D’ Laney", 
               "BB": "Will the Bandit", "RR": "Rascal of Roatan", "JJ": "Juanita Jade", "HG": "Harry the Giant"}

# Deck creation
def create_deck():
    deck = []
    for suit, numbers in NUMERIC_CARDS.items():
        deck.extend((suit, number) for number in numbers)
    for card, count in SPECIAL_CARDS.items():
        deck.extend([(card, None)] * count)
    return deck

