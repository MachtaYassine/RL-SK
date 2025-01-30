from collections import defaultdict
from Trick import Trick
import random

class Round:
    def __init__(self, round_number, players, deck, starting_player):
        self.round_number = round_number
        self.players = players
        self.deck = deck
        self.undealt_cards = None
        self.hands = defaultdict(list)
        self.tricks = []
        self.bets = []
        self.trick_wins = defaultdict(int)
        self.last_trick_winner = starting_player

    def deal_cards(self):
        random.shuffle(self.deck)
        cards_to_deal = self.round_number
        for player in self.players:
            self.hands[player] = [self.deck.pop() for _ in range(cards_to_deal)]

    def bet_on_tricks(self):
        for player in self.players:
            print(f"{player}'s hand: {self.hands[player]}")
            bet = int(input(f"{player}, enter your bet for tricks: "))
            self.bets.append(bet)

    def play_trick(self,index):
        #Initialize trick
        trick = Trick(players=self.players, hands=self.hands, bets=self.bets, remaining_deck=self.deck ,index=index,round_number=self.round_number)
        ordered_players = self.players[self.players.index(self.last_trick_winner):] + self.players[:self.players.index(self.last_trick_winner)]
        for player in ordered_players:
            print(f"Current trick: {trick.current_trick}")
            # player plays a card
            card = trick.play(player)
                         
        winner, _ = trick.resolve()
        print('####################################################')
        print(f"{winner} wins the trick!")
        self.trick_wins[winner] += 1
        self.tricks.append(trick)

    

    def calculate_scores(self):
        scores = {}
        for player, bet in zip(self.players, self.bets):
            wins = self.trick_wins[player]
            if bet == 0:
                scores[player] = 10 * self.round_number if wins == 0 else -10 * self.round_number
            elif bet == wins:
                scores[player] = 20 * bet
            else:
                scores[player] = -10 * abs(bet - wins)
        return scores

