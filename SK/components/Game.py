from SK.components.Round import Round
from SK.components.Cards import create_deck

class Game:
    def __init__(self, num_players):
        self.num_players = num_players
        self.players = [f"Player {i+1}" for i in range(num_players)]
        self.deck = create_deck()
        self.scores = {player: 0 for player in self.players}

    def play_game(self):
        for round_number in range(1, 11):
            print("####################################################")
            print(f"Starting Round {round_number}")
            self.starting_player = self.players[(round_number-1) % self.num_players]
            round_obj = Round(round_number, self.players, self.deck, self.starting_player)
            round_obj.deal_cards()
            round_obj.bet_on_tricks()
            for i in range(round_number):
                round_obj.play_trick(i)
            round_scores = round_obj.calculate_scores()
            print(f"Scores for Round {round_number}: {round_scores}")
            for player, score in round_scores.items():
                self.scores[player] += score
            print(f"Total scores after Round {round_number}: {self.scores}")
        print("Game over! Final scores:")
        for player, score in self.scores.items():
            print(f"{player}: {score}")
        print(" Rankings:") 
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for i, (player, score) in enumerate(sorted_scores):
            print(f"{i+1}. {player}: {score}")

# Main script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=4, help="Number of players")
    args = parser.parse_args()

    game = Game(args.players)
    game.play_game()
