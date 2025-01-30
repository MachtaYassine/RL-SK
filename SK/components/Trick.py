from SK.components.Cards import NUMERIC_CARDS, suit_mapper,SPECIAL_CARDS,SUITS,PIRATE_ABILITIES,SPECIAL_CARDS_WINNING_COMBINATIONS,SKIP_CARDS

class Trick:
    def __init__(self, players, hands, bets, remaining_deck , index, round_number):
        self.undealt_deck=remaining_deck
        self.players = players
        self.current_trick = []
        self.leading_suit = None
        self.highest_suit_card = None
        self.highest_card = None
        self.winner = None
        self.hands = hands
        #Info for RL policy ?
        self.bets = bets
        self.index = index
        self.round_number = round_number
        self.mermaid_in_trick = None
        self.skull_king_in_trick = False
        self.pirate_in_trick = False
        self.rosie_d_laney_player = None
        
        
    
    
    def handle_special_card(self,card,player):
        if card[0] == 'Mermaid' and self.mermaid_in_trick is None:
            self.mermaid_in_trick = player
        elif card[0] == 'Skull King':
            self.skull_king_in_trick = True # No need to keep track of the player who played the Skull King or Pirate
        elif card[0] in PIRATE_ABILITIES:
            self.pirate_in_trick = True
    
    def play(self, player):
        card= self.get_card_input(player,self.leading_suit,self.highest_suit_card)
        self.handle_special_card(card,player)
        # set leading suit if it's the first card played
        if self.leading_suit is None and card[0] in NUMERIC_CARDS:
            self.leading_suit = card[0]
            self.highest_suit_card = card
        
        self.current_trick.append((player, card))
        if self.is_higher_card(self.highest_card, card, self.leading_suit):
            self.highest_card = card
            self.winner = player
        if self.current_trick_has_all_3_special_cards():
            self.winner = self.mermaid_in_trick # this keeps track of the first player to play a mermaid
            self.highest_card = ('Mermaid',None)

    def is_higher_card(self, card_tuple1, card_tuple2, leading_suit):
        if card_tuple2[0] in SKIP_CARDS:
            return False # Skip cards are not compared
        if card_tuple1 is None :
            return True

        suit1, value1 = card_tuple1
        suit2, value2 = card_tuple2

        match (suit1, suit2):
            # Case 1: Both cards are numeric cards in valid suits
            case (s1, s2) if s1 in SUITS and s2 in SUITS:
                if s1 == s2: # If suits match, return value2 > value1
                    return value2 > value1
                if s2 == "Jolly Roger":
                    return True  # Jolly Roger (trump) always wins
                if s1 == "Jolly Roger":
                    return False # Card 2 is a regular suit card but card 1 is trump
                return s2 == leading_suit # both cards are regular suit cards, the one with the leading suit wins. (There should be no case where none of the cards are leading suit cards)

            # Case 2: Both cards are special cards
            case (s1, s2) if s1 in SPECIAL_CARDS and s2 in SPECIAL_CARDS:
                if s1 == s2:
                    return False  # Same special card (Can only be Mermaid vs Mermaid)
                if s1 in PIRATE_ABILITIES and s2 in PIRATE_ABILITIES:
                    return False  # 1st Pirate played wins
                if (s2, s1) in SPECIAL_CARDS_WINNING_COMBINATIONS:
                    return True  # Check winning combinations
                if (s1, s2) in SPECIAL_CARDS_WINNING_COMBINATIONS:
                    return False # s1 wins
            
            # Case 3: One card is a special card and the other is a numeric card
            case (s1, s2) if s1 in SPECIAL_CARDS and s2 in SUITS:
                return False  # numeric card loses compared to special card
            case (s1, s2) if s1 in SUITS and s2 in SPECIAL_CARDS:
                return True  # Special card wins compared to numeric card
                
        # Default: No higher card determined
        return False

            
        
                 

    def resolve(self):
        return self.winner, self.highest_card
    
    def get_card_input(self, player, leading_suit, highest_suit_card):
        # Available cards to play are cards from the same suit as the leading suit
        # if the player has cards from that suit higher than the highest suit card he must play one of them
        # Special cards can be played at any time
        available_cards = self.get_available_cards(player,leading_suit,highest_suit_card)
        print(f"{player}'s hand: {available_cards}")
        while True:
            try:
                card_input = input(f"{player}, enter the card you want to play (format: suit,value): ")
                # Split and transform input into a tuple
                split_input = card_input.split(',')
                suit, value = split_input[0], split_input[1] if len(split_input) > 1 else 'None'
                suit = suit_mapper.get(suit.strip())  # Map suit using suit_mapper
                value = int(value.strip()) if value.strip().isdigit() else None
                
                
                    
                
                # Form the card tuple
                card = (suit, value)
                
                # Check if the card is valid
                if card in available_cards:
                    self.hands[player].remove(card)
                    
                    if suit in PIRATE_ABILITIES.keys():
                        print(f" This is a special card with an ability: {PIRATE_ABILITIES[suit]}")
                        self.handle_card_special_ability(suit,player)
                    
                    
                    return card
                else:
                    print(f"{player}, you don't have that card. Please try again.")
            except Exception as e:
                print(f"Invalid input: {e}. Make sure to follow the format 'suit,value'. Try again.")

    def get_suit_and_special_cards(self,player,leading_suit):
        suit_cards,special_cards=[],[]
        for card in self.hands[player]:
            if card[0]==leading_suit:
                suit_cards.append(card)
            elif card[0] in SPECIAL_CARDS:
                special_cards.append(card)
        return suit_cards,special_cards
    
    
    def get_available_cards(self,player,leading_suit,highest_suit_card):
        # Available cards to play are cards from the same suit as the leading suit
        # if the player has cards from that suit higher than the highest suit card he must play one of them
        # Special cards can be played at any time
        if leading_suit is not None:
            suit_cards,special_cards=self.get_suit_and_special_cards(player,leading_suit)
            if len(suit_cards) > 0:
                suit_cards_higher_than_highest_suit_card = [card for card in suit_cards if card[1] > highest_suit_card[1]]
                if len(suit_cards_higher_than_highest_suit_card) > 0:
                    available_cards = suit_cards_higher_than_highest_suit_card+special_cards
                   
                else:
                    available_cards = suit_cards+special_cards
            else:
                available_cards = self.hands[player]
        else:
            available_cards = self.hands[player]
            
        return available_cards
    
    def current_trick_has_all_3_special_cards(self):
        return self.mermaid_in_trick is not None and self.skull_king_in_trick and self.pirate_in_trick
    
    
    def handle_card_special_ability(self,suit,player):
        if suit == 'Tigress':
            # THis pirate card has an ability to choose to play as a Pirate or as an Escape card
            type=input("Do you want to play as a Pirate or as an Escape card? (P,Esc)")
            if type == 'P':
                return ('Tigress',None)
            else:
                return ('Escape',None)
        elif suit == 'Rosie D’ Laney':
            # This pirate card has an ability to choose a player to lead the next trick
            player = input("Choose a player to lead the next trick:(1,2 etc..) ")
            self.rosie_d_laney_player = self.players[int(player)-1]
            return ('Rosie D’ Laney',None)
        elif suit == 'Will the Bandit':
            # This pirate card has an ability to draw 2 cards from the deck and discard 2
            # pop 2 cards from the deck and add them to the player's hand
            for i in range(2):
                card = self.undealt_deck.pop()
                self.hands[player].append(card)
            # discard 2 cards from the player's hand
            print(f"{player}'s new hand: {self.hands[player]}")
            try:
                first_card = tuple(input("Enter the first card you want to discard (format: suit,value): ").split(',')) # Handle addition of Nones !
                first_card = (suit_mapper.get(first_card[0].strip()), int(first_card[1].strip()))
                self.hands[player].remove(first_card)
                second_card = tuple(input("Enter the second card you want to discard (format: suit,value): ").split(','))
                second_card = (suit_mapper.get(second_card[0].strip()), int(second_card[1].strip()))
                self.hands[player].remove(second_card)
            except Exception as e:
                print(f"Invalid input: {e}. Make sure to follow the format 'suit,value'. Try again.")
            #add them back to deck
            self.undealt_deck.append(first_card)
            self.undealt_deck.append(second_card)
            
            return ('Will the Bandit',None)
        elif suit == 'Rascal of Roatan':
            # This pirate card has an ability to bet 0, 10, or 20 points. Earn/lose based on bid success
            bet = int(input("Bet 0, 10, or 20 points: "))
            
            return ('Rascal of Roatan',bet)
        elif suit == 'Juanita Jade':
            # This pirate card has an ability to privately look through any undealt cards for the 
            bool= input("Do you want to look through the undealt cards? (Y/N)")
            if bool == 'Y':
                print("             #######          ")
                print(f"Undealt cards: {self.undealt_deck}")
                print("             #######          ")
            
            return ('Juanita Jade',None)
        elif suit == 'Harry the Giant':
            # This pirate card has an ability to change your bid by +/-1 or leave it the same
            change = input("Change your bid by +/-1 or leave it the same: (+,-,0)")
            if change == '+':
                change = 1
            elif change == '-':
                change = -1
            else:
                change = 0
            self.bets[self.players.index(player)]+=change
            return ('Harry the Giant',None)
        else:
            raise ValueError(f"Invalid special card: {suit}")