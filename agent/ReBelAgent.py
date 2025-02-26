import random
import numpy as np
import networkx as nx
from .SimpleAgent import SkullKingAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # NEW: progress bar


class TupleDataset(Dataset):
    def __init__(self, training_dict):
        self.data = list(training_dict.items())  # List of ((pbs, hand), target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (pbs, hand), target = self.data[idx]
        return pbs, hand, target  # Return tuple directly


class ReBelValueNN(nn.Module):
    """
    Estimates the score obtained by each player from a public belief state (pbs) and the hands of each player.

    Args:
        max_players (int): Maximum number of players the model should support (default=5).
        max_rounds (int): Maximum number of rounds the model should support (default=10).
        card_dim (int): Number of card features (default=4).
        hidden_dim (int): Number of units in the hidden layers (default=128).
        lr (float): Learning rate of the optimizer (default=0.001)

    Inputs:
        - pbs (torch.Tensor): Tensor of shape ((n_rounds+1) * n_players).
        - set_of_hands (torch.Tensor): Tensor of shape (n_rounds * n_players) (transposed so each column represents a player).

    Internally:
        - pbs is masked and reshaped to (max_rounds+1) * max_players.
        - set_of_hands is masked, transposed, and reshaped to max_rounds * max_players.
        - The final input is concatenated into a single vector of size (max_players * (2*max_rounds+1)).

    Output:
        - A tensor of shape (1 * max_players) representing the estimated scores for each player.
    """

    def __init__(self, max_players=5, max_rounds=10, card_dim=4, hidden_dim=128, lr=0.001):
        super(ReBelValueNN, self).__init__()
        self.max_players = max_players
        self.max_rounds = max_rounds
        self.hidden_dim = hidden_dim
        self.card_dim = card_dim

        # Compute input and output dimensions
        self.input_dim = (
            self.max_players * self.card_dim + self.max_rounds * self.card_dim * self.max_players + self.max_rounds * self.card_dim * self.max_players
        ) # bid + pbs + hands
        self.output_dim = self.max_players  # (1*5)

        # Define a simple feedforward network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, pbs, set_of_hands, device="cpu"):
        """
        Forward pass to estimate player scores.

        Args:
            pbs (torch.Tensor): Shape (n_rounds+1, n_players, card_dim).
            set_of_hands (torch.tensor): Shape (n_players, n_rounds, card_dim).
            device (str): Device to use for computation (default="cpu").

        Returns:
            torch.Tensor: Estimated scores of shape (1, max_players).
        """
        batch_size, n_players, n_rounds, card_dim = set_of_hands.size()

        # Pad/mask pbs to size (max_rounds+1, max_players)
        padded_pbs = torch.full((batch_size, self.max_rounds + 1, self.max_players, card_dim), 0.0).to(device)
        padded_pbs[:, : n_rounds + 1, :n_players] = pbs

        # Pad/mask set_of_hands to size (max_rounds, max_players)
        padded_hands = torch.full((batch_size, self.max_rounds, self.max_players, card_dim), 0.0).to(device)
        padded_hands[:, :n_rounds, :n_players] = torch.swapaxes(set_of_hands, 1, 2)

        # Flatten and concatenate
        x = torch.cat(
            (padded_pbs.view(batch_size, -1), padded_hands.view(batch_size, -1)), dim=1
        ) # Shape: (batch_size, input_dim)

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # TODO : Maybe will need to set the extra values to zero, we'll see
        return x  # Shape: (1, max_players)

    def train_model(self, training_dict, epochs=10, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TupleDataset(training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for pbs, set_of_hands, targets in dataloader:
                pbs = pbs.to(device)
                set_of_hands = set_of_hands.to(device)
                targets = torch.nan_to_num(targets, nan=0.0).to(device)
                self.optimizer.zero_grad()
                outputs = self(pbs, set_of_hands, device)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
        self.to("cpu")
        self.eval()


class ReBelPolicyNN(nn.Module):
    """
    Estimates the policy (probabilities of playing each card) given a public belief state (pbs) and the player's hand.

    Args:
        max_players (int): Maximum number of players the model should support (default=5).
        max_rounds (int): Maximum number of rounds the model should support (default=10).
        card_dim (int): Number of card features (default=4).
        hidden_dim (int): Number of units in the hidden layers (default=128).
        lr (float): Learning rate of the optimizer (default=0.001)

    Inputs:
        - pbs (torch.Tensor): Shape ((n_rounds+1) * n_players).
        - hand (torch.Tensor): Shape (n_rounds,).

    Internally:
        - pbs is padded to (max_rounds+1, max_players).
        - hand is padded to (max_rounds, 1).
        - The final input is concatenated into a single vector of size ((2 * max_rounds + 1) * max_players + max_rounds).

    Output:
        - A probability distribution over the player's hand (shape: max_rounds, 1).
    """

    def __init__(self, max_players=5, max_rounds=10, card_dim=4, hidden_dim=128, lr=0.001):
        super(ReBelPolicyNN, self).__init__()
        self.max_players = max_players
        self.max_rounds = max_rounds
        self.hidden_dim = hidden_dim
        self.card_dim = card_dim

        # Compute input and output dimensions
        self.input_dim = (
            self.max_players * self.card_dim  + self.max_rounds * self.card_dim * self.max_players + self.max_rounds * self.card_dim
        ) # bid + pbs + hand
        self.output_dim = self.max_rounds  # Probabilities over each card in hand (10)

        # Define a feedforward network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pbs, hand, device="cpu"):
        """
        Forward pass to estimate action probabilities.

        Args:
            pbs (torch.Tensor): Shape (n_rounds+1, n_players, card_dim).
            hand (torch.Tensor): Shape (n_rounds,).
            device (str): Device to use for computation (default="cpu").

        Returns:
            torch.Tensor: Probability distribution over the hand (shape: max_rounds, 1).
        """
        # TODO : Make sure that the cards already played are masked
        # TODO : Mask the cards that can't be played
        batch_size, n_rounds, n_players, _ = pbs.size()
        n_rounds = n_rounds - 1  # ignore first row (bids)
        hand_size = hand.size(1)
        padded_pbs = torch.full((batch_size, self.max_rounds + 1, self.max_players, self.card_dim), 0.0).to(device)
        padded_pbs[:, : n_rounds + 1, :n_players] = pbs
        padded_hand = torch.full((batch_size, self.max_rounds, self.card_dim), 0.0).to(device)
        padded_hand[:, :hand_size] = hand
        
        x = torch.cat(
            (padded_pbs.view(batch_size, -1), padded_hand.view(batch_size, -1)), dim=1
        )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Compute raw probabilities using softmax
        card_mask = torch.full((batch_size, self.max_rounds), 1.0).to(device)
        card_mask[n_rounds:] = 0.0  # Mask the last rounds
        x = x * card_mask
        probs = F.softmax(x, dim=0)  # shape: (max_rounds,)
        
        # # ----- NEW MASKING SECTION -----
        # # Assume batch_size == 1 and that the hand corresponds to player 0.
        # # Extract played cards for that player from pbs (rows 1 onward)
        # played_cards = padded_pbs[0, 1: n_rounds+1, 0, :]  # shape: (n_rounds, card_dim)
        # hand_cards = padded_hand[0, :n_rounds, :]           # shape: (n_rounds, card_dim)
        # available_mask = torch.ones(hand_cards.size(0), device=device)
        # for i in range(hand_cards.size(0)):
        #     for j in range(played_cards.size(0)):
        #         # Only consider valid played cards (those not equal to -1)
        #         if (played_cards[j] != -1).all() and torch.all(hand_cards[i] == played_cards[j]):
        #             available_mask[i] = 0
        #             break
        # # Apply mask: set probabilities for unavailable cards to zero.
        # probs = probs * available_mask
        # # Force negatives or nan values to zero
        # probs[probs < 0] = 0
        # probs[torch.isnan(probs)] = 0
        # # Renormalize if any probability is available
        # total = probs.sum()
        # if total > 0:
        #     probs = probs / total
        # # ----- END NEW MASKING SECTION -----
        
        return probs  # Final probability distribution over the hand

    def train_model(self, training_dict, epochs=10, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TupleDataset(training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for pbs, hand, probas in dataloader:
                pbs = pbs.to(device)
                hand = hand.to(device)
                probas = torch.nan_to_num(probas, nan=0.0).to(device)
                self.optimizer.zero_grad()
                outputs = self(pbs, hand, device)
                loss = self.loss_fn(outputs, probas)
                loss.backward()
                self.optimizer.step()
        self.to("cpu")
        self.eval()


class ReBelBidNN(nn.Module):
    """
    Estimates how many tricks a player will bid based on their hand.

    Args:
        max_rounds (int): Maximum number of rounds (default=10).
        card_dim (int): Number of card features (default=4).
        hidden_dim (int): Number of hidden layer units (default=128).
        lr (float): Learning rate of the optimizer (default=0.001)

    Inputs:
        - hand (torch.Tensor): Shape (n_rounds,).

    Internally:
        - The hand is padded to (max_rounds,).
        - It is then passed through a feedforward network.

    Output:
        - A single value representing the estimated number of tricks to win.
    """

    def __init__(self, max_rounds=10, card_dim=4, hidden_dim=128, lr=0.001):
        super(ReBelBidNN, self).__init__()
        self.max_rounds = max_rounds
        self.input_dim = max_rounds * card_dim
        self.hidden_dim = hidden_dim
        self.card_dim = card_dim

        # Define the feedforward network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, 1)  # Output is a single value

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, hand, device="cpu"):
        """
        Forward pass to estimate the number of tricks the player will bid.

        Args:
            hand (torch.Tensor): Shape (n_rounds,).
            device (str): Device to use for computation (default="cpu").

        Returns:
            torch.Tensor: A single estimated bid value (shape: (1,)).
        """
        batch_size, n_rounds, _ = hand.size()

        # Pad/mask hand to size (max_rounds,)
        padded_hand = torch.full((batch_size, self.max_rounds, self.card_dim), 0.0).to(device)
        padded_hand[:, :n_rounds] = hand

        # Forward pass through the network
        x = F.relu(self.fc1(padded_hand.view(batch_size, -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # TODO : Make sure the output is less than n_rounds
        return x  # Output a single scalar value

    def train_model(self, training_dict, epochs=10, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # Process training dict to extract individual hands
        processed_training_dict = {
            (hand, 0): tricks[idx]
            for hands, tricks in training_dict.items()
            for idx, hand in enumerate(hands)
        }
        dataset = TupleDataset(processed_training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for hand, _, bid in dataloader:
                hand = hand.to(device)
                bid = torch.nan_to_num(bid, nan=0.0).to(device)
                self.optimizer.zero_grad()
                outputs = self(hand, device)
                loss = self.loss_fn(outputs.view(-1), bid)
                loss.backward()
                self.optimizer.step()
        self.to("cpu")
        self.eval()


class ReBel:
    """
    Implementation of the ReBel algorithm for training neural networks in the Skull King game.

    Attributes:
        n_players (int): Number of players in the game
        n_rounds (int): Number of rounds in the game
        K (int): Number of different hand sets to generate
        N (int): Number of public belief states to generate per hand set
        T (int): Number of iterations for policy improvement
        value_network (ReBelValueNN): Network for estimating player scores
        policy_network (ReBelPolicyNN): Network for action selection
        bidding_network (ReBelBidNN): Network for bidding decisions
        deck (list): List of cards in the deck
        simu_depth (int): Depth of simulation tree
        warm_start (bool): Whether to use pre-trained networks
    """

    def __init__(
        self,
        n_players,
        n_rounds,
        K,
        N,
        T,
        value_network,
        policy_network,
        bidding_network,
        deck,
        simu_depth=1,
        warm_start=False,
    ):
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.K = K
        self.N = N
        self.T = T
        self.value_network = value_network
        self.policy_network = policy_network
        self.bidding_network = bidding_network
        self.deck = deck
        self.simu_depth = simu_depth
        self.warm_start = warm_start

    def train(self):
        """Main training loop implementing the ReBel algorithm."""
        # We start by sampling K sets of hands from the deck
        sets_of_hands = self.get_sets_of_hands()
        bidding_network_training_dict = {}
        # Set all the networks to evaluation mode
        self.value_network.eval()
        self.policy_network.eval()
        self.bidding_network.eval()
        for set_of_hands in tqdm(sets_of_hands, desc="Set of Hands"):
            # Get bid of each player
            bids = self.get_bids(set_of_hands)
            # Sample N public states at random from the hands as a starting point
            set_of_pbs = self.get_initial_pbs(set_of_hands, bids)
            value_network_training_dict, policy_network_training_dict = {}, {}

            # Now iterate through these public states and run the ReBel algorithm
            with torch.no_grad():
                for pbs in tqdm(set_of_pbs, desc="PBS states", leave=False):
                    # Begin iterative exploration until reaching the end of the game
                    while not self._is_terminal(pbs):
                        # Get the exploration tree
                        G = self._initialize_tree(pbs, set_of_hands)
                        rolling_mean_policy = (
                            G.copy()
                        )  # We will update the weights on this graph as a way of representing a policy
                        # Initial value associated to the pbs
                        expected_value = self._get_expected_value(G, bids, set_of_hands)
                        # Sample a timestep `t_sample` with increasing probability
                        t_sample = np.random.choice(
                            np.arange(self.T),
                            p=np.linspace(0.1, 1, self.T)
                            / np.sum(np.linspace(0.1, 1, self.T)),
                        )
                        for t in tqdm(range(self.T), desc="Time steps", leave=False):
                            if t == t_sample:
                                new_pbs = self._sample_leaf(G)
                            G = self._update_policy(
                                G, set_of_hands
                            )  # Weights now represent \pi^{t}
                            for u, v in rolling_mean_policy.edges():
                                old = rolling_mean_policy[u][v]["weight"]
                                new = G[u][v]["weight"] if G.has_edge(u, v) else old
                                rolling_mean_policy[u][v]["weight"] = (t/(t+1))*old + (1/(t+1))*new
                            expected_value = (t / (t + 1)) * expected_value + (
                                1 / (t + 1)
                            ) * self._get_expected_value(
                                G, bids, set_of_hands
                            )  # Update expected value
                        value_network_training_dict[(pbs, torch.tensor(set_of_hands, dtype=torch.float32))] = expected_value
                        for state in list(rolling_mean_policy.nodes):
                            # Find the player with the first -1 in pbs after bids.
                            mask = pbs[1:] == -1
                            player = torch.nonzero(mask, as_tuple=False)[0][1].item()  # first -1 in player column
                            player_hand = set_of_hands[player]  # assumed to be a list of cards (each card as list)
                            # Create a probability vector over the player's hand.
                            probas = torch.full((self.value_network.max_rounds,), 0.0)
                            # Iterate over successors of pbs.
                            for child in G.successors(pbs):
                                child_state = torch.tensor(child)
                                # Find the trick index where the parent's pbs is -1 but child's state is not.
                                # We consider rows starting from index 1 (ignoring bids row).
                                diff = (child_state[1:, player] != pbs[1:, player])
                                indices = torch.nonzero(diff, as_tuple=False)
                                if len(indices) > 0:
                                    trick_i = indices[0][0].item() + 1  # add 1 to adjust index offset
                                    card_played = child_state[trick_i, player].tolist()
                                    try:
                                        idx = player_hand.index(card_played)
                                        weight = G[pbs][child]["weight"]
                                        probas[idx] = weight
                                    except ValueError:
                                        # If the card is not found, skip it.
                                        continue
                            policy_network_training_dict[(state, torch.tensor(player_hand, dtype=torch.float32))] = probas
                        pbs = new_pbs
            self.value_network.train_model(
                value_network_training_dict
            )  # Update value network
            self.policy_network.train_model(
                policy_network_training_dict
            )  # Update policy network
            self.warm_start = True if np.random.rand() > 0.5 else False
            bidding_network_training_dict[torch.tensor(set_of_hands, dtype=torch.float32)] = self._get_average_tricks_won(
                value_network_training_dict
            )
        self.bidding_network.train_model(
            bidding_network_training_dict
        )  # Update bidding network

    def get_sets_of_hands(self):
        """
        Simulate dealing n_rounds cards from a fresh copy of the deck for K iterations.
        Ensures each hand contains unique cards.
        Returns:
            sets_of_hands: list of lists, where each inner list contains hands for all players.
        """
        sets_of_hands = []
        for _ in range(self.K):
            # Make a local copy of the deck
            if isinstance(self.deck, torch.Tensor):
                deck_copy = self.deck.clone()
                perm = torch.randperm(deck_copy.size(0))
                deck_copy = deck_copy[perm]
                deck_list = deck_copy.tolist()  # convert to list for easy slicing
            else:
                deck_list = self.deck.copy()
                random.shuffle(deck_list)
            total_cards = self.n_players * self.n_rounds
            dealt_cards = deck_list[:total_cards]  # deal only necessary cards
            # Split dealt_cards into hands: each hand gets exactly n_rounds cards
            hands = [dealt_cards[i * self.n_rounds : (i + 1) * self.n_rounds] for i in range(self.n_players)]
            sets_of_hands.append(hands)
        return sets_of_hands

    def get_bids(self, set_of_hand):
        """
        Determine a bid for each player given their hands.

        Args:
            set_of_hand (numpy.ndarray): Shape (n_players, n_rounds), contains the hand of each player.

        Returns:
            bids (torch.Tensor): Tensor of shape (n_players, 1), containing each player's bid.
        """
        # Convert numpy array to a torch tensor (if not already)
        set_of_hand = torch.tensor(set_of_hand, dtype=torch.float32)

        # Process each player's hand through the bidding network
        bids = torch.stack(
            [self.bidding_network(hand.unsqueeze(0)).squeeze() for hand in set_of_hand]
        )
        # Transform bids to integers
        bids = torch.round(bids).int()
        return bids  # Shape: (n_players, 1)

    def get_initial_pbs(self, set_of_hand, bids):
        """
        Simulates N possible states that can be reached given the players' cards.
        A public state contains all the cards that have been played and the players' bids.

        Args:
            set_of_hand (numpy.ndarray): Shape (n_players, n_rounds, card_dim), contains the hand of each player.
            bids (torch.Tensor): Shape (n_players, 1), containing the bid of each player.

        Returns:
            set_of_pbs (torch.Tensor): Tensor of shape (N, (n_rounds+1) * n_players * card_dim),
                                    representing possible public states.
        """

        # Convert hands and bids to tensors
        set_of_hand = torch.tensor(set_of_hand, dtype=torch.float32)
        n_players, n_rounds, card_dim = set_of_hand.size()
        bids = bids.float().view(n_players)  # Flatten bids
        

        set_of_pbs = []

        for _ in range(self.N):
            # Initialize public state with -1 for unplayed moves
            pbs = torch.full((n_rounds + 1, n_players, card_dim), -1.0)

            # Fill in bids 
            pbs[0, :n_players] = bids.expand(card_dim, n_players).T

            # Copy hands to modify without affecting the input
            hands_copy = set_of_hand.clone()
            total_cards = n_players * n_rounds

            # Randomly decide how many cards have already been played
            num_played_cards = np.random.randint(0, total_cards + 1)

            played_cards = []  # Stores played cards in order

            # Players play in order: 0 -> 1 -> 2 -> ... -> (n_players - 1) -> repeat
            # TODO : If the results are bad maybe think of integrating the order in which a trick is played
            for i in range(num_played_cards):
                current_player = i % n_players  # Determines the player turn

                # Get the available rows (cards) for this player (rows with all values not -1)
                player_hand = hands_copy[current_player]  # shape: [n_rounds, card_dim]
                valid_mask = (player_hand != -1).all(dim=1)  # boolean mask for valid rows
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    chosen_idx = valid_indices[np.random.choice(len(valid_indices))]
                    chosen_card = player_hand[chosen_idx].clone()  # clone to preserve value
                    played_cards.append((current_player, chosen_card))
                    # Set the chosen row to -1 to mark as played.
                    hands_copy[current_player][chosen_idx] = -1

            # Fill the played cards into the public state matrix
            round_idx = 1
            for i, (player, card) in enumerate(played_cards):
                pbs[round_idx, player] = card
                if (
                    i + 1
                ) % n_players == 0:  # Move to the next round after all players played
                    round_idx += 1

            set_of_pbs.append(pbs)  # Flatten to match network input format

        return torch.stack(set_of_pbs)  # Shape: (N, (n_rounds+1) * n_players)

    def _is_terminal(self, pbs):
        """
        Return True if starting from the pbs there are cards left to play.

        Args:
            pbs (torch.tensor): Public state of shape ((n_rounds + 1), n_players).

        Returns:
            bool: True if there are fewer than simu_depth tricks left, False otherwise.
        """

        return torch.sum((pbs[1:] == -1).any(dim=2)) == 0

    def _initialize_tree(self, pbs, set_of_hands):
        """
        Creates the exploration tree with a starting public state. The nodes are public states,
        and the edges are weighted with the policy probability.

        Args:
            pbs (torch.tensor): Public state.
            set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]])

        Returns:
            G (nx.DiGraph): Weighted exploration tree.
        """

        # Initialize graph with directed edges
        G = nx.DiGraph()

        # Add root node (initial public state)
        G.add_node(pbs)  # Convert tensor to a hashable type

        d = 0  # Depth (number of full tricks played)

        while d < self.simu_depth:
            for _ in range(
                self.n_players
            ):  # Ensure all players play one trick before incrementing d
                new_nodes = []
                leaves = [
                    node for node in G.nodes if G.out_degree(node) == 0
                ]  # Get current leaves

                for leaf in leaves:
                    current_pbs = torch.tensor(leaf)  # Retrieve the pbs for this leaf

                    # Find the first empty slot (-1) in the tricks (ignoring bids row)
                    mask = current_pbs[1:] == -1
                    if not mask.any():
                        continue  # No more moves to play

                    trick_idx, current_player, _ = torch.nonzero(mask, as_tuple=False)[
                        0
                    ]  # Get first empty trick slot
                    current_player = current_player.item()

                    # Get the player's remaining hand
                    played_cards = []
                    for trick in current_pbs[1:]:
                        for player_idx in range(trick.shape[0]):
                            card = trick[player_idx]
                            if not torch.equal(card, torch.full((mask.size(2),), -1.0)):
                                played_cards.append(card.tolist())
                    # current_hand: filter cards from player's hand that have been played.
                    current_hand = [
                        card for card in set_of_hands[current_player] if card not in played_cards
                    ]

                    if not current_hand:
                        continue  # No cards left to play

                    # Get probabilities for each card
                    if self.warm_start:
                        hand_tensor = torch.tensor(
                            set_of_hands[current_player] , dtype=torch.float32
                        ).unsqueeze(0)
                        proba = (
                            self.policy_network(current_pbs.unsqueeze(0), hand_tensor).squeeze(0)
                        )
                    else:
                        proba = torch.full((len(current_hand),), 1 / len(current_hand))

                    # Generate new nodes for each possible card play
                    for card, prob in zip(current_hand, proba.tolist()):
                        new_pbs = current_pbs.clone()
                        new_pbs[trick_idx + 1, current_player] = torch.tensor(
                            card  # Update trick row for this player
                        )

                        if new_pbs not in G:
                            G.add_node(new_pbs)
                            new_nodes.append(new_pbs)

                        # Add edge with probability weight
                        G.add_edge(leaf, new_pbs, weight=prob)

                if not new_nodes:
                    return G  # Stop early if no new nodes were generated

            d += 1  # Update depth only after all players have played in a trick

        return G

    def _get_expected_value(self, G, bids, set_of_hands):
        """
        Compute the average value of each player starting from the root of G.

        Args:
            G (nx.Graph): Weighted exploration tree.
            bids (torch.tensor): Tensor containing the bid of each player (size n_players * 1).
            set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]]).

        Returns:
            expected_value (torch.tensor): Tensor containing the scores of every player.
        """
        expected_value = torch.zeros(
            self.value_network.max_players, dtype=torch.float32
        )  # Initialize expected values for all players

        # Get leaf nodes of the tree
        leaves = [node for node in G.nodes if G.out_degree(node) == 0]

        for leaf in leaves:
            pbs = torch.tensor(leaf)  # Convert leaf (public state) to a tensor

            # Check if the game is fully played (no -1 present in the tricks)
            if (pbs[1:] == -1).sum() == 0:
                values = self._compute_final_scores(pbs, bids)  # Compute exact scores
                # Pad values to match the number of players
                values = torch.cat(
                    (values, torch.zeros(self.value_network.max_players - len(values)))
                )
            else:
                set_of_hands_tensor = torch.tensor(set_of_hands, dtype=torch.float32).unsqueeze(0)
                values = self.value_network(pbs.unsqueeze(0), set_of_hands_tensor) # Estimate scores
                values = values.squeeze(0)  # Remove batch dimension

            # Compute probability of reaching this leaf
            proba = 1.0
            node = leaf
            path = [node]
            while True:
                in_edges = list(G.in_edges(node))
                if not in_edges:
                    break  # reached the root
                parent = in_edges[0][0]
                path.insert(0, parent)
                node = parent
            for i in range(len(path) - 1):
                proba *= G[path[i]][path[i + 1]]["weight"]

            expected_value += values * proba  # Accumulate weighted values

        # Mask values that correspond to player added for padding 
        expected_value[self.n_players:] = float("-inf")
        return expected_value.detach()

    def _compute_final_scores(self, pbs, bids):
        """
        Compute the final scores for each player based on the game rules.

        Args:
            pbs (torch.tensor): Public state (tricks played, bids).
            bids (torch.tensor): Players' bids (n_players * 1).

        Returns:
            scores (torch.tensor): Tensor containing the final scores for each player.
        """
        # TODO : Check for illegal moves
        atout_suit = 3  # The 4th index (0-based) is the atout suit

        tricks_won = torch.zeros(self.n_players, dtype=torch.int32)

        for t in range(1, self.n_rounds + 1):  # Iterate over tricks (excluding bid row)
            trick = pbs[t]
            lead_suit = None
            highest_card = -1
            winner = None

            for i in range(self.n_players):
                card = trick[i]

                if (
                    card.max() == -1
                ):  # If -1, the player hasn't played (not possible in a complete game)
                    continue

                suit = card.argmax()  # Get suit index
                value = card[suit]  # Get card value

                if lead_suit is None:
                    lead_suit = suit  # First card determines the lead suit

                if suit == atout_suit:  # Atout beats all non-atout cards
                    if winner is None or (trick[winner][atout_suit] < value):
                        highest_card = value
                        winner = i
                elif (
                    suit == lead_suit
                    and winner is None
                    or (suit == lead_suit and highest_card < value)
                ):
                    highest_card = value
                    winner = i

            if winner is not None:
                tricks_won[winner] += 1

        scores = torch.zeros(self.n_players, dtype=torch.int32)

        for i in range(self.n_players):
            bid = bids[i].item()
            trick_won = tricks_won[i].item()

            if bid == 0:
                if trick_won == 0:
                    scores[i] = 10 * self.n_rounds
                else:
                    scores[i] = -10 * self.n_rounds
            elif bid == trick_won:
                scores[i] = 20 * trick_won
            else:
                scores[i] = -10 * abs(trick_won - bid)

        return scores

    def _sample_leaf(self, G):
        """
        Sample the pbs at the leaf of G with the highest probability of happening.

        Args:
            G (nx.Graph): Weighted exploration tree.

        Returns:
            pbs (torch.tensor): Public state.
        """
        leaves = [node for node in G.nodes if G.out_degree(node) == 0]  # Get leaf nodes

        max_proba = 0
        best_leaf = None

        for leaf in leaves:
            proba = 1.0
            node = leaf
            path = [node]
            while True:
                in_edges = list(G.in_edges(node))
                if not in_edges:
                    break
                parent = in_edges[0][0]
                path.insert(0, parent)
                node = parent
            for i in range(len(path) - 1):
                proba *= G[path[i]][path[i + 1]]["weight"]

            if proba >= max_proba:
                max_proba = proba
                best_leaf = leaf

        return torch.tensor(best_leaf)

    def _update_policy(self, G, set_of_hands):
        """
        Update the policy, hence the weights of G, using on-the-fly expected value computation
        with memoization to avoid redundant subgraph constructions.
        
        Args:
            G (nx.Graph): Weighted exploration tree.
            set_of_hands (numpy.ndarray): Set of hands (list of lists).
        
        Returns:
            updated_G (nx.Graph)
        """
        # Use a memoization dictionary to cache computed expected values
        memo = {}
        set_of_hands = torch.tensor(set_of_hands, dtype=torch.float32).unsqueeze(0)
        for edge in G.edges:
            parent, child = edge
            # Compute expected values for parent and child nodes
            expected_parent = self._compute_expected_value_recursive(G, parent, set_of_hands, memo)
            expected_child = self._compute_expected_value_recursive(G, child, set_of_hands, memo)
            # Identify the player: first index in parent where value is -1 in the pbs (ignoring first row)
            player = torch.where(torch.tensor(parent)[1:] == -1)[1][0].item()
            regret = max(0, expected_child[player] - expected_parent[player])
            # Update edge weight based on normalized regret
            if parent not in memo:  # safety check, though not needed here
                memo[parent] = expected_parent
            # We update each edge later after grouping per parent:
            G[parent][child]["regret"] = regret  # temporarily store regret
        
        # Now update weights per parent node.
        for parent in set([e[0] for e in G.edges]):
            # Get all edges from parent with stored regrets.
            parent_edges = [(parent, child) for child in G.successors(parent)]
            total_regret = sum(G[parent][child].get("regret", 0) for _, child in parent_edges)
            if total_regret > 0:
                for _, child in parent_edges:
                    G[parent][child]["weight"] = G[parent][child]["regret"] / total_regret
            else:
                num_edges = len(parent_edges)
                for _, child in parent_edges:
                    G[parent][child]["weight"] = 1 / num_edges
            # Cleanup temporary "regret" value.
            for _, child in parent_edges:
                del G[parent][child]["regret"]
        return G

    def _compute_expected_value_recursive(self, G, node, set_of_hands, memo):
        # Memoization to avoid recomputation.
        bids = node[0].T[0]
        if node in memo:
            return memo[node].squeeze()
        node_tensor = torch.tensor(node)
        children = list(G.successors(node))
        if not children:
            # If leaf, compute exact scores if game is complete, else use value_network.
            if (node_tensor[1:] == -1).sum() == 0:
                value = self._compute_final_scores(node_tensor, bids)
                # Pad values to match the number of players
                value = torch.cat(
                    (value, torch.zeros(self.value_network.max_players - len(value)))
                )
            else:
                value = self.value_network(node_tensor, set_of_hands)
            memo[node] = value
            return value.squeeze()
        # Otherwise, expected value is the weighted sum over children.
        value = torch.zeros_like(self.value_network(node_tensor, set_of_hands))
        for child in children:
            w = G[node][child]["weight"]
            child_val = self._compute_expected_value_recursive(G, child, set_of_hands, memo)
            value += w * child_val
        memo[node] = value
        return value.squeeze()

    def _get_average_tricks_won(self, value_network_training_dict):
        """
        Compute the average number of tricks won by each player.

        Args:
            value_network_training_dict (dict): Dictionary mapping hands to their computed scores (tensors).

        Returns:
            avg_tricks_won (torch.Tensor): Tensor of shape (n_players,) containing the average tricks won per player.
        """
        # TODO : Value are not gonna be mutiples of 20 and 10
        total_tricks = torch.zeros(self.n_players)  # Initialize trick count tensor
        count = len(value_network_training_dict)  # Number of hands evaluated

        for (
            value
        ) in (
            value_network_training_dict.values()
        ):  # value is a tensor of shape (max_players,)
            value = value[:self.n_players]
            tricks_won = torch.zeros_like(value)

            # Compute tricks won for each player
            tricks_won[value > 0] = value[value > 0] / 20  # If positive, divide by 20
            tricks_won[(value < 0) & (value != -10 * self.n_rounds)] = (
                torch.abs(value[(value < 0) & (value != -10 * self.n_rounds)]) / 10
            )  # If negative but not -10 * self.n_rounds, divide by -10
            tricks_won[value == -10 * self.n_rounds] = (
                1  # If exactly -10 * self.n_rounds, player bid 0 but won at least 1 trick
            )

            total_tricks += tricks_won  # Accumulate tricks won

        return (
            total_tricks / count if count > 0 else total_tricks
        )  # Avoid division by zero


class ReBelSkullKingAgent(SkullKingAgent):
    """
    A Skull King agent trained using the ReBel algorithm.

    This agent creates and trains its own networks using ReBel, then uses them for bidding and playing.
    """

    def __init__(self, num_players):
        super().__init__(num_players)
        # Initialize new networks if not provided
        self.policy_network = ReBelPolicyNN()
        self.bidding_network = ReBelBidNN()
        self.value_network = ReBelValueNN()

    def train(self, n_players=5, n_rounds=10, K=20, N=10, T=50, simu_depth=1):
        """
        Train the agent using the ReBel algorithm with vectorized cards.

        Args:
            n_players (int): Number of players in the game
            n_rounds (int): Number of rounds in the game
            K (int): Number of different hand sets to generate
            N (int): Number of public belief states per hand set
            T (int): Number of iterations for policy improvement
            simu_depth (int): Depth of simulation tree
        """
        # Create vectorized deck where each card is [Parrot, Treasure Chest, Treasure Map, Jolly Roger]
        deck = []
        for rank in range(1, 15):
            for suit_idx in range(4):  # 4 suits
                card = [0, 0, 0, 0]  # one-hot encoding for suit
                card[suit_idx] = rank  # put rank in the suit's position
                deck.append(card)

        # Convert deck to tensor for ReBel
        deck = torch.tensor(deck, dtype=torch.float32)

        # Initialize ReBel trainer
        rebel = ReBel(
            n_players=n_players,
            n_rounds=n_rounds,
            K=K,
            N=N,
            T=T,
            value_network=self.value_network,
            policy_network=self.policy_network,
            bidding_network=self.bidding_network,
            deck=deck,
            simu_depth=simu_depth,
        )

        # Train networks
        rebel.train()

    def _process_observation(self, observation):
        """Convert observation into tensor format for ReBel networks with finalized inputs."""
        suit_mapping = {
            "Parrot": 0,
            "Treasure Chest": 1,
            "Treasure Map": 2,
            "Jolly Roger": 3,
        }
        # For bidding phase: convert hand to tensor of shape (1, hand_length, 4)
        if observation.get("bidding_phase", False):
            hand = observation["hand"]
            hand_tensor = []
            for card in hand:
                card_vec = [0.0, 0.0, 0.0, 0.0]
                suit_idx = suit_mapping[card[0]]
                card_vec[suit_idx] = float(card[1])
                hand_tensor.append(card_vec)
            return torch.tensor(hand_tensor, dtype=torch.float32).unsqueeze(0)
        else:
            # For playing phase: build pbs tensor of shape (1, rounds+1, num_players, 4)
            rounds = observation["round_number"]
            pbs = torch.full((rounds + 1, self.num_players, 4), -1.0, dtype=torch.float32)
            # Fill first row with bids (replicate each bid across all 4 dimensions)
            bids = observation["all_bids"]
            pbs[0, :self.num_players] = torch.tensor(bids, dtype=torch.float32).expand(4, self.num_players).T
            # Fill played cards from current_trick in subsequent rows
            for i, (player, card) in enumerate(observation["current_trick"]):
                card_vec = [0.0, 0.0, 0.0, 0.0]
                suit_idx = suit_mapping[card[0]]
                card_vec[suit_idx] = float(card[1])
                pbs[i, player] = torch.tensor(card_vec, dtype=torch.float32)
            pbs = pbs.unsqueeze(0)  # Batch dimension
            
            # Convert hand to tensor of shape (1, rounds, 4)
            hand = observation["hand"]
            hand_list = []
            for card in hand:
                card_vec = [0.0, 0.0, 0.0, 0.0]
                suit_idx = suit_mapping[card[0]]
                card_vec[suit_idx] = float(card[1])
                hand_list.append(card_vec)
            hand_tensor = torch.tensor(hand_list, dtype=torch.float32).unsqueeze(0)
            return pbs, hand_tensor

    def bid(self, observation):
        """Use ReBel bidding network to determine bid using finalized input shape."""
        hand_tensor = self._process_observation(observation)  # shape: (1, hand_len, 4)
        with torch.no_grad():
            bid_output = self.bidding_network(hand_tensor)
        bid_value = max(0, min(int(bid_output.item()), observation["round_number"]))
        return bid_value

    def play_card(self, observation):
        """Use ReBel policy network to determine which card to play with finalized input."""
        pbs, hand_tensor = self._process_observation(observation)  # pbs: (1, rounds+1, num_players, 4) and hand: (1, rounds, 4)
        with torch.no_grad():
            probs = self.policy_network(pbs, hand_tensor)
        # Mask invalid moves: keep only as many probabilities as there are cards in hand
        hand_len = hand_tensor.shape[1]
        probs = probs[0][:hand_len]
        if probs.sum() == 0:
            return np.random.randint(hand_len)
        action = torch.multinomial(probs, 1).item()
        return action

    def act(self, observation, bidding_phase):
        """Main action selection method"""
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)
