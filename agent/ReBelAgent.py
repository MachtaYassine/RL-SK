import numpy as np
import networkx as nx
from .SimpleAgent import SkullKingAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


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

    def __init__(self, max_players=5, max_rounds=10, hidden_dim=128, lr=0.001):
        super(ReBelValueNN, self).__init__()
        self.max_players = max_players
        self.max_rounds = max_rounds
        self.hidden_dim = hidden_dim

        # Compute input and output dimensions
        self.input_dim = self.max_players * (
            2 * self.max_rounds + 1
        )  # (11*5) + (5*10) = 105
        self.output_dim = self.max_players  # (1*5)

        # Define a simple feedforward network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, pbs, set_of_hands):
        """
        Forward pass to estimate player scores.

        Args:
            pbs (torch.Tensor): Shape (n_rounds+1, n_players).
            set_of_hands (torch.Tensor): Shape (n_rounds, n_players).

        Returns:
            torch.Tensor: Estimated scores of shape (1, max_players).
        """
        n_rounds, n_players = set_of_hands.shape  # Transposed format: (rounds, players)

        # Pad/mask pbs to size (max_rounds+1, max_players)
        padded_pbs = torch.full((self.max_rounds + 1, self.max_players), float("-inf"))
        padded_pbs[: n_rounds + 1, :n_players] = pbs

        # Pad/mask set_of_hands to size (max_rounds, max_players)
        padded_hands = torch.full((self.max_rounds, self.max_players), float("-inf"))
        padded_hands[:n_rounds, :n_players] = set_of_hands

        # Flatten and concatenate
        x = torch.cat(
            (padded_pbs.flatten(), padded_hands.flatten()), dim=0
        )  # Shape: (input_dim,)

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # TODO : Maybe will need to set the extra values to zero, we'll see
        return x  # Shape: (1, max_players)

    def train(self, training_dict, epochs=10, batch_size=32):
        dataset = TupleDataset(training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for pbs, set_of_hands, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self(pbs, set_of_hands)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()


class ReBelPolicyNN(nn.Module):
    """
    Estimates the policy (probabilities of playing each card) given a public belief state (pbs) and the player's hand.

    Args:
        max_players (int): Maximum number of players the model should support (default=5).
        max_rounds (int): Maximum number of rounds the model should support (default=10).
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

    def __init__(self, max_players=5, max_rounds=10, hidden_dim=128, lr=0.001):
        super(ReBelPolicyNN, self).__init__()
        self.max_players = max_players
        self.max_rounds = max_rounds
        self.hidden_dim = hidden_dim

        # Compute input and output dimensions
        self.input_dim = (
            2 * self.max_rounds + 1
        ) * self.max_players + self.max_rounds  # (11*5) + 10 = 115
        self.output_dim = self.max_rounds  # Probabilities over each card in hand (10)

        # Define a feedforward network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pbs, hand):
        """
        Forward pass to estimate action probabilities.

        Args:
            pbs (torch.Tensor): Shape (n_rounds+1, n_players).
            hand (torch.Tensor): Shape (n_rounds,).

        Returns:
            torch.Tensor: Probability distribution over the hand (shape: max_rounds, 1).
        """
        # TODO : Make sure that the cards already played are masked
        # TODO : Mask the cards that can't be played
        n_rounds, n_players = pbs.shape[0] - 1, pbs.shape[1]

        # Pad/mask pbs to size (max_rounds+1, max_players)
        padded_pbs = torch.full((self.max_rounds + 1, self.max_players), float("-inf"))
        padded_pbs[: n_rounds + 1, :n_players] = pbs

        # Pad/mask hand to size (max_rounds, 1)
        padded_hand = torch.full((self.max_rounds, 1), float("-inf"))
        padded_hand[:n_rounds, 0] = hand

        # Flatten and concatenate
        x = torch.cat(
            (padded_pbs.flatten(), padded_hand.flatten()), dim=0
        )  # Shape: (input_dim,)

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply softmax to output probabilities
        return F.softmax(x, dim=0)  # Shape: (max_rounds,)

    def train(self, training_dict, epochs=10, batch_size=32):
        dataset = TupleDataset(training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for pbs, hand, probas in dataloader:
                self.optimizer.zero_grad()
                outputs = self(pbs, hand)
                loss = self.loss_fn(outputs, probas)
                loss.backward()
                self.optimizer.step()


class ReBelBidNN(nn.Module):
    """
    Estimates how many tricks a player will bid based on their hand.

    Args:
        max_rounds (int): Maximum number of rounds (default=10).
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

    def __init__(self, max_rounds=10, hidden_dim=128, lr=0.001):
        super(ReBelBidNN, self).__init__()
        self.max_rounds = max_rounds
        self.hidden_dim = hidden_dim

        # Define the feedforward network
        self.fc1 = nn.Linear(self.max_rounds, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, 1)  # Output is a single value

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, hand):
        """
        Forward pass to estimate the number of tricks the player will bid.

        Args:
            hand (torch.Tensor): Shape (n_rounds,).

        Returns:
            torch.Tensor: A single estimated bid value (shape: (1,)).
        """
        n_rounds = hand.shape[0]

        # Pad/mask hand to size (max_rounds,)
        padded_hand = torch.full((self.max_rounds,), float("-inf"))
        padded_hand[:n_rounds] = hand

        # Forward pass through the network
        x = F.relu(self.fc1(padded_hand))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # TODO : Make sure the output is less than n_rounds
        return x  # Output a single scalar value

    def train(self, training_dict, epochs=10, batch_size=32):
        # Process training dict to extract individual hands
        processed_training_dict = {
            (hand,): tricks[idx]
            for hands, tricks in training_dict.items()
            for idx, hand in enumerate(hands)
        }

        dataset = TupleDataset(processed_training_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for (hand,), bid in dataloader:
                self.optimizer.zero_grad()
                outputs = self(hand)
                loss = self.loss_fn(outputs, bid)
                loss.backward()
                self.optimizer.step()


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
        simu_depth=2,
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

        for set_of_hands in sets_of_hands:
            # Get bid of each player
            bids = self.get_bids(set_of_hands)
            # Sample N public states at random from the hands as a starting point
            set_of_pbs = self.get_initial_pbs(set_of_hands, bids)
            value_network_training_dict, policy_network_training_dict = {}, {}

            # Now iterate through these public states and run the ReBel algorithm
            for pbs in set_of_pbs:
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
                    for t in range(self.T):
                        if t == t_sample:
                            new_pbs = self._sample_leaf(G)
                        G = self._update_policy(
                            G, set_of_hands
                        )  # Weights now represent \pi^{t}
                        rolling_mean_policy.weights = (
                            t / t + 1
                        ) * rolling_mean_policy.weights + (
                            1 / t + 1
                        ) * G.weights  # Update average policy
                        expected_value = (t / t + 1) * expected_value + (
                            1 / t + 1
                        ) * self._get_expected_value(
                            G, bids, set_of_hands
                        )  # Update expected value
                    value_network_training_dict[(pbs, set_of_hands)] = expected_value
                    for state in rolling_mean_policy.node:
                        mask = pbs[1:] == -1
                        _, player = torch.nonzero(mask, as_tuple=True)[
                            0
                        ]  # Get the first -1
                        player = player.item()
                        player_hand = set_of_hands[player]
                        probas = torch.tensor(
                            [
                                G[tuple(map(tuple, pbs.numpy()))][child]["weight"]
                                for child in G.successors(
                                    tuple(map(tuple, pbs.numpy()))
                                )
                            ]
                        )  # Weights of the edges originating from state in a tensor
                        policy_network_training_dict[(state, player_hand)] = probas
                    pbs = new_pbs
            self.value_network.train(
                value_network_training_dict
            )  # Update value network
            self.policy_network.train(
                policy_network_training_dict
            )  # Update policy network
            self.warm_start = True if np.rand.rand() > 0.5 else False
            bidding_network_training_dict[set_of_hands] = self._get_average_tricks_won(
                value_network_training_dict
            )
        self.bidding_network.train(
            bidding_network_training_dict
        )  # Update bidding network

    def get_sets_of_hands(self):
        """
        Simulate the dealing of n_rounds cards from the deck to the n_players K times.

        Returns:
            sets_of_hands (numpy.ndarray): Sets of hands of the form array([[[hand_1], ...,  [hand_n_players]] * K])
        """
        sets_of_hands = []
        for _ in range(self.K):
            np.random.shuffle(self.deck)
            hands = np.array_split(
                self.deck[: self.n_players * self.n_rounds], self.n_players
            )
            sets_of_hands.append(hands)
        return np.array(sets_of_hands)

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
            [self.bidding_network(hand).unsqueeze(0) for hand in set_of_hand]
        )

        return bids  # Shape: (n_players, 1)

    def get_initial_pbs(self, set_of_hand, bids):
        """
        Simulates N possible states that can be reached given the players' cards.
        A public state contains all the cards that have been played and the players' bids.

        Args:
            set_of_hand (numpy.ndarray): Shape (n_players, n_rounds), contains the hand of each player.
            bids (torch.Tensor): Shape (n_players, 1), containing the bid of each player.

        Returns:
            set_of_pbs (torch.Tensor): Tensor of shape (N, (n_rounds+1) * n_players),
                                    representing possible public states.
        """

        n_players, n_rounds = set_of_hand.shape

        # Convert hands and bids to tensors
        set_of_hand = torch.tensor(set_of_hand, dtype=torch.float32)
        bids = bids.float().view(n_players)  # Flatten bids

        set_of_pbs = []

        for _ in range(self.N):
            # Initialize public state with -1 for unplayed moves
            pbs = torch.full((n_rounds + 1, n_players), -1.0)

            # Fill in bids (first row)
            pbs[0, :n_players] = bids

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

                # Find the available cards for this player
                available_cards = hands_copy[current_player][
                    hands_copy[current_player] != -1
                ]

                if len(available_cards) > 0:
                    chosen_card = np.random.choice(
                        available_cards.numpy()
                    )  # Pick a random card
                    played_cards.append((current_player, chosen_card))

                    # Remove the card from the player's hand
                    hands_copy[
                        current_player, hands_copy[current_player] == chosen_card
                    ] = -1

            # Fill the played cards into the public state matrix
            round_idx = 1
            for i, (player, card) in enumerate(played_cards):
                pbs[round_idx, player] = card
                if (
                    i + 1
                ) % n_players == 0:  # Move to the next round after all players played
                    round_idx += 1

            set_of_pbs.append(pbs.flatten())  # Flatten to match network input format

        return torch.stack(set_of_pbs)  # Shape: (N, (n_rounds+1) * n_players)

    def _is_terminal(self, pbs):
        """
        Return True if starting from the pbs there are fewer than simu_depth tricks left to play.

        Args:
            pbs (torch.tensor): Public state of shape ((n_rounds + 1), n_players).

        Returns:
            bool: True if there are fewer than simu_depth tricks left, False otherwise.
        """

        # Exclude the first row (bids) and count the number of fully unplayed tricks
        tricks_played = torch.sum(
            (pbs[1:] != -1).any(dim=1)
        )  # Count rows with at least one played card
        total_tricks = pbs.shape[0] - 1  # Total number of tricks (excluding bids row)

        tricks_left = total_tricks - tricks_played

        return tricks_left < self.simu_depth

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
        G.add_node(tuple(map(tuple, pbs.numpy())))  # Convert tensor to a hashable type

        d = 0  # Depth (number of full tricks played)

        while d < self.simu_depth:
            for _ in range(
                set_of_hands.shape[0]
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

                    trick_idx, current_player = torch.nonzero(mask, as_tuple=True)[
                        0
                    ]  # Get first empty trick slot
                    current_player = current_player.item()

                    # Get the player's remaining hand
                    played_cards = set(current_pbs[1:].flatten().tolist()) - {
                        -1
                    }  # Exclude -1 (unplayed)
                    current_hand = [
                        card
                        for card in set_of_hands[current_player]
                        if card not in played_cards
                    ]

                    if not current_hand:
                        continue  # No cards left to play

                    # Get probabilities for each card
                    if self.warm_start:
                        hand_tensor = torch.tensor(
                            current_hand, dtype=torch.float32
                        ).view(1, -1)
                        proba = (
                            self.policy_network(hand_tensor).softmax(dim=1).squeeze(0)
                        )
                    else:
                        proba = torch.full((len(current_hand),), 1 / len(current_hand))

                    # Generate new nodes for each possible card play
                    for card, prob in zip(current_hand, proba.tolist()):
                        new_pbs = current_pbs.clone()
                        new_pbs[trick_idx + 1, current_player] = (
                            card  # Update trick row for this player
                        )

                        new_pbs_tuple = tuple(map(tuple, new_pbs.numpy()))

                        if new_pbs_tuple not in G:
                            G.add_node(new_pbs_tuple)
                            new_nodes.append(new_pbs_tuple)

                        # Add edge with probability weight
                        G.add_edge(leaf, new_pbs_tuple, weight=prob)

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
            set_of_hands.shape[0]
        )  # Initialize expected values for all players

        # Get leaf nodes of the tree
        leaves = [node for node in G.nodes if G.out_degree(node) == 0]

        for leaf in leaves:
            pbs = torch.tensor(leaf)  # Convert leaf (public state) to a tensor

            # Check if the game is fully played (no -1 present in the tricks)
            if (pbs[1:] == -1).sum() == 0:
                values = self._compute_final_scores(pbs, bids)  # Compute exact scores
            else:
                values = self.value_network(
                    torch.cat((pbs.flatten(), set_of_hands.flatten())).unsqueeze(0)
                )  # Estimate scores

            # Compute probability of reaching this leaf
            proba = 1.0
            path = list(nx.shortest_path(G, source=list(G.nodes)[0], target=leaf))
            for i in range(len(path) - 1):
                proba *= G[path[i]][path[i + 1]]["weight"]

            expected_value += values.squeeze() * proba  # Accumulate weighted values

        return expected_value

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
        n_players = bids.shape[0]
        n_rounds = pbs.shape[0] - 1  # First row of pbs is bids
        atout_suit = 3  # The 4th index (0-based) is the atout suit

        tricks_won = torch.zeros(n_players, dtype=torch.int32)

        for t in range(1, n_rounds + 1):  # Iterate over tricks (excluding bid row)
            trick = pbs[t]
            lead_suit = None
            highest_card = -1
            winner = None

            for i in range(n_players):
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

        scores = torch.zeros(n_players, dtype=torch.int32)

        for i in range(n_players):
            bid = bids[i].item()
            trick_won = tricks_won[i].item()

            if bid == 0:
                if trick_won == 0:
                    scores[i] = 10 * n_rounds
                else:
                    scores[i] = -10 * n_rounds
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
            path = list(
                nx.shortest_path(G, source=list(G.nodes)[0], target=leaf)
            )  # Get path from root to leaf

            for i in range(len(path) - 1):
                proba *= G[path[i]][path[i + 1]][
                    "weight"
                ]  # Multiply probabilities along the path

            if proba > max_proba:
                max_proba = proba
                best_leaf = leaf

        return torch.tensor(
            best_leaf
        )  # Return the public state with highest probability

    def _update_policy(self, G, set_of_hands):
        """
        Update the policy, hence the weights of G, using the CFR algorithm.

        Args:
            G (nx.Graph): Weighted exploration tree.
            set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]]).

        Returns:
            updated_G (nx.Graph)
        """
        subgame_dict = {}  # Stores expected values of subgames {pbs: expected_value}
        regret_dict = {}  # Stores regret values {pbs: {edge: regret}}

        # Compute regret for each edge in the graph
        for edge in G.edges:
            parent_pbs, child_pbs = (
                edge  # The edge represents a move from parent_pbs to child_pbs
            )

            if parent_pbs not in subgame_dict:
                subgame_dict[parent_pbs] = self._get_expected_value(
                    G.subgraph(nx.descendants(G, parent_pbs)),
                    self.value_network,
                    set_of_hands,
                )

            if child_pbs not in subgame_dict:
                subgame_dict[child_pbs] = self._get_expected_value(
                    G.subgraph(nx.descendants(G, child_pbs)),
                    self.value_network,
                    set_of_hands,
                )

            player = torch.where(parent_pbs == -1)[1][
                0
            ].item()  # Identify the player who played the move

            # Compute regret: max(0, value of taking action - value of parent state)
            regret = max(
                0, subgame_dict[child_pbs][player] - subgame_dict[parent_pbs][player]
            )

            if parent_pbs not in regret_dict:
                regret_dict[parent_pbs] = {}
            regret_dict[parent_pbs][edge] = regret

        # Compute new policy by normalizing regrets into probabilities
        for pbs, edges in regret_dict.items():
            total_regret = sum(edges.values())
            if total_regret > 0:
                for edge, regret in edges.items():
                    G[edge[0]][edge[1]]["weight"] = regret / total_regret
            else:
                # If all regrets are zero, use uniform probability
                num_edges = len(edges)
                for edge in edges:
                    G[edge[0]][edge[1]]["weight"] = 1 / num_edges

        return G

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
        ):  # value is a tensor of shape (n_players,)
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

    def train(self, n_rounds=10, K=100, N=50, T=1000, simu_depth=2):
        """
        Train the agent using the ReBel algorithm with vectorized cards.

        Args:
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
            n_players=self.num_players,
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
        """Convert observation into tensor format for ReBel networks"""
        # For bidding: convert hand to one-hot encoding
        suit_mapping = {
            "Parrot": 0,
            "Treasure Chest": 1,
            "Treasure Map": 2,
            "Jolly Roger": 3,
        }

        if observation["bidding_phase"]:
            # Process for bidding network (hand only)
            hand = observation["hand"]
            hand_tensor = torch.zeros(
                (self.num_players, 4, 14)
            )  # (players, suits, ranks)
            for card in hand:
                suit_idx = suit_mapping[card[0]]
                rank_idx = card[1] - 1
                hand_tensor[0, suit_idx, rank_idx] = 1
            return hand_tensor.flatten()
        else:
            # Process for policy network (hand + public belief state)
            pbs = torch.zeros((observation["round_number"] + 1, self.num_players))
            # Fill bids
            pbs[0] = torch.tensor(observation["all_bids"])
            # Fill played cards
            for i, (player, card) in enumerate(observation["current_trick"]):
                if card[0] in suit_mapping:
                    pbs[i + 1, player] = card[1] + suit_mapping[card[0]] * 14

            hand_tensor = torch.zeros(observation["round_number"])
            for i, card in enumerate(observation["hand"]):
                hand_tensor[i] = card[1] + suit_mapping[card[0]] * 14

            return pbs, hand_tensor

    def bid(self, observation):
        """Use ReBel bidding network to determine bid"""
        hand_tensor = self._process_observation(observation)
        with torch.no_grad():
            bid = self.bidding_network(hand_tensor)
        bid_value = max(0, min(int(bid.item()), observation["round_number"]))
        return bid_value

    def play_card(self, observation):
        """Use ReBel policy network to determine which card to play"""
        pbs, hand_tensor = self._process_observation(observation)
        with torch.no_grad():
            probs = self.policy_network(pbs, hand_tensor)

        # Mask invalid moves
        valid_moves = torch.zeros(len(observation["hand"]))
        for i in range(len(observation["hand"])):
            valid_moves[i] = 1
        probs = probs[: len(observation["hand"])] * valid_moves

        if probs.sum() == 0:
            return np.random.randint(len(observation["hand"]))

        probs = probs / probs.sum()
        action = torch.multinomial(probs, 1).item()
        return action

    def act(self, observation, bidding_phase):
        """Main action selection method"""
        if bidding_phase:
            return self.bid(observation)
        else:
            return self.play_card(observation)
