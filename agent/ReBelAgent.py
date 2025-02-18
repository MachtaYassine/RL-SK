import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .SimpleAgent import SkullKingAgent

class ReBelValueNN(nn.Module):
    """
    Estimates the score obtained by each player from a pbs (bids, current state of the trick) and the hands of each player.

    TODO : The sizes are bids(n_players), state(n_round*n_players), hands(n_players*n_round)
    Make sure the input are masked to be of consistent size no matter the number of players and the number of rounds

    """
    def __init__(self):
        super(ReBelValueNN, self).__init__()
        pass

    def forward(self, x):
        pass

class ReBelPolicyNN(nn.Module):
    """
    Estimates the policy (move to play) given a pbs and the hand of the player

    TODO : The sizes are pbs((n_round + 1)*n_players), hand(n_round)
    Make sure the input are masked to be of consistent size no matter the number of players and the number of rounds
    """
    def __init__(self):
        super(ReBelPolicyNN, self).__init__()
        # Initialize to uniform
        pass

    def forward(self, x):
        pass

class ReBelBidNN(nn.Module):
    """
    Estimates how many tricks each player is gonna bid given his cards

    TODO : Masking for size
    """
    def __init__(self):
        super(ReBelBidNN, self).__init__()
        pass

    def forward(self, x):
        pass

def rebel_loop(
    n_players: int,
    n_rounds: int,
    K: int,
    N: int,
    T: int,
    value_network: ReBelValueNN,
    policy_network: ReBelPolicyNN,
    bidding_network: ReBelBidNN,
    deck,
    simu_depth: int = 2,
    warm_start: bool = False,
):
    """
    Adaptation of the ReBel algorithm presented in 'Combining Deep Reinforcement Learning and Search for Imperfect-Information Games'

    Args:
        n_players (int): number of players in game.
        n_rounds (int): Number of rounds in this game.
        K (int): Number of sets of hands that will be sampled.
        N (int): Number of possible public state from a set of hands that will be sampled.
        T (int): Number of iteration for updating the policy.
        value_network (ReBelValueNN): NN estimating the expected value of each player given a public state.
        policy_network (ReBelPolicyNN): NN that determines a strategy for playing a hand.
        bidding_network (ReBelBidNN): NN that determines a bidding strategy.
        deck : All available cards.
        simu_depth (int): Depth of exploration for the ReBel algorithm.
        warm_start (bool): Whether to use the policy network to initialize the exploration tree.
    """
    # We start by sampling K sets of hands from the deck
    sets_of_hands = get_sets_of_hands(n_players, n_rounds, K, deck)
    bidding_network_training_dict = {}

    for set_of_hands in sets_of_hands:
        # Get bid of each player
        bids = get_bids(set_of_hands, bidding_network)
        # Sample N public states at random from the hands as a starting point 
        set_of_pbs = get_initial_pbs(set_of_hands, bids, N)
        value_network_training_dict, policy_network_training_dict = {}, {}

        # Now iterate through these public states and run the ReBel algorithm
        for pbs in set_of_pbs:
            # Begin iterative exploration until reaching the end of the game
            while not is_terminal(pbs, simu_depth):
                # Get the exploration tree
                G = initialize_tree(pbs, set_of_hands, policy_network, warm_start)
                rolling_mean_policy = G.copy() # We will update the weights on this graph as a way of representing a policy
                # Initial value associated to the pbs
                expected_value = get_expected_value(G, bids, value_network)
                t_sample = None # Proba on [0, ..., T] growing with t 
                for t in range(T):
                    if t == t_sample:
                        new_pbs = sample_leaf(G)
                    G = update_policy(G) # Weights now represent \pi^{t}
                    rolling_mean_policy.weights = (t/t+1) * rolling_mean_policy.weights + (1/t+1) * G.weights # Update average policy
                    expected_value = (t/t+1) * expected_value + (1/t+1) * get_expected_value(G, bids, value_network) # Update expected value
                value_network_training_dict[torch.cat(pbs, set_of_hands)] = expected_value
                for state in rolling_mean_policy.node:
                    player = None # Deduce these from the first inf in state
                    player_hand = set_of_hands[player]
                    probas = None # Weights of the edges originating from state in a tensor
                    info = torch.cat(state, player_hand)
                    policy_network_training_dict[info] = probas
                pbs = new_pbs
        value_network.train(value_network_training_dict) # Update value network
        policy_network.train(policy_network_training_dict) # Update policy network
        warm_start = True if np.rand.rand() > 0.5 else False
        bidding_network_training_dict[set_of_hands] = get_average_tricks_won(value_network_training_dict)
    bidding_network.train(bidding_network_training_dict) # Update bidding network

        
                




def get_sets_of_hands(n_players, n_rounds, K, deck):
    """
    Simulate the dealing of n_rounds cards from the deck to the n_players K times.

    Args:
        n_players (int): number of players in game.
        n_rounds (int): Number of rounds in this game.
        K (int): Number of sets of hands that will be sampled.
        deck : All available cards.

    Returns:
        sets_of_hands (numpy.ndarray): Sets of hands of the form array([[[hand_1], ...,  [hand_n_players]] * K])
    """
    return None

def get_bids(set_of_hand, bidding_network):
    """
    Determine a bid for each player given their hands.

    Args:
        set_of_hand (numpy.ndarray): Contains the hand of each player.
        bidding_network (ReBelBidNN): NN that determines a bidding strategy.

    Returns:
        bids (torch.tensor): Tensor containing the bid of each player (size n_players * 1)
    """
    # Transform set_of_hand to a batch to be processed by the bidding network

    return None

def get_initial_pbs(set_of_hand, bids, N):
    """
    Simulates N possible states that can be reached given the players' cards.
    A public state contains all the cards that have been played and the players' bids.

    Args:
        set_of_hand (numpy.ndarray): Contains the hand of each player.
        bids (torch.tensor): Tensor containing the bid of each player (size n_players * 1).
        N (int): Number of possible public state from a set of hands that will be sampled.

    Returns:
        set_of_pbs (list): List of public states.

    Example of a public state with 2 players with 3 cards each where they bid 0 and 2 and 3 cards where played:
    torc.tensor([
        [0, 2] # Bids
        [4, 5] # Cards played on the first trick
        [8, inf] # Only the first player played in the second trick at this point
        [inf, inf] # Remaining trick that hasn't been observed
    ]) -> This doesn't represent the masking for size coherence for the networks
    """
    return None

def is_terminal(pbs, simu_depth):
    """
    Return True if starting from the pbs they are less than simu_depth tricks to play.

    Args:
        pbs (torch.tensor): Public state.
        simu_depth (int): Depth of exploration.
    """
    return None

def initialize_tree(pbs, simu_depth, set_of_hands, policy_network, warm_start):
    """
    Creates the exploration tree with a starting public state. The nodes are public states
    and the edges are weighted with the policy probability.

    Args:
        pbs (torch.tensor): Public state.
        simu_depth (int): Depth of exploration.
        set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]])
        policy_network (ReBelPolicyNN): NN that determines a strategy for playing a hand.
        warm_start (bool): If False the policy is random.

    Returns:
        G (nx.Graph): Weighted exploration tree.
    """
    # Determine which player gets to play first from the public state
    current_player = None # Index along second dim of the first inf in pbs
    # Determine the current hands fo the players from the public state
    current_set_of_hands = None # Remove cards already played by the players
    # Initialize the graph with pbs as root
    G = None
    d = 0
    while d < simu_depth:
        while current_player < set_of_hands.shape[1]:
            for card in current_set_of_hands[current_player]:
                for leaf in G.leaves:
                    proba = None # Get the proba of playing this card from the policy network or 1/n_cards if not warm_start
                    # Create a node with the new public state after card was played and add the edge with weight proba
            current_player += 1
        current_player = 0
        d += 1

def get_expected_value(G, bids, value_network, set_of_hands):
    """
    Compute the average value of each player starting from the root of G.

    Args:
        G (nx.Graph): Weighted exploration tree.
        bids (torch.tensor): Tensor containing the bid of each player (size n_players * 1).
        value_network (ReBelValueNN): NN estimating the expected value of each player given a public state.
        set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]]).

    Returns:
        expected_value (torch.tensor): Tensor containing the scores of every players.
    """
    # Iterate on the leaves of graph and for each estimate/compute the scores and proba to reach it
    expected_value = 0
    for pbs in G.leaves:
        if inf not in pbs: # This means that all cards were played
            values = None # Compute the scores from the tricks won by each player in the pbs and his bid
        else: # Estimate using the value network
            values = value_network(torch.cat(pbs, set_of_hands))
        proba = None # Multiply the weights that lead to pbs
        expected_value += values * proba
    return expected_value

def sample_leaf(G):
    """
    Sample the pbs at the leaf of G with the highest probabilty of happening.

    Args:
        G (nx.Graph): Weighted exploration tree.

    Returns:
        pbs (torch.tensor): Public state.
    """
    return None


def update_policy(G, bids, value_network, set_of_hands):
    """
    Update the policy, hence the weights of G, using CFR algorithm.

    Args:
        G (nx.Graph): Weighted exploration tree.
        bids (torch.tensor): Tensor containing the bid of each player (size n_players * 1).
        value_network (ReBelValueNN): NN estimating the expected value of each player given a public state.
        set_of_hands (numpy.ndarray): Set of hands of the form array([[hand_1], ...,  [hand_n_players]]).

    Returns:
        updated_G (nx.Graph)
    """
    subgame_dict = {} # Key: pbs, Value: expected value
    regret_dict = {} # {pbs : {edge : regret}}
    for edge in G.edges:
        original_pbs = None # Determine what was the public state before the action
        subgame_expected_value = None # Subgraph that originate from original_pbs # Keep in a dict
        player = None # Determine the player that played the action the edge corresponds to
        subgame_with_action = None # Subgraph resulting from the player making the action
        regret = max(
            0, get_expected_value(subgame_with_action, bids, value_network, set_of_hands)[player] - subgame_expected_value[player]
        )
        regret_dict[original_pbs][edge] = regret
    # Compute new policy 
    for pbs in regret_dict.keys():
        total_regret = sum(regret_dict[pbs].values())
        for edge in G.edges(pbs): # Edges originated from pbs
            edge.weight = regret_dict[pbs][edge] / total_regret
    return G


def get_average_tricks_won(value_network_training_dict):
    """
    From the value saved during the ReBel loop, deduce the average number of tricks won by each hand
    """
    # Iterate through the values in the dict
    # If positive divide by 20 
    # If negative
    #   If value != -10 * n_round, just divide by -10
    #   Else it means they bid 0 and won at least 1 trick so just set to 1 trick
    return None

