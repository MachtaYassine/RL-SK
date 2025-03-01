import argparse
import torch
import os
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
from agent.NeuralAgent import LearningSkullKingAgent, PolicyNetwork
from agent.IntermediateAgent import IntermediateSkullKingAgent
from agent.AgressiveAgent import AggressiveSkullKingAgent
from agent.SimpleAgent import SkullKingAgent as SimpleSkullKingAgent
from main import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained neural agent against baseline agents"
    )
    parser.add_argument("--models_path", type=str, required=True,
                        help="Path to directory containing the model files")
    parser.add_argument("--n_eval_games", type=int, default=10,
                        help="Number of evaluation games to run")
    parser.add_argument("--num_players", type=int, default=5,
                        help="Number of players in evaluation games")
    parser.add_argument("--shared_networks", action="store_true",
                        help="Use shared networks among learning agents")
    parser.add_argument("--bid_hidden_dims", nargs="+", type=int, default=[128, 128, 128, 128],
                        help="Hidden dimensions for the bid network")
    parser.add_argument("--play_hidden_dims", nargs="+", type=int, default=[128, 128, 128, 128],
                        help="Hidden dimensions for the play network")
    parser.add_argument("--agents", nargs="+", type=str, required=True,
                        help="List of agent types to participate in the game, e.g., 'neural neural intermediate intermediate'")
    return parser.parse_args()

def load_neural_agent(args, num_players, logger):
    # Create a temporary environment to retrieve dimensions and properties.
    temp_env = SkullKingEnvNoSpecials(num_players=num_players, logger=logger)
    temp_props = temp_env.get_env_properties()

    hand_size = temp_props["hand_size"]
    suit_count = temp_props["suit_count"]
    max_rank = temp_props["max_rank"]
    max_players = temp_props["max_players"]
    bid_dimension_vars = temp_props["bid_dimension_vars"]
    play_dimension_vars = temp_props["play_dimension_vars"]

    # Calculate input dimensions (using the same formulas as in main.py)
    bid_input_dim = hand_size * 5 + 4 + max_players
    play_input_dim = hand_size * 5 + max_players * 5 + 3 + 1 + 1 + max_players + max_players + 1 + 5 + 2

    # Create networks with the same architecture as in training.
    bid_net = PolicyNetwork(bid_input_dim, output_dim=11, hidden_dims=args.bid_hidden_dims)
    play_net = PolicyNetwork(play_input_dim, output_dim=hand_size, hidden_dims=args.play_hidden_dims)
    hidden_dim = args.play_hidden_dims[-1] if args.play_hidden_dims else 128
    trick_win_predictor = torch.nn.Linear(hidden_dim, 11)

    # Load pre-trained weights
    bid_path = os.path.join(args.models_path, "LearningSkullKingAgent_bid_net.pth")
    play_path = os.path.join(args.models_path, "LearningSkullKingAgent_play_net.pth")
    trick_path = os.path.join(args.models_path, "LearningSkullKingAgent_trick_predictor.pth")

    logger.info(f"Loading models from {args.models_path}")
    bid_net.load_state_dict(torch.load(bid_path))
    play_net.load_state_dict(torch.load(play_path))
    trick_win_predictor.load_state_dict(torch.load(trick_path))

    bid_net.eval()
    play_net.eval()
    trick_win_predictor.eval()

    # Initialize the learning agent with the loaded shared networks.
    agent = LearningSkullKingAgent(
        num_players=num_players,
        hand_size=hand_size,
        shared_bid_net=bid_net if args.shared_networks else None,
        shared_play_net=play_net if args.shared_networks else None,
        shared_trick_win_predictor=trick_win_predictor if args.shared_networks else None,
        suit_count=suit_count,
        max_rank=max_rank,
        max_players=max_players,
        bid_dimension_vars=bid_dimension_vars,
        play_dimension_vars=play_dimension_vars,
        bid_hidden_dims=args.bid_hidden_dims,
        play_hidden_dims=args.play_hidden_dims
    )
    return agent

def create_agent(agent_type, total_players, args, logger):
    if agent_type == "neural":
        return load_neural_agent(args, total_players, logger)
    elif agent_type == "intermediate":
        return IntermediateSkullKingAgent(total_players)
    elif agent_type == "aggressive":
        return AggressiveSkullKingAgent(total_players)
    elif agent_type == "simple":
        return SimpleSkullKingAgent(total_players)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def run_evaluation(args):
    logs_dir = "logs/"
    logger = setup_logger(logs_dir, False)
    args.num_players = len(args.agents)
    total_players = args.num_players

    # Create agents based on the provided list
    eval_agents = [create_agent(agent_type, total_players, args, logger) for agent_type in args.agents]

    logger.info(f"Starting evaluation over {args.n_eval_games} games with {total_players} players")

    # Track indices of neural agents
    neural_indices = [i for i, agent_type in enumerate(args.agents) if agent_type == "neural"]

    neural_victories = 0
    neural_avg_score = 0
    all_agents_scores = [[] for _ in range(total_players)]
    agent_wins = defaultdict(int)
    neural_victory_margins = []
    neural_defeat_margins = []

    for game in range(args.n_eval_games):
        env = SkullKingEnvNoSpecials(num_players=total_players, logger=logger)
        obs = env.reset()
        done = False

        while not done:
            current_agent = eval_agents[env.current_player]
            if env.bidding_phase:
                action = current_agent.bid(obs)
            else:
                action = current_agent.play_card(obs)
            obs, reward, done, info = env.step(action)

        for i in range(total_players):
            all_agents_scores[i].append(env.total_scores[i])

        # Calculate the average score for neural agents
        for idx in neural_indices:
            neural_avg_score += env.total_scores[idx]

        winner_idx = max(range(total_players), key=lambda i: env.total_scores[i])
        winner_type = args.agents[winner_idx]
        agent_wins[winner_type] += 1

        # Check if any neural agent won
        if winner_idx in neural_indices:
            neural_victories += 1
            # Calculate victory margin for the winning neural agent
            margin = env.total_scores[winner_idx] - max(score for j, score in enumerate(env.total_scores) if j != winner_idx)
            neural_victory_margins.append(margin)
        else:
            # Calculate defeat margin for each neural agent
            for idx in neural_indices:
                margin = env.total_scores[idx] - env.total_scores[winner_idx]
                neural_defeat_margins.append(margin)

        logger.info(f"Game {game+1}: Winner is {winner_type.capitalize()} (Player {winner_idx}) with score {env.total_scores[winner_idx]}")
        logger.info(f"Scores: {env.total_scores}")

    neural_avg_score /= (args.n_eval_games * len(neural_indices)) if len(neural_indices)>0 else 1
    neural_win_rate = (neural_victories / args.n_eval_games) * 100 if len(neural_indices)>0 else 0

    logger.info("=== Evaluation Results ===")
    logger.info(f"Neural agent win rate: {neural_win_rate:.1f}%")
    logger.info(f"Neural agent average score: {neural_avg_score:.1f}")

    avg_scores = {agent_type: sum(all_agents_scores[i]) / len(all_agents_scores[i]) for i, agent_type in enumerate(args.agents)}

    logger.info("Average scores per agent type:")
    for agent_type, avg_score in avg_scores.items():
        logger.info(f"  {agent_type.capitalize()}: {avg_score:.1f}")

    plot_evaluation(all_agents_scores, neural_win_rate, avg_scores, agent_wins, neural_victory_margins, neural_defeat_margins, logs_dir, args)


def plot_evaluation(all_scores, neural_win_rate, avg_scores, agent_wins, neural_victory_margins, neural_defeat_margins, logs_dir, args):
    agent_types = args.agents
    config_description = "_".join(agent_types)  # Create a description of the configuration
    logs_dir = os.path.join(logs_dir, config_description)
    os.makedirs(logs_dir, exist_ok=True)

    # Bar chart: Average scores per agent type
    plt.figure(figsize=(8, 6))
    plt.bar(agent_types, [avg_scores[agent_type] for agent_type in agent_types], color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.ylabel("Average Score")
    plt.title(f"Average Game Score per Agent Type ({config_description})")
    plt.savefig(os.path.join(logs_dir, f"avg_scores_bar_{config_description}.png"))
    plt.close()

    # Pie chart: Win Distribution
    plt.figure(figsize=(6, 6))
    plt.pie(agent_wins.values(), labels=agent_wins.keys(), autopct="%1.1f%%", startangle=90, colors=['blue', 'orange', 'green', 'red', 'purple'])
    plt.title(f"Win Distribution Among Agent Types ({config_description})")
    plt.savefig(os.path.join(logs_dir, f"win_distribution_pie_{config_description}.png"))
    plt.close()

    # Histogram: Neural Victory Margins
    plt.figure(figsize=(8, 6))
    if neural_victory_margins:
        plt.hist(neural_victory_margins, bins=10, color='blue', alpha=0.7)
        plt.xlabel("Victory Margin")
        plt.ylabel("Frequency")
        plt.title(f"Neural Agent Victory Margins ({config_description})")
        plt.savefig(os.path.join(logs_dir, f"neural_victory_margins_hist_{config_description}.png"))
        plt.close()

    # Histogram: Neural Defeat Margins
    plt.figure(figsize=(8, 6))
    if neural_defeat_margins:
        plt.hist(neural_defeat_margins, bins=10, color='red', alpha=0.7)
        plt.xlabel("Defeat Margin")
        plt.ylabel("Frequency")
        plt.title(f"Neural Agent Defeat Margins ({config_description})")
        plt.savefig(os.path.join(logs_dir, f"neural_defeat_margins_hist_{config_description}.png"))
        plt.close()

    # Scatter Plot: Neural Score vs. Opponents' Best Score
    neural_scores = all_scores[0]
    opponents_best = [max(all_scores[j][i] for j in range(1, len(all_scores))) for i in range(len(neural_scores))]
    plt.figure(figsize=(8, 6))
    plt.scatter(neural_scores, opponents_best, color='purple', alpha=0.7)
    plt.xlabel("Neural Agent Score")
    plt.ylabel("Best Opponent Score")
    plt.title(f"Neural Score vs. Best Opponent Score per Game ({config_description})")
    plt.plot([min(neural_scores), max(neural_scores)], [min(neural_scores), max(neural_scores)], 'k--', label="Equal Score")
    plt.legend()
    plt.savefig(os.path.join(logs_dir, f"neural_vs_opponent_scatter_{config_description}.png"))
    plt.close()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    # Title with larger font size
    fig.suptitle(f"Evaluation Results for {args.n_eval_games} Games with Configuration: {config_description}", fontsize=28, fontweight='bold')

    # Bar chart: Average scores per agent type
    axes[0].bar(agent_types, [avg_scores[agent_type] for agent_type in agent_types], color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[0].set_ylabel("Average Score", fontsize=26)
    axes[0].set_title("Average Game Score per Agent Type", fontsize=30)
    axes[0].tick_params(axis='both', which='major', labelsize=26)

    # Pie chart: Win Distribution
    axes[1].pie(agent_wins.values(), labels=agent_wins.keys(), autopct="%1.1f%%", startangle=90, colors=['blue', 'orange', 'green', 'red', 'purple'],textprops={'fontsize': 26})
    axes[1].set_title("Win Distribution Among Agent Types", fontsize=30)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ensures the title is not cut off

    # Save the figure
    plt.savefig(os.path.join(logs_dir, f"evaluation_results_{config_description}.png"))
    plt.close()
    
    
    
if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
