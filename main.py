import argparse
import importlib
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import os  # NEW: for file system operations
import matplotlib.pyplot as plt  # NEW: for plotting
import threading               # NEW: for running live server
import http.server             # NEW: for live server
import socketserver            # NEW: for live server
from torch.utils.tensorboard import SummaryWriter  # NEW: for TensorBoard logging
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
from agent.HumanAgent import HumanAgent

# Define a custom logger that accepts a "color" parameter, which can be a name.
class CustomLogger(logging.Logger):
    COLOR_MAP = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "reset": "\033[0m"
    }
    
    def _apply_color(self, msg, color):
        if color in self.COLOR_MAP:
            return f"{self.COLOR_MAP[color]}{msg}{self.COLOR_MAP['reset']}"
        # Otherwise assume it's a valid ANSI code.
        return f"{color}{msg}{self.COLOR_MAP['reset']}"
    
    def debug(self, msg, color=None, *args, **kwargs):
        if color:
            msg = self._apply_color(msg, color)
        super().debug(msg, *args, **kwargs)

    def info(self, msg, color=None, *args, **kwargs):
        if color:
            msg = self._apply_color(msg, color)
        super().info(msg, *args, **kwargs)
    
    def warning(self, msg, color=None, *args, **kwargs):
        if color:
            msg = self._apply_color(msg, color)
        super().warning(msg, *args, **kwargs)
    
    def error(self, msg, color=None, *args, **kwargs):
        if color:
            msg = self._apply_color(msg, color)
        super().error(msg, *args, **kwargs)

# Set our custom logger as the default logger class.
logging.setLoggerClass(CustomLogger)

# Custom colored formatter for logging
class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': "\033[36m",  # Cyan
        'INFO': "\033[32m",   # Green
        'WARNING': "\033[33m",# Yellow
        'ERROR': "\033[31m",  # Red
        'CRITICAL': "\033[41m"  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

# Import the environment
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials

# Mapping from a string to an agent class path.
# The agent classes must be accessible in the PYTHONPATH.
AGENT_CLASSES = {
    "simple": "agent.SimpleAgent.SkullKingAgent",
    "learning": "agent.NeuralAgent.LearningSkullKingAgent",
    "intermediate": "agent.IntermediateAgent.IntermediateSkullKingAgent",
    "agressive": "agent.AgressiveAgent.AggressiveSkullKingAgent",
}

ENVIORNMENT_CLASSES = {
    "no_specials": "env.SKEnvNoSpecials.SkullKingEnvNoSpecials",
    "specials": "env.SKEnv.SkullKingEnv",
}

def load_agent_class(module_class_str):
    module_path, class_name = module_class_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Launch Skull King RL experiments")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to run")
    parser.add_argument("--num_players", type=int, default=3,
                        help="Number of players/agents in the game")
    parser.add_argument("--env_name", type=str, default="no_specials",
                        help="Name of the environment to use")
    parser.add_argument("--hand_size", type=int, default=10,
                        help="Maximum hand size for agents")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for learning agents")
    parser.add_argument("--agent_types", nargs="+", default=["learning", "simple", "simple"],
                        help="List of agent types (simple or learning) for each player")
    # New argument for reproducible experiments.
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for full episode tracing")  
    parser.add_argument("--plot_port", type=int, default=8000,
                        help="Port for live plot server")  # keep for legacy, or ignore
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable TensorBoard logging")  
    parser.add_argument("--shared_networks", action="store_true", help="Share policy networks among learning agents")
    parser.add_argument("--double_positive_rewards", action="store_true", help="Double positive rewards for learning agents")
    parser.add_argument("--training_regimen", type=str, default="base", choices=["base", "sub_episode"],
                        help="Select training regimen: base or sub_episode")
    # New flags for network architecture customization:
    parser.add_argument("--bid_hidden_dims", nargs="+", type=int, default=[64, 64],
                        help="Hidden dimensions for the bid network")
    parser.add_argument("--play_hidden_dims", nargs="+", type=int, default=[64, 64],
                        help="Hidden dimensions for the play network")
    return parser.parse_args()

def setup_logger(logs_dir, debug):
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    plain_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d M%H:%M:%S")
    details_handler = logging.FileHandler(f"{logs_dir}/details.log")
    details_handler.setLevel(logging.DEBUG)
    details_handler.setFormatter(plain_formatter)
    general_handler = logging.FileHandler(f"{logs_dir}/general.log")
    general_handler.setLevel(logging.INFO)
    general_handler.setFormatter(plain_formatter)
    logging.root.handlers = [handler, details_handler, general_handler]
    logging.root.setLevel(logging.DEBUG if debug else logging.INFO)
    return logging.getLogger(__name__)

def init_agents(args, logger):
    agents = []
    shared_bid_net = None
    shared_play_net = None
    shared_trick_win_predictor = None  # NEW: shared trick predictor
    from agent.NeuralAgent import PolicyNetwork, LearningSkullKingAgent
    for agent_type in args.agent_types:
        if agent_type not in AGENT_CLASSES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        AgentClass = load_agent_class(AGENT_CLASSES[agent_type])
        temp_env = SkullKingEnvNoSpecials(num_players=args.num_players, logger=logger)
        temp_props = temp_env.get_env_properties()
        suit_count = temp_props["suit_count"]
        max_rank = temp_props["max_rank"]
        max_players = temp_props["max_players"]
        hand_size = temp_props["hand_size"]
        bid_dimension_vars = temp_props["bid_dimension_vars"]
        play_dimension_vars = temp_props["play_dimension_vars"]
        fixed_bid_features = temp_props["fixed_bid_features"]
        fixed_play_features = temp_props["fixed_play_features"]
        bid_input_dim = hand_size * 5 + 4 + max_players   # 10 from total_scores + 4 extra scalars: position_in_order, player_id, round_number, bidding_phase
        play_input_dim = hand_size * 5 + max_players * 5 +3+ 1 + 1 +max_players+ max_players +1 +5 +2   # Computed as: hand_vec + 3+40 (trick_vec) +2 ([personal_bid, tricks_wincount]) + 10 +10 + 1 +4 +1+2
        
        if agent_type == "learning" and args.shared_networks:
            if shared_bid_net is None:
                shared_bid_net = PolicyNetwork(bid_input_dim, output_dim=11, hidden_dims=args.bid_hidden_dims)
                shared_play_net = PolicyNetwork(play_input_dim, output_dim=hand_size, hidden_dims=args.play_hidden_dims)
                # NEW: Create shared trick predictor using the same hidden dimension as play network's last hidden layer.
                shared_trick_win_predictor = nn.Linear(args.play_hidden_dims[-1] if args.play_hidden_dims else 64, 11)
            agent = AgentClass(args.num_players, hand_size=hand_size, learning_rate=args.learning_rate,
                               shared_bid_net=shared_bid_net, shared_play_net=shared_play_net,
                               shared_trick_win_predictor=shared_trick_win_predictor,  # Pass shared trick predictor
                               suit_count=suit_count, max_rank=max_rank, max_players=max_players,
                               bid_dimension_vars=bid_dimension_vars, play_dimension_vars=play_dimension_vars,
                               bid_hidden_dims=args.bid_hidden_dims, play_hidden_dims=args.play_hidden_dims)
        elif agent_type == "learning":
            agent = AgentClass(args.num_players, hand_size=hand_size, learning_rate=args.learning_rate,
                               suit_count=suit_count, max_rank=max_rank, max_players=max_players,
                               bid_dimension_vars=bid_dimension_vars, play_dimension_vars=play_dimension_vars,
                               bid_hidden_dims=args.bid_hidden_dims, play_hidden_dims=args.play_hidden_dims)
        else:
            agent = AgentClass(args.num_players)
        agents.append(agent)
    return agents

def run_training(args, env, agents, logger, writer):
    if args.training_regimen == "sub_episode":
        from train.sub_episode_training import run_sub_episode_training
        return run_sub_episode_training(args, env, agents, logger, writer)
    else:
        from train.base_training import run_base_training
        return run_base_training(args, env, agents, logger, writer)

def run_plots(logs_dir, agent_rewards, agent_losses, num_players):
    """
    Generate plots for rewards and losses over episodes.

    Parameters:
    - logs_dir (str): Directory to save the plots.
    - agent_rewards (list of lists): Rewards per agent over episodes.
    - agent_losses (tuple): Tuple containing (all_losses, all_bid_losses, all_play_head_losses, all_policy_losses).
    - num_players (int): Number of players/agents in the environment.
    """
    os.makedirs(logs_dir, exist_ok=True)
    
    all_losses, all_bid_losses, all_play_head_losses, all_policy_losses = agent_losses
    num_episodes = len(agent_rewards[0])  # Assuming all agents have same number of episodes
    episodes = np.arange(num_episodes)
    
    print(f'num_episodes: {num_episodes}')
    print(f'len(agent_rewards): {len(agent_rewards)}')
    print(f'len(all_losses): {len(all_losses)}')
    print(f'len(all_bid_losses): {len(all_bid_losses)}')
    
    
    # Plot rewards per agent
    plt.figure(figsize=(10, 5))
    for i in range(num_players):
        plt.plot(episodes, agent_rewards[i], label=f'Agent {i} Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'rewards.png'))
    plt.close()
    
    # Plot bid loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, all_bid_losses, label='Bid Loss', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Bid Loss per Episode')
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'bid_loss.png'))
    plt.close()
    
    # Plot play head loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, all_play_head_losses, label='Play Head Loss', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Play Head Loss per Episode')
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'play_head_loss.png'))
    plt.close()
    
    # Plot policy loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, all_policy_losses, label='Policy Loss', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy Loss per Episode')
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'policy_loss.png'))
    plt.close()

def run_interactive(args, agents, logger):
    from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
    from agent.HumanAgent import HumanAgent
    logger.info("Appending Human Agent as the last player.", color="cyan")
    agents.append(HumanAgent(len(agents) + 1))
    total_players = len(agents)
    env = SkullKingEnvNoSpecials(num_players=total_players, logger=logger)
    # NEW: Enable rotation of starting player during interactive play.
    env.player_0_always_starts = False
    done = False
    obs = env.reset()
    while not done:
        current_player = env.current_player
        current_agent = agents[current_player]
        print(f"\n--- Player {current_player}'s turn ---")
        if isinstance(current_agent, HumanAgent):
            print("Your observation:", obs)
            print("Your hand:", obs['hand'])
        else:
            print(f"Player {current_player} (AI) is acting.")
        if env.bidding_phase:
            valid = list(range(env.action_space.n))
            if isinstance(current_agent, HumanAgent):
                action = int(input(f"Your bid (valid options: {valid}): "))
            else:
                action = current_agent.bid(obs)
        else:
            legal = env.get_legal_actions(env.current_player)
            if isinstance(current_agent, HumanAgent):
                print("Global info - Current trick:", env.current_trick)
                action = int(input(f"Your play (choose card index from {legal}): "))
            else:
                action = current_agent.play_card(obs)
        obs, reward, done, info = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}\n")
        if "trick_info" in info:
            trick_info = info["trick_info"]
            print("=== Trick Summary ===")
            print("Trick cards:")
            for player, card in trick_info["trick_cards"]:
                print(f"  Player {player}: {card}")
            print(f"Trick winner: Player {trick_info['trick_winner']}")
        if "round_bids" in info and "round_tricks_won" in info:
            print("=== Round Summary ===")
            print("Bids:", info["round_bids"])
            print("Tricks won:", info["round_tricks_won"])
            print("Total scores:", obs.get("total_scores", "N/A"))
    print("Game complete. Final scores:", env.total_scores)
    play_again = input("Play again? (y/n): ")
    if play_again.lower() == 'y':
        run_interactive(args, agents, logger)



def main():
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    args = parse_args()
    
    # Enforce shared networks for sub-episode training.
    if args.training_regimen == "sub_episode" and not args.shared_networks:
        # Force shared networks for sub-episode training.
        args.shared_networks = True
        print("Warning: Sub-episode training requires shared networks. Enabling shared_networks.")

    # Set up logs folder and compute experiment-specific logs_dir.
    logs_dir = "logs"
    experiment_properties = f"players_{args.num_players}_episodes_{args.num_episodes}_agents_{'_'.join(args.agent_types)}_env_{args.env_name}"
    logs_dir = os.path.join(logs_dir, experiment_properties)
    os.makedirs(logs_dir, exist_ok=True)
    
    # NEW: Use a tensorboard logs folder inside logs_dir.
    tensorboard_log_dir = os.path.join(logs_dir, "tensorboard_logs")
    if args.enable_tensorboard:
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        # NEW: Launch TensorBoard immediately without waiting for training to finish.
        import subprocess
        tb_process = subprocess.Popen(["tensorboard", "--logdir", tensorboard_log_dir])
        logger_temp = logging.getLogger(__name__)
        logger_temp.info("TensorBoard started at http://localhost:6006", color="green")
    else:
        writer = None

    logger = setup_logger(logs_dir, args.debug)

    # Set seeds only if a seed is provided.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if len(args.agent_types) != args.num_players:
        raise ValueError("Number of agent_types must equal num_players")

    agents = init_agents(args, logger)
    
    if args.training_regimen == "sub_episode":
        from env.SubEpisodeEnv import SubEpisodeEnv
        env = SubEpisodeEnv(num_players=args.num_players, logger=logger)
    else:
        from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
        env = SkullKingEnvNoSpecials(num_players=args.num_players, logger=logger)
    
    agent_rewards, agent_losses = run_training(args, env, agents, logger, writer)
    

    logger.info("Training complete.", color="green")

    if writer:
        writer.close()

    run_plots(logs_dir, agent_rewards, agent_losses, args.num_players)

    # Save the agents' shared networks if they exist. in the logs directory.
    if args.shared_networks:
        for agent in agents:
            if hasattr(agent, 'bid_net'):
                torch.save(agent.bid_net.state_dict(), os.path.join(logs_dir, f"{agent.__class__.__name__}_bid_net.pth"))
                torch.save(agent.play_net.state_dict(), os.path.join(logs_dir, f"{agent.__class__.__name__}_play_net.pth"))
                torch.save(agent.trick_win_predictor.state_dict(), os.path.join(logs_dir, f"{agent.__class__.__name__}_trick_predictor.pth"))
                # only need to save one copy of the shared networks
                break

    # Call run evaluation here, pass the trained networks from the agent to run_evaluation here
    # NEW: Set learnable agents to evaluation mode
    for agent in agents:
        if hasattr(agent, 'bid_net'):
            agent.bid_net.eval()
            agent.play_net.eval()

    run_interactive(args, agents, logger)


if __name__ == "__main__":
    main()