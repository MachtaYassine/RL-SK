import argparse
import importlib
import numpy as np
import random
import torch
import logging
import os  # NEW: for file system operations
import matplotlib.pyplot as plt  # NEW: for plotting
import threading               # NEW: for running live server
import http.server             # NEW: for live server
import socketserver            # NEW: for live server
from torch.utils.tensorboard import SummaryWriter  # NEW: for TensorBoard logging
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
from agent.HumanAgent import HumanAgent
import pickle  # NEW: for saving pickle files

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
    "rebel": "agent.ReBelAgent.ReBelSkullKingAgent",
}

ENVIORNMENT_CLASSES = {
    "no_specials": "env.SKEnvNoSpecials.SkullKingEnvNoSpecials",
    "specials": "env.SKEnv.SkullKingEnv",
}

def load_agent_class(module_class_str):
    module_path, class_name = module_class_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def start_live_server(port):
    # Serve current directory (including live_training_plot.html) on the given port
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving live plot at http://localhost:{port}")
        httpd.serve_forever()

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
                        help="List of agent types (simple, learning) for each player")
    # New argument for reproducible experiments.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for full episode tracing")  
    parser.add_argument("--plot_port", type=int, default=8000,
                        help="Port for live plot server")  # keep for legacy, or ignore
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable TensorBoard logging")  
    parser.add_argument("--load_weights", action="store_true", help="Load model weights and skip training")
    return parser.parse_args()

def main():
    args = parse_args()

    # NEW: Initialize lists to record value network loss and bidding loss.
    value_net_losses = []
    bidding_losses = []

    # Initialize TensorBoard SummaryWriter if enabled.
    writer = None
    if args.enable_tensorboard:
        writer = SummaryWriter(log_dir="tensorboard_logs")
    
    # Set up logging and create logs folder with two log files (general & details)
    logs_dir = "logs"
    experiment_properties= f"players_{args.num_players}_episodes_{args.num_episodes}_agents_{'_'.join(args.agent_types)}_env_{args.env_name}"
    logs_dir = os.path.join(logs_dir, experiment_properties)
    os.makedirs(logs_dir, exist_ok=True)

    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    # NEW: plain formatter for file handlers (no color codes)
    plain_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # File handler for detailed logs (DEBUG and above)
    details_handler = logging.FileHandler(f"{logs_dir}/details.log")
    details_handler.setLevel(logging.DEBUG)
    details_handler.setFormatter(plain_formatter)

    # File handler for general summary logs (INFO and above)
    general_handler = logging.FileHandler(f"{logs_dir}/general.log")
    general_handler.setLevel(logging.INFO)
    general_handler.setFormatter(plain_formatter)

    # Configure root logger with stream and file handlers
    logging.root.handlers = [handler, details_handler, general_handler]
    logging.root.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger = logging.getLogger(__name__)

    # Set seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.agent_types) != args.num_players:
        raise ValueError("Number of agent_types must equal num_players")

    # Instantiate agents
    agents = []
    for agent_type in args.agent_types:
        if agent_type not in AGENT_CLASSES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        AgentClass = load_agent_class(AGENT_CLASSES[agent_type])
        agent = AgentClass(args.num_players)
        agents.append(agent)

    env = SkullKingEnvNoSpecials(num_players=args.num_players)

    agent_rewards = [[] for _ in range(args.num_players)]
    # For non-learning agents, leave loss list empty.
    agent_losses = [[] for _ in range(args.num_players)]

    if args.load_weights:
        for idx, agent in enumerate(agents):
            if args.agent_types[idx] in ["learning", "rebel"]:
                model_dir = os.path.join("models", args.agent_types[idx])
                bidding_path = os.path.join(model_dir, "bidding_net.pth.tar")
                policy_path  = os.path.join(model_dir, "policy_net.pth.tar")
                agent.bidding_network.load_state_dict(torch.load(bidding_path))
                agent.policy_network.load_state_dict(torch.load(policy_path))
                logging.getLogger(__name__).info(
                    f"Loaded weights for agent {idx} of type {args.agent_types[idx]}",
                    color="green"
                )
        logging.getLogger(__name__).info(
            "Weights loaded. Skipping training.",
            color="green"
        )
        for episode in range(args.num_episodes):
            logger.debug(f"=== Episode {episode+1} start ===", color="magenta")
            obs = env.reset()
            done = False
            episode_rewards = [0] * args.num_players
            while not done:
                current_player = env.current_player
                agent = agents[current_player]
                logger.debug(f"Player {current_player}'s turn. Using step_with_agent.", color="cyan")
                obs, reward, done, _ = env.step_with_agent(agent)
                # Summing rewards if reward is a list (trick/reward per agent) or single value.
                if isinstance(reward, list):
                    episode_rewards = [er + r for er, r in zip(episode_rewards, reward)]
                else:
                    episode_rewards[current_player] += reward
            # Record per-agent rewards.
            for i in range(args.num_players):
                agent_rewards[i].append(episode_rewards[i])
    else:

        for episode in range(args.num_episodes):
            logger.debug(f"=== Episode {episode+1} start ===", color="magenta")
            
            if "rebel" in args.agent_types:
                rebel_agent = agents[args.agent_types.index("rebel")]
                n_players = random.randint(2, 5)
                n_rounds = random.randint(1, 10)
                value_loss, policy_loss, bidding_loss = rebel_agent.train(n_players, n_rounds)
                # Record losses from rebel training.
                agent_losses[0].extend(policy_loss)
                value_net_losses.extend(value_loss)
                bidding_losses.extend(bidding_loss)
                other_rebel_agents = [agent for i, agent in enumerate(agents) if args.agent_types[i]=="rebel" and agent != rebel_agent]
                for other_agent in other_rebel_agents:
                    other_agent.bidding_network.load_state_dict(rebel_agent.bidding_network.state_dict())
                    other_agent.policy_network.load_state_dict(rebel_agent.policy_network.state_dict())
                obs = env.reset()
                done = False
                episode_rewards = [0] * args.num_players
                while not done:
                    current_player = env.current_player
                    agent = agents[current_player]
                    logger.debug(f"Player {current_player}'s turn. Using step_with_agent.", color="cyan")
                    obs, reward, done, _ = env.step_with_agent(agent)
                    # Summing rewards if reward is a list (trick/reward per agent) or single value.
                    if isinstance(reward, list):
                        episode_rewards = [er + r for er, r in zip(episode_rewards, reward)]
                    else:
                        episode_rewards[current_player] += reward
                # Record per-agent rewards.
                for i in range(args.num_players):
                    agent_rewards[i].append(episode_rewards[i])

                # Save loss lists to pickle files in logs_dir.
                with open(os.path.join(logs_dir, "value_net_losses.pkl"), "wb") as f:
                    pickle.dump(value_net_losses, f)
                with open(os.path.join(logs_dir, "bidding_losses.pkl"), "wb") as f:
                    pickle.dump(bidding_losses, f)
                
                
            else:
                obs = env.reset()
                done = False
                episode_rewards = [0] * args.num_players

                while not done:
                    current_player = env.current_player
                    agent = agents[current_player]
                    logger.debug(f"Player {current_player}'s turn. Using step_with_agent.", color="cyan")
                    obs, reward, done, _ = env.step_with_agent(agent)
                    # Summing rewards if reward is a list (trick/reward per agent) or single value.
                    if isinstance(reward, list):
                        episode_rewards = [er + r for er, r in zip(episode_rewards, reward)]
                    else:
                        episode_rewards[current_player] += reward

                logger.debug("Episode complete. Updating learning agents if applicable.", color="blue")
                for i, agent in enumerate(agents):
                    if hasattr(agent, "optimizer_bid") and agent.log_probs:
                        R = 0
                        agent_policy_losses = []
                        for log_prob in reversed(agent.log_probs):
                            R = 1 + 0.99 * R  # placeholder for return
                            agent_policy_losses.insert(0, -log_prob * R)
                        loss = torch.stack(agent_policy_losses).sum()
                        agent.optimizer_bid.zero_grad()
                        agent.optimizer_play.zero_grad()
                        loss.backward()
                        agent.optimizer_bid.step()
                        agent.optimizer_play.step()
                        logger.debug(f"Player {i} updated with loss: {loss.item()}")
                        agent.log_probs.clear()
                        # Record loss into agent_losses for learning agents.
                        agent_losses[i].append(loss.item())
                    else:
                        # For non-learning agents, add a dummy value (or skip).
                        agent_losses[i].append(None)
                # Record per-agent rewards.
                for i in range(args.num_players):
                    agent_rewards[i].append(episode_rewards[i])
                    if writer:
                        writer.add_scalar(f"Agent_{i}/Reward", episode_rewards[i], episode)
                        if agent_losses[i][-1] is not None:
                            writer.add_scalar(f"Agent_{i}/Loss", agent_losses[i][-1], episode)
            
            
            logger.info(f"--- Episode {episode+1} complete ---", color="green")
            # Save networks every 5 episodes
            if (episode + 1) % 5 == 0:
                for idx, agent_type in enumerate(args.agent_types):
                    if agent_type == "rebel":
                        torch.save(
                            agents[idx].policy_network.state_dict(),
                            os.path.join(logs_dir, f"policy_net_episode_{episode+1}.pth.tar")
                        )
                        torch.save(
                            agents[idx].value_network.state_dict(),
                            os.path.join(logs_dir, f"value_net_episode_{episode+1}.pth.tar")
                        )
                        torch.save(
                            agents[idx].bidding_network.state_dict(),
                            os.path.join(logs_dir, f"bidding_net_episode_{episode+1}.pth.tar")
                        )

        logger.info("Training complete.", color="green")

        if writer:
            writer.close()

    # Update agents' internal number of players to include the human player.
    total_players = len(agents) + 1  # new total players including human
    for agent in agents:
        agent.num_players = total_players

    plt.figure(figsize=(10, 5))
    for i in range(args.num_players):
        plt.plot(agent_rewards[i], label=f"Agent {i} Reward (Type: {args.agent_types[i]})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Curve (Reward per Episode)")
    plt.legend()
    plt.show()
    plt.savefig(f"{logs_dir}/training_curve.png")

    
    plt.figure(figsize=(10, 5))
    for i in range(args.num_players):
        if any(x is not None for x in agent_losses[i]):
            plt.plot([x for x in agent_losses[i] if x is not None], label=f"Agent {i} Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Policy Loss Curve")
    plt.legend()
    plt.show()
    plt.savefig(f"{logs_dir}/policy_loss_curve.png")

# Set learnable agents to evaluation mode
    for agent in agents:
        if hasattr(agent, 'bid_net'):
            agent.bid_net.eval()
            agent.play_net.eval()

    # Launch an interactive game with a HumanAgent.
    logger.info("Launching interactive play mode...", color="green")
    # Prepare a new game with total players = trained agents + one human.
    total_players = len(agents) + 1
    game_env = SkullKingEnvNoSpecials(num_players=total_players)
    # Copy existing trained agents then append the HumanAgent.
    agents_with_human = agents.copy()
    agents_with_human.append(HumanAgent(total_players))
    
    # Interactive game loop.
    play = True
    while play:
        obs = game_env.reset()
        done = False
        logger.info("--- New Interactive Game Started ---", color="green")
        phase='Bidding'
        logger.info(f"Phase: {phase}", color="magenta")
        while not done:
            current_player = game_env.current_player
            agent = agents_with_human[current_player]
            if game_env.bidding_phase==0 and phase=='Bidding':
                phase='Playing'
                logger.info(f"Phase: {phase}", color="magenta")
            elif game_env.bidding_phase==1 and phase=='Playing':
                phase='Bidding'
                logger.info(f"Phase: {phase}", color="magenta")
            
            logger.info(f"Interactive play: Player {current_player}'s turn.", color="cyan")
            observation=game_env._get_observation()
            if isinstance(agent, HumanAgent):
                print(f"Observation: {observation}")

            obs, reward, done, _ = game_env.step_with_agent(agent)
            
            if reward:
                print(f"Reward: {reward}")
        print(f"\nFinal Scores: {obs.get('total_scores', 'N/A')}")
        play_again = input("Play again? (y/n): ")
        if play_again.lower() != 'y':
            play = False


if __name__ == "__main__":
    main()