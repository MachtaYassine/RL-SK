import argparse
import importlib
import numpy as np
import random
import torch
import logging
import os  # NEW: for file system operations

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
    parser.add_argument("--agent_types", nargs="+", default=["learning", "simple", "simple"],
                        help="List of agent types (simple or learning) for each player")
    # New argument for reproducible experiments.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for full episode tracing")  # new
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging and create logs folder with two log files (general & details)
    logs_dir = "logs"
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

    episode_rewards = [0] * args.num_players

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
            logger.debug(f"After action, obs: {obs}, reward: {reward}, done: {done}")

        logger.debug("Episode complete. Updating learning agents if applicable.", color="blue")
        for i, agent in enumerate(agents):
            if hasattr(agent, "optimizer_bid") and agent.log_probs:
                R = 0
                policy_loss = []
                # Process log_probs in reverse order with rewards assigned via environment
                # Here we use a placeholder reward sequence (e.g., 1 per step) since rewards may vary.
                # In a refined design, rewards should be stored alongside log_probs.
                for log_prob in reversed(agent.log_probs):
                    R = 1 + 0.99 * R  # placeholder for return
                    policy_loss.insert(0, -log_prob * R)
                loss = torch.stack(policy_loss).sum()
                agent.optimizer_bid.zero_grad()
                agent.optimizer_play.zero_grad()
                loss.backward()
                agent.optimizer_bid.step()
                agent.optimizer_play.step()
                logger.debug(f"Player {i} updated with loss: {loss.item()}")
                agent.log_probs.clear()  # clear buffer for the next episode

        logger.info(f"--- Episode {episode+1} complete ---", color="green")

    logger.info("Training complete.", color="green")


if __name__ == "__main__":
    main()