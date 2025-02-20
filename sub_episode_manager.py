import random
import torch
import torch.nn.functional as F
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
from agent.NeuralAgent import LearningSkullKingAgent  # NEW import

class SubEpisodeManager:
    def __init__(self, args, agents, logger):
        self.args = args
        self.agents = agents
        self.logger = logger

    def run_sub_episode(self):
        # Generate random round number [1, 10]
        round_num = random.randint(1, 10)
        # Generate random number of players [3, 8]
        num_players = random.randint(3, 8)
        # Generate arbitrary total scores per player (range based on round number)
        total_scores = [random.randint(-round_num * 10, round_num * 10) for _ in range(num_players)]
        
        # Instantiate as many agents as generated players using shared networks.
        # Assume that self.agents[0] is an instance of LearningSkullKingAgent with shared networks.
        shared_bid_net = self.agents[0].bid_net
        shared_play_net = self.agents[0].play_net
        hand_size = self.args.hand_size
        learning_rate = self.args.learning_rate
        new_agents = []
        for _ in range(num_players):
            agent = LearningSkullKingAgent(num_players, hand_size=hand_size, learning_rate=learning_rate,
                                           shared_bid_net=shared_bid_net, shared_play_net=shared_play_net)
            new_agents.append(agent)

        # Create a new environment instance for this sub-episode.
        env = SkullKingEnvNoSpecials(num_players, logger=self.logger)
        # Set the round and total scores for this sub-episode.
        env.round_number = round_num
        env.max_rounds = round_num  # Only one round will be played.
        # Pad total_scores to env.MAX_PLAYERS if needed.
        if len(total_scores) < env.MAX_PLAYERS:
            total_scores += [0] * (env.MAX_PLAYERS - len(total_scores))
        env.total_scores = total_scores
        env.reset()  # Initiate the round (deal cards, etc.)

        shared_loss = None
        round_score_loss_coef = 0.1

        done = False
        while not done:
            current_player = env.current_player
            agent = new_agents[current_player]  # Use the newly instantiated agents.
            obs, reward, done, _ = env.step_with_agent(agent)
            # If a reward is returned, update loss for learning agents.
            if reward:
                round_reward = reward if not isinstance(reward, list) else reward[current_player]
                if hasattr(agent, "optimizer_bid") and agent.log_probs:
                    policy_loss = torch.stack([-lp * round_reward for lp in agent.log_probs]).sum()
                    # Placeholder auxiliary loss (could be refined as needed).
                    aux_loss = 0.0
                    loss = policy_loss + round_score_loss_coef * aux_loss
                    shared_loss = loss if shared_loss is None else shared_loss + loss
                    agent.log_probs.clear()
        if shared_loss is not None:
            # After the sub-episode, update shared optimizers.
            return shared_loss.item()
        return 0.0
