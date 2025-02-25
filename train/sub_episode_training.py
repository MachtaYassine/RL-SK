from .base_training import run_base_training
import random
import torch
import torch.nn.functional as F
import numpy as np
from agent.NeuralAgent import LearningSkullKingAgent, PolicyNetwork  # necessary imports

# Consolidated SubEpisodeManager class:
class SubEpisodeManager:
    def __init__(self, args, agents, logger):
        self.args = args
        self.agents = agents
        self.logger = logger
        # Initialize shared networks to reuse across reinitializations.
        self.shared_bid_net = None
        self.shared_play_net = None

    def reinitialize_agents(self, env):
        # Randomize number of agents between 3 and 8.
        new_num = random.randint(3, 8)
        # Removed verbose logging:
        # self.logger.info(f"Randomly initializing agents list with {new_num} agents.", color="cyan")
        temp_props = env.get_env_properties()
        hand_size = temp_props["hand_size"]
        bid_hidden_dims = self.args.bid_hidden_dims
        play_hidden_dims = self.args.play_hidden_dims
        fixed_bid_features = temp_props["fixed_bid_features"]
        fixed_play_features = temp_props["fixed_play_features"]
        bid_input_dim = hand_size * 5 + fixed_bid_features
        play_input_dim = hand_size * 5 + fixed_play_features
        if self.args.shared_networks:
            if self.shared_bid_net is None or self.shared_play_net is None:
                self.shared_bid_net = PolicyNetwork(bid_input_dim, output_dim=11, hidden_dims=bid_hidden_dims)
                self.shared_play_net = PolicyNetwork(play_input_dim, output_dim=hand_size, hidden_dims=play_hidden_dims)
            shared_bid_net = self.shared_bid_net
            shared_play_net = self.shared_play_net
        else:
            shared_bid_net, shared_play_net = None, None

        new_agents = []
        for i in range(new_num):
            agent = LearningSkullKingAgent(
                new_num, hand_size=hand_size, learning_rate=self.args.learning_rate,
                shared_bid_net=shared_bid_net, shared_play_net=shared_play_net,
                suit_count=temp_props["suit_count"],
                max_rank=temp_props["max_rank"],
                max_players=temp_props["max_players"],
                bid_dimension_vars=temp_props["bid_dimension_vars"],
                play_dimension_vars=temp_props["play_dimension_vars"],
                bid_hidden_dims=bid_hidden_dims,
                play_hidden_dims=play_hidden_dims
            )
            new_agents.append(agent)
        self.agents = new_agents
        env.num_players = new_num
        # Removed verbose logging:
        # self.logger.info(f"Environment updated: now simulating {new_num} agents.", color="cyan")

    def run_sub_episode(self, env):
        # Removed verbose log:
        # self.logger.info(f"Sub-episode training: Input agents count ({len(self.agents)}) does not matter; manager will randomly sample agents.", color="cyan")
        obs = env.reset()
        last_round = obs["round_number"]
        episode_rewards = [0] * env.num_players
        round_bid_losses = []
        round_play_head_losses = []
        done = False
        current_round = obs["round_number"]
        self.logger.info(
            f"Starting Round {current_round}: Simulated players: {env.num_players}, Hand size: {env.round_number}",
            color="magenta"
        )
        last_round = current_round

        while not done and obs["round_number"] == last_round:
            agent = self.agents[env.current_player]
            obs, reward, done, _ = env.step_with_agent(agent)
            if isinstance(reward, list):
                episode_rewards = [er + r for er, r in zip(episode_rewards, reward)]
            else:
                episode_rewards[env.current_player] += reward

        # Log recap: original bids and tricks won.
        self.logger.info(
            f"Round {last_round} recap: Bids: {getattr(env, 'last_bids', 'N/A')}, Tricks won: {getattr(env, 'last_round_tricks', 'N/A')}",
            color="blue"
        )

        for i, ag in enumerate(self.agents):
            actual_tricks = env.last_round_tricks[i] if hasattr(env, "last_round_tricks") else env.tricks_won[i]
            bid_input = (ag.bid_prediction if hasattr(ag, "bid_prediction") and ag.bid_prediction is not None 
                         else torch.tensor(0.0, dtype=torch.float32))
            bid_loss = F.mse_loss(bid_input, torch.tensor(actual_tricks, dtype=torch.float32))
            if ag.trick_win_predictons:
                predictions = torch.stack(ag.trick_win_predictons) if isinstance(ag.trick_win_predictons[0], torch.Tensor) \
                              else torch.tensor(ag.trick_win_predictons, dtype=torch.float32)
                play_head_loss = F.mse_loss(predictions, torch.full((len(ag.trick_win_predictons),), actual_tricks, dtype=torch.float32))
            else:
                play_head_loss = torch.tensor(0.0, dtype=torch.float32)
            round_bid_losses.append(bid_loss)
            round_play_head_losses.append(play_head_loss)
            ag.bid_prediction = None
            ag.trick_win_predictons.clear()

        avg_bid_loss = torch.mean(torch.stack(round_bid_losses)) if round_bid_losses else torch.tensor(0.0, dtype=torch.float32)
        avg_play_loss = torch.mean(torch.stack(round_play_head_losses)) if round_play_head_losses else torch.tensor(0.0, dtype=torch.float32)
        total_reward = torch.tensor(sum(episode_rewards), dtype=torch.float32)
        total_loss = avg_bid_loss + avg_play_loss
        policy_Loss = - total_reward / torch.tensor(len(self.agents), dtype=torch.float32)
        total_loss = total_loss + policy_Loss
        self.logger.info(
            f"Round {last_round} complete: Rewards {episode_rewards}, Loss {total_loss.item()}, Policy Loss {policy_Loss.item()}, Bid Loss {avg_bid_loss.item()}, Play Head Loss {avg_play_loss.item()}",
            color="green"
        )
        return total_loss, avg_bid_loss, avg_play_loss, total_reward

# Main training function for sub-episode regimen.
def run_sub_episode_training(args, env, agents, logger, writer=None):
    sub_manager = SubEpisodeManager(args, agents, logger)
    if args.shared_networks:
        shared_rewards = []
        shared_losses = []
        shared_bid_losses = []
        shared_play_head_losses = []
        shared_total_rewards = []
        # Initialize shared optimizers from the first agent (they are shared)
        shared_opts = (agents[0].optimizer_bid, agents[0].optimizer_play)
    else:
        all_rewards = [[] for _ in range(args.num_players)]
        all_losses = [[] for _ in range(args.num_players)]
    
    for episode in range(args.num_episodes):
        logger.info(f"--- Sub-episode {episode+1} start ---", color="magenta")
        sub_manager.reinitialize_agents(env)
        # Compute loss and metrics for the sub-episode.
        loss, avg_bid_loss, avg_play_loss, total_reward = sub_manager.run_sub_episode(env)
        logger.info(f"Sub-episode {episode+1} complete, loss: {loss}", color="green")
        # If shared networks, update parameters using the aggregated loss.
        if args.shared_networks:
            # Here you could aggregate losses over multiple rounds if desired.
            if loss is not None:
                from train.base_training import update_shared
                update_shared(shared_opts[0], shared_opts[1], loss)
            writer.add_scalar("Sub_Episode/Shared_Bid_Loss", avg_bid_loss, episode)
            writer.add_scalar("Sub_Episode/Shared_PlayHead_Loss", avg_play_loss, episode)
            writer.add_scalar("Sub_Episode/Shared_Reward_Total", total_reward, episode)
            shared_rewards.append(total_reward)
            shared_losses.append(loss)
            shared_bid_losses.append(avg_bid_loss)
            shared_play_head_losses.append(avg_play_loss)
        else:
            for i in range(args.num_players):
                all_rewards[i].append(0)
                all_losses[i].append(loss if loss is not None else 0)
        # End of episode loop.
    if args.shared_networks:
        return shared_rewards, (shared_losses, shared_bid_losses, shared_play_head_losses)
    else:
        return all_rewards, (all_losses, None, None)
