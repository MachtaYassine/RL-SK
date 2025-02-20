import random
import torch
import torch.nn.functional as F
import numpy as np
from env.SKEnvNoSpecials import SkullKingEnvNoSpecials
from agent.NeuralAgent import LearningSkullKingAgent  # NEW import

class SubEpisodeManager:
    def __init__(self, args, agents, logger):
        self.args = args
        self.agents = agents
        self.logger = logger

    def run_sub_episode(self, env):
        episode_rewards = [0] * self.args.num_players
        round_bid_losses = []         # to record bid prediction loss per round
        round_play_head_losses = []   # to record play head loss per round
        done = False
        obs = env.reset()
        last_round = obs["round_number"]
        # Choose a random sub-episode length: run for an additional 1-3 rounds.
        target_round = last_round + random.randint(1, 3)
        self.logger.info(f"Sub-episode target rounds: {target_round}", color="magenta")
        while not done and obs["round_number"] < target_round:
            agent = self.agents[env.current_player]
            obs, reward, done, _ = env.step_with_agent(agent)
            # Update rewards.
            if isinstance(reward, list):
                episode_rewards = [er + r for er, r in zip(episode_rewards, reward)]
            else:
                episode_rewards[env.current_player] += reward
            current_round = obs.get("round_number", last_round)
            if current_round > last_round:
                # At round transition, compute losses per agent.
                for i, ag in enumerate(self.agents):
                    actual_tricks = env.last_round_tricks[i] if hasattr(env, "last_round_tricks") else env.tricks_won[i]
                    bid_pred = ag.bid_prediction if hasattr(ag, "bid_prediction") and ag.bid_prediction is not None else 0
                    bid_loss = F.mse_loss(
                        torch.tensor(bid_pred, dtype=torch.float32),
                        torch.tensor(actual_tricks, dtype=torch.float32)
                    )
                    if ag.trick_win_predictons:
                        predictions = torch.tensor(ag.trick_win_predictons, dtype=torch.float32)
                        target = torch.full((len(ag.trick_win_predictons),), actual_tricks, dtype=torch.float32)
                        play_head_loss = F.mse_loss(predictions, target)
                    else:
                        play_head_loss = torch.tensor(0.0, dtype=torch.float32)
                    round_bid_losses.append(bid_loss.item())
                    round_play_head_losses.append(play_head_loss.item())
                    # Reset predictions for next round.
                    ag.bid_prediction = None
                    ag.trick_win_predictons.clear()
                last_round = current_round
        avg_bid_loss = np.mean(round_bid_losses) if round_bid_losses else 0.0
        avg_play_loss = np.mean(round_play_head_losses) if round_play_head_losses else 0.0
        total_loss = avg_bid_loss + avg_play_loss
        self.logger.info(f"Sub-episode complete: Rewards {episode_rewards}, Loss {total_loss}", color="green")
        return total_loss
