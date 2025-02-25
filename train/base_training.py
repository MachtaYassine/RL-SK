import torch
import torch.nn.functional as F
import numpy as np
import sys
from utils.check_gradients import check_optimizer_gradients

# Helper: update shared optimizers if in shared mode.
def update_shared(shared_opt_bid, shared_opt_play, loss, logger):
    shared_opt_bid.zero_grad()
    shared_opt_play.zero_grad()
    loss.backward()
    # Check gradients on shared bid optimizer.
    missing_bid = check_optimizer_gradients(shared_opt_bid, logger)
    if missing_bid:
        logger.warning(f"Shared bid optimizer missing gradients for parameters: {missing_bid}")
        # Optionally, sys.exit(1)
    # Check gradients on shared play optimizer.
    missing_play = check_optimizer_gradients(shared_opt_play, logger)
    if missing_play:
        logger.warning(f"Shared play optimizer missing gradients for parameters: {missing_play}")
        # Optionally, sys.exit(1)
    shared_opt_bid.step()
    shared_opt_play.step()

# Helper: process one episode.
def run_episode(args, env, agents, logger, round_score_loss_coef, double_positive_rewards):
    episode_rewards = [0] * args.num_players
    round_bid_losses = []         # list of torch tensors per round
    round_play_head_losses = []   # list of torch tensors per round
    round_policy_losses = []      # list of torch tensors per round
    episode_bid_losses,episode_play_head_losses,episode_policy_losses,episode_total_loss = [],[],[],[]
    cumulative_loss = torch.tensor(0.0, dtype=torch.float32)  # NEW: cumulative loss across rounds
    done = False
    obs = env.reset()
    last_round = obs["round_number"]

    while not done:
        agent_idx = env.current_player
        agent = agents[agent_idx]
        logger.debug(f"Player {agent_idx}'s turn.", color="cyan")
        obs, reward, done, _ = env.step_with_agent(agent)
        if isinstance(reward, list):
            episode_rewards = [er + (2 * r if r > 0 and double_positive_rewards else r)
                               for er, r in zip(episode_rewards, reward)]
        else:
            episode_rewards[agent_idx] += reward

        current_round = obs.get("round_number", last_round)
        if current_round > last_round:
            targets = env.last_round_tricks if hasattr(env, "last_round_tricks") else env.tricks_won
            for i, ag in enumerate(agents):
                actual_tricks = targets[i]
                # Use the stored network output for bid_prediction
                bid_input = (ag.bid_prediction if ag.bid_prediction is not None 
                             else torch.tensor(0.0, dtype=torch.float32))
                target_tensor = torch.tensor(actual_tricks, dtype=torch.float32)
                bid_loss = F.l1_loss(bid_input, target_tensor)
                # Compute play head loss using one-hot target for predictions
                if ag.trick_win_predictons:
                    predictions = torch.stack(ag.trick_win_predictons) \
                                  if isinstance(ag.trick_win_predictons[0], torch.Tensor) \
                                  else torch.tensor(ag.trick_win_predictons, dtype=torch.float32)
                    N, D = predictions.shape
                    actual_index = min(int(actual_tricks), D-1)
                    target_dist = torch.zeros_like(predictions)
                    target_dist[:, actual_index] = 1.0
                    play_head_loss = F.l1_loss(predictions, target_dist)
                else:
                    play_head_loss = torch.tensor(0.0, dtype=torch.float32)
                    
                # POlicy loss, get log_probs
                log_porbs_for_round = torch.stack(ag.log_probs) # One action for bid and n_round for cards played
                reward[i]= 2*reward[i] if reward[i]<0 else reward[i] # double positive rewards to Punish more negative rewards
                round_score_for_agent = 1-reward[i]/(last_round*20)
                if round_score_for_agent<0:
                    logger.warning(f"Negative score for agent {i} in round {last_round}, reward {reward[i]} and last_round {last_round}")
                #if the log probs are positive also warn:
                if torch.any(log_porbs_for_round>0):
                    logger.warning(f"Positive log probs for agent {i} in round {last_round}")
                #make tensore shape of log probs with round_score as value for all entreis
                round_score_tensor = torch.full_like(log_porbs_for_round, round_score_for_agent)
                #compute policy loss
                policy_loss = -torch.mean(round_score_tensor * log_porbs_for_round).sum() 
                logger.info(
                    f"Agent {i}: Actual tricks won {actual_tricks} | "
                    f"Bid prediction distribution: {np.array2string(bid_input.detach().cpu().numpy(), formatter={'float_kind': lambda x: f'{x:.1e}'})} | "
                    f"Bid loss: {bid_loss.item():.1e} || "
                    f"Play head prediction distribution (mean): {np.array2string(torch.mean(predictions, dim=0).detach().cpu().numpy(), formatter={'float_kind': lambda x: f'{x:.1e}'})} | "
                    f"Play head loss: {play_head_loss.item():.1e} | "
                    f"Policy loss: {policy_loss.item():.1e}",
                    color="blue"
                )

                round_bid_losses.append(bid_loss)
                round_play_head_losses.append(play_head_loss)
                round_policy_losses.append(policy_loss)
                if not args.shared_networks:
                    # Update the optimizer for this agent.
                    ag.optimizer_bid.zero_grad()
                    ag.optimizer_play.zero_grad()
                    (bid_loss + play_head_loss+policy_loss).backward()
                    ag.optimizer_bid.step()
                    ag.optimizer_play.step()
                # Reset predictions for next round.
                ag.bid_prediction = None
                ag.trick_win_predictons.clear()
                ag.log_probs.clear()
            # NEW: Compute and update loss for the round
            round_avg_bid_loss = torch.mean(torch.stack(round_bid_losses)) if round_bid_losses else torch.tensor(0.0, dtype=torch.float32)
            round_avg_play_loss = torch.mean(torch.stack(round_play_head_losses)) if round_play_head_losses else torch.tensor(0.0, dtype=torch.float32)
            round_avg_policy_loss = torch.mean(torch.stack(round_policy_losses)) if round_policy_losses else torch.tensor(0.0, dtype=torch.float32)
            round_loss = round_avg_bid_loss + round_avg_play_loss+round_avg_policy_loss
            if args.shared_networks:
                # Assume shared_opts is available in the outer scope of run_base_training (see below)
                update_shared(agents[0].optimizer_bid, agents[0].optimizer_play, round_loss, logger)
                
            cumulative_loss += round_loss
            # Reset per-round loss lists for the next round.
            round_bid_losses.clear()
            round_play_head_losses.clear()
            round_policy_losses.clear()
            last_round = current_round
            
            episode_bid_losses.append(round_avg_bid_loss.detach().cpu().numpy())
            episode_play_head_losses.append(round_avg_play_loss.detach().cpu().numpy())
            episode_policy_losses.append(round_avg_policy_loss.detach().cpu().numpy())
            episode_total_loss.append(round_loss.detach().cpu().numpy())

    return episode_rewards, cumulative_loss, np.mean(episode_bid_losses), np.mean(episode_play_head_losses), np.mean(episode_policy_losses), np.mean(episode_total_loss)

# Main functional training procedure.
# Main functional training procedure.
def run_base_training(args, env, agents, logger, writer=None):
    round_score_loss_coef = 0.0001
    # Setup shared optimizers if needed.
    # shared_opts = None
    # if args.shared_networks:
    #     for agent in agents:
    #         if hasattr(agent, "optimizer_bid"):
    #             shared_opts = (agent.optimizer_bid, agent.optimizer_play)
    #             break

    # Run episodes.
    all_rewards = [[] for _ in range(args.num_players)]
    all_losses = [[] for _ in range(args.num_players)]
    all_bid_losses = []         # NEW: to collect bid losses per round across episodes
    all_play_head_losses = []   # NEW: to collect play head losses per round across episodes
    all_policy_losses = []      # NEW: to collect policy losses per round across episodes
    all_total_rewards = []

    for episode in range(args.num_episodes):
        logger.debug(f"=== Episode {episode+1} start ===", color="magenta")
        ep_rewards, cumul_loss, avg_bid_loss, avg_play_loss, avg_policy_loss, total_loss_round = run_episode(
            args, env, agents, logger, round_score_loss_coef, args.double_positive_rewards
        )
        total_reward = sum(ep_rewards)
        for i in range(args.num_players):
            all_rewards[i].append(ep_rewards[i])
            all_losses[i].append(cumul_loss if cumul_loss is not None else 0)
        if args.shared_networks and cumul_loss is not None :
            # NOTE: The per‐round update was done inside run_episode, so no further update is required here.
            all_total_rewards.append(total_reward)
            if writer:
                writer.add_scalar("Shared/Reward_Total", total_reward, episode)
        else:
            for i in range(args.num_players):
                if writer:
                    writer.add_scalar(f"Agent_{i}/Reward", ep_rewards[i], episode)

        # NEW: Log the losses for this episode.
        all_bid_losses.append(avg_bid_loss)
        all_play_head_losses.append(avg_play_loss)
        all_policy_losses.append(avg_policy_loss)  # NEW: Store policy loss
        all_total_rewards.append(total_reward)

        if writer:
            writer.add_scalar("Loss/Bid", avg_bid_loss, episode)
            writer.add_scalar("Loss/PlayHead", avg_play_loss, episode)
            writer.add_scalar("Loss/Policy", avg_policy_loss, episode)  # NEW: Log policy loss
            writer.add_scalar("Reward/Total", total_reward, episode)
            if cumul_loss is not None:
                writer.add_scalar("Loss/Shared", cumul_loss.item(), episode)

        logger.info(f"--- Episode {episode+1} complete ---", color="green")
        logger.info(f"Episode rewards: {ep_rewards}", color="green")

    # Pack the loss curves with rewards for plotting.
    return all_rewards, (all_losses, all_bid_losses, all_play_head_losses, all_policy_losses)
