import torch
import torch.nn.functional as F

# Helper: update shared optimizers if in shared mode.
def update_shared(shared_opt_bid, shared_opt_play, loss):
    shared_opt_bid.zero_grad()
    shared_opt_play.zero_grad()
    loss.backward()
    shared_opt_bid.step()
    shared_opt_play.step()

# Helper: process one episode.
def run_episode(args, env, agents, logger, round_score_loss_coef, double_positive_rewards):
    episode_rewards = [0] * args.num_players
    shared_loss = None
    done = False
    obs = env.reset()
    
    while not done:
        agent_idx = env.current_player
        agent = agents[agent_idx]
        logger.debug(f"Player {agent_idx}'s turn.", color="cyan")
        obs, reward, done, _ = env.step_with_agent(agent)
        if isinstance(reward, list):
            episode_rewards = [
                er + (2 * r if r > 0 and double_positive_rewards else r)
                for er, r in zip(episode_rewards, reward)
            ]
        else:
            episode_rewards[agent_idx] += reward
        # Process loss if reward returned.
        if reward:
            round_reward = reward if not isinstance(reward, list) else reward[agent_idx]
            if hasattr(agent, "optimizer_bid") and agent.log_probs:
                policy_loss = torch.stack([-lp * round_reward for lp in agent.log_probs]).sum()
                if agent.trick_win_predictons:
                    aux_loss = sum(
                        F.mse_loss(pred, torch.tensor(0.0))
                        for pred in agent.trick_win_predictons
                    ) / len(agent.trick_win_predictons)
                else:
                    aux_loss = 0.0
                loss = policy_loss + round_score_loss_coef * aux_loss
                if args.shared_networks:
                    shared_loss = loss if shared_loss is None else shared_loss + loss
                else:
                    agent.optimizer_bid.zero_grad()
                    agent.optimizer_play.zero_grad()
                    loss.backward()
                    agent.optimizer_bid.step()
                    agent.optimizer_play.step()
                agent.log_probs.clear()
                agent.trick_win_predictons.clear()
    return episode_rewards, shared_loss

# Main functional training procedure.
def run_base_training(args, env, agents, logger, writer=None):
    round_score_loss_coef = 0.0001
    # Setup shared optimizers if needed.
    shared_opts = None
    if args.shared_networks:
        for agent in agents:
            if hasattr(agent, "optimizer_bid"):
                shared_opts = (agent.optimizer_bid, agent.optimizer_play)
                break

    # Run episodes.
    all_rewards = [[] for _ in range(args.num_players)]
    all_losses = [[] for _ in range(args.num_players)]
    for episode in range(args.num_episodes):
        logger.debug(f"=== Episode {episode+1} start ===", color="magenta")
        ep_rewards, shared_loss = run_episode(args, env, agents, logger, round_score_loss_coef, args.double_positive_rewards)
        if args.shared_networks and shared_loss is not None and shared_opts is not None:
            update_shared(shared_opts[0], shared_opts[1], shared_loss)
        for i in range(args.num_players):
            all_rewards[i].append(ep_rewards[i])
            if writer:
                writer.add_scalar(f"Agent_{i}/Reward", ep_rewards[i], episode)
        logger.info(f"--- Episode {episode+1} complete ---", color="green")
        logger.info(f"Episode rewards: {ep_rewards}", color="green")
    return all_rewards, all_losses
