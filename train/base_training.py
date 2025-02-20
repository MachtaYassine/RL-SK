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
    round_bid_losses = []         # to record bid prediction loss per round
    round_play_head_losses = []   # to record play head loss per round
    shared_loss = None
    done = False
    obs = env.reset()
    last_round = obs["round_number"]  # Track round number

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

        # Detect round transition by monitoring round number change.
        current_round = obs.get("round_number", last_round)
        if current_round > last_round:
            # Use stored round targets from environment.
            targets = env.last_round_tricks if hasattr(env, "last_round_tricks") else env.tricks_won
            for i, ag in enumerate(agents):
                actual_tricks = targets[i]
                bid_pred = ag.bid_prediction if ag.bid_prediction is not None else 0
                # MSE: (bid_pred - actual_tricks)^2 -> e.g., (5-1)^2 == 16.
                bid_loss = F.mse_loss(
                    torch.tensor(bid_pred, dtype=torch.float32),
                    torch.tensor(actual_tricks, dtype=torch.float32)
                )
                # Compute play head loss: compare each prediction to actual_tricks.
                if ag.trick_win_predictons:
                    predictions = torch.tensor(ag.trick_win_predictons, dtype=torch.float32)
                    target = torch.full((len(ag.trick_win_predictons),), actual_tricks, dtype=torch.float32)
                    play_head_loss = F.mse_loss(predictions, target)
                else:
                    play_head_loss = torch.tensor(0.0, dtype=torch.float32)
                logger.info(
                    f"Agent {i}: Predicted bid: {bid_pred} vs Actual tricks: {actual_tricks} | "
                    f"Bid loss: {bid_loss.item()} || "
                    f"Play head predictions: {ag.trick_win_predictons} vs Target: {actual_tricks} | "
                    f"Play head loss: {play_head_loss.item()}",
                    color="blue"
                )
                round_bid_losses.append(bid_loss.item())
                round_play_head_losses.append(play_head_loss.item())
                # Reset predictions for next round.
                ag.bid_prediction = None
                ag.trick_win_predictons.clear()
            last_round = current_round  # update round tracking

    # Return losses along with rewards.
    return episode_rewards, shared_loss, round_bid_losses, round_play_head_losses

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
    all_bid_losses = []         # NEW: to collect bid losses per round across episodes
    all_play_head_losses = []   # NEW: to collect play head losses per round across episodes
    for episode in range(args.num_episodes):
        logger.debug(f"=== Episode {episode+1} start ===", color="magenta")
        ep_rewards, shared_loss, round_bid_losses, round_play_head_losses = run_episode(
            args, env, agents, logger, round_score_loss_coef, args.double_positive_rewards
        )
        if args.shared_networks and shared_loss is not None and shared_opts is not None:
            update_shared(shared_opts[0], shared_opts[1], shared_loss)
        for i in range(args.num_players):
            all_rewards[i].append(ep_rewards[i])
            if writer:
                writer.add_scalar(f"Agent_{i}/Reward", ep_rewards[i], episode)
        # NEW: Log the losses for this episode (round-wise).
        all_bid_losses.extend(round_bid_losses)
        all_play_head_losses.extend(round_play_head_losses)
        logger.info(f"--- Episode {episode+1} complete ---", color="green")
        logger.info(f"Episode rewards: {ep_rewards}", color="green")
    # Pack the loss curves with rewards for plotting.
    return all_rewards, (all_losses, all_bid_losses, all_play_head_losses)
