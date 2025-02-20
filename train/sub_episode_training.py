from base_training import run_base_training

def run_sub_episode_training(args, env, agents, logger, writer=None):
    # Using a sub-episode manager to run training.
    from sub_episode_manager import SubEpisodeManager  # Assumes this module exists
    sub_manager = SubEpisodeManager(args, agents, logger)
    all_rewards = [[] for _ in range(args.num_players)]
    all_losses = [[] for _ in range(args.num_players)]
    for episode in range(args.num_episodes):
        logger.info(f"--- Sub-episode {episode+1} start ---", color="magenta")
        loss = sub_manager.run_sub_episode()
        logger.info(f"Sub-episode {episode+1} complete, loss: {loss}", color="green")
        if writer:
            writer.add_scalar("Sub_Episode/Loss", loss, episode)
        for i in range(args.num_players):
            all_rewards[i].append(0)
            all_losses[i].append(loss if loss is not None else 0)
    return all_rewards, all_losses
