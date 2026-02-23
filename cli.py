"""CLI entry point: train / play / evaluate."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Skull King RL")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train agent via self-play")
    train_p.add_argument("--games", type=int, default=100_000)
    train_p.add_argument("--players", type=int, default=4)
    train_p.add_argument("--vary-players", action="store_true",
                         help="Train across 2-8 players each game")
    train_p.add_argument("--min-players", type=int, default=2)
    train_p.add_argument("--max-players", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--batch-size", type=int, default=0,
                         help="PPO batch size (0=auto-scale to GPU)")
    train_p.add_argument("--update-interval", type=int, default=0,
                         help="Games between PPO updates (0=auto-scale with batch)")
    train_p.add_argument("--workers", type=int, default=0,
                         help="Parallel game workers (0=auto, 1=single-threaded)")
    train_p.add_argument("--device", type=str, default="cpu")
    train_p.add_argument("--log-dir", type=str, default="runs")
    train_p.add_argument("--save-dir", type=str, default="checkpoints")
    train_p.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from (default: auto-detect latest.pt)")

    # Play
    play_p = sub.add_parser("play", help="Play interactively against the agent")
    play_p.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    play_p.add_argument("--players", type=int, default=4)

    # GUI
    gui_p = sub.add_parser("gui", help="Play with pygame GUI")
    gui_p.add_argument("--checkpoint", type=str, default=None,
                       help="Neural agent checkpoint (default: use heuristic)")
    gui_p.add_argument("--opponents", type=int, default=3)

    # Evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate agent vs heuristic")
    eval_p.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    eval_p.add_argument("--games", type=int, default=1000)
    eval_p.add_argument("--players", type=int, default=4)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.command == "train":
        from config import PPOConfig, GameConfig, TrainConfig
        from training.trainer import Trainer

        ppo_cfg = PPOConfig(lr=args.lr, batch_size=args.batch_size)
        game_cfg = GameConfig(
            num_players=args.players,
            min_players=args.min_players,
            max_players=args.max_players,
            vary_players=args.vary_players,
        )
        train_cfg = TrainConfig(
            total_games=args.games,
            update_interval=args.update_interval,
            num_workers=args.workers,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
        )
        trainer = Trainer(ppo_cfg, game_cfg, train_cfg, device=args.device)
        resume_path = args.resume
        if resume_path is None:
            default = os.path.join(args.save_dir, "latest.pt")
            if os.path.exists(default):
                resume_path = default
        if resume_path:
            trainer.load_checkpoint(resume_path)
        trainer.train()

    elif args.command == "play":
        from agents.human_agent import HumanAgent
        from agents.neural_agent import NeuralAgent
        from agents.heuristic_agent import HeuristicAgent
        from skull_king.game import SkullKingGame, Phase

        neural = NeuralAgent(args.checkpoint)
        human = HumanAgent()
        heuristics = [HeuristicAgent(seed=i) for i in range(args.players - 2)]

        agents = [human, neural] + heuristics
        game = SkullKingGame(num_players=len(agents))
        game.reset()

        while not game.is_game_over():
            pid = game.get_current_player()
            state = game.get_state(pid)
            agent = agents[pid]

            if game.phase == Phase.BIDDING:
                bid = agent.choose_bid(state)
                game.step_bid(pid, bid)
            elif game.phase == Phase.PLAYING:
                hand_idx, tigress = agent.choose_play(state)
                game.step_play(pid, hand_idx, tigress)

        print("\n=== GAME OVER ===")
        for i, s in enumerate(game.get_scores()):
            label = "You" if i == 0 else f"Player {i}"
            print(f"  {label}: {s}")
        print(f"Winner: {'You' if game.get_winner() == 0 else f'Player {game.get_winner()}'}")

    elif args.command == "gui":
        from agents.heuristic_agent import HeuristicAgent
        from gui.game_manager import GameManager
        from gui.app import run as gui_run

        agents = []
        for i in range(args.opponents):
            if args.checkpoint and os.path.exists(args.checkpoint):
                from agents.neural_agent import NeuralAgent
                agents.append(NeuralAgent(args.checkpoint))
            else:
                agents.append(HeuristicAgent(seed=i))

        manager = GameManager(agents)
        gui_run(manager)

    elif args.command == "evaluate":
        from agents.heuristic_agent import HeuristicAgent
        from networks.bid_network import BidActorCritic
        from networks.play_network import PlayActorCritic
        from networks.features import encode_bid_state, encode_play_state, get_legal_bid_mask, get_legal_play_mask
        from skull_king.cards import SpecialType
        from skull_king.game import SkullKingGame, Phase

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        bid_net = BidActorCritic()
        play_net = PlayActorCritic()
        bid_net.load_state_dict(ckpt["bid_net"])
        play_net.load_state_dict(ckpt["play_net"])
        bid_net.eval()
        play_net.eval()

        wins = 0
        total_score = 0
        for g in range(args.games):
            game = SkullKingGame(num_players=args.players)
            game.reset()
            heuristic = HeuristicAgent()

            while not game.is_game_over():
                pid = game.get_current_player()
                state = game.get_state(pid)

                if game.phase == Phase.BIDDING:
                    if pid == 0:
                        f = encode_bid_state(state)
                        m = get_legal_bid_mask(state)
                        a, _, _, _ = bid_net.get_action_and_value(f, m)
                        game.step_bid(pid, a)
                    else:
                        game.step_bid(pid, heuristic.choose_bid(state))
                elif game.phase == Phase.PLAYING:
                    if pid == 0:
                        f = encode_play_state(state)
                        m = get_legal_play_mask(state)
                        a, _, _, _ = play_net.get_action_and_value(f, m)
                        tig = None
                        if a < len(state.hand) and state.hand[a].special == SpecialType.TIGRESS:
                            tig = True
                        game.step_play(pid, a, tig)
                    else:
                        hi, tig = heuristic.choose_play(state)
                        game.step_play(pid, hi, tig)

            if game.get_winner() == 0:
                wins += 1
            total_score += game.players[0].score

        print(f"Win rate: {wins/args.games:.2%} ({wins}/{args.games})")
        print(f"Avg score: {total_score/args.games:.1f}")
        print(f"ELO from checkpoint: {ckpt.get('player_elo', 'N/A')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
