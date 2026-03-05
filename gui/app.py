"""Main pygame loop: events, state machine, rendering."""

from __future__ import annotations

from typing import Optional

import pygame

from gui.constants import *
from gui.game_manager import GameManager
from gui.renderer import Renderer
from skull_king.game import Phase


class AppState:
    BIDDING = "bidding"
    PLAYING = "playing"
    AI_TURN = "ai_turn"          # Pause to show an AI card
    TRICK_PAUSE = "trick_pause"
    ROUND_OVER = "round_over"
    GAME_OVER = "game_over"
    TIGRESS_PROMPT = "tigress_prompt"


def run(manager: GameManager):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Skull King")
    clock = pygame.time.Clock()
    renderer = Renderer(screen)

    app_state = AppState.BIDDING
    pause_timer = 0
    tigress_hand_index: Optional[int] = None
    last_round_scores = None
    pending_trick_pause = False  # Show AI card before trick pause

    # Fast-forward AI bids at the start (bidding doesn't need per-card viz)
    _advance_ai_bids(manager)

    running = True
    while running:
        dt = clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        clicked = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if app_state == AppState.ROUND_OVER:
                    manager.clear_round_ended()
                    app_state = AppState.BIDDING if not manager.is_game_over() else AppState.GAME_OVER
                    _advance_ai_bids(manager)
                    continue
                if app_state == AppState.GAME_OVER:
                    manager.reset()
                    app_state = AppState.BIDDING
                    _advance_ai_bids(manager)
                    continue
                if app_state == AppState.TIGRESS_PROMPT and tigress_hand_index is not None:
                    if event.key == pygame.K_p:
                        result = manager.human_play(tigress_hand_index, True)
                        tigress_hand_index = None
                        app_state, pause_timer, last_round_scores, pending_trick_pause = _after_human_play(manager, result)
                    elif event.key == pygame.K_e:
                        result = manager.human_play(tigress_hand_index, False)
                        tigress_hand_index = None
                        app_state, pause_timer, last_round_scores, pending_trick_pause = _after_human_play(manager, result)
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True
                if app_state == AppState.ROUND_OVER:
                    manager.clear_round_ended()
                    app_state = AppState.BIDDING if not manager.is_game_over() else AppState.GAME_OVER
                    _advance_ai_bids(manager)
                    clicked = False
                elif app_state == AppState.GAME_OVER:
                    manager.reset()
                    app_state = AppState.BIDDING
                    _advance_ai_bids(manager)
                    clicked = False

        if not running:
            break

        # --- Timer-driven state transitions ---

        if app_state == AppState.AI_TURN:
            pause_timer -= dt
            if pause_timer <= 0:
                if pending_trick_pause:
                    # Card was shown, now do the trick/round pause
                    pending_trick_pause = False
                    if manager.round_just_ended:
                        last_round_scores = manager.last_round_scores
                        app_state = AppState.TRICK_PAUSE
                        pause_timer = TRICK_PAUSE_MS
                    elif manager.is_game_over():
                        app_state = AppState.GAME_OVER
                    else:
                        app_state = AppState.TRICK_PAUSE
                        pause_timer = TRICK_PAUSE_MS
                else:
                    app_state, pause_timer, rs, pending_trick_pause = _step_next_ai_or_human(manager)
                    if rs:
                        last_round_scores = rs

        elif app_state == AppState.TRICK_PAUSE:
            pause_timer -= dt
            if pause_timer <= 0:
                manager.clear_completed_trick()
                if manager.round_just_ended:
                    app_state = AppState.ROUND_OVER
                elif manager.is_game_over():
                    app_state = AppState.GAME_OVER
                elif manager.is_human_turn():
                    app_state = AppState.PLAYING
                else:
                    # More AI cards to play in next trick
                    app_state, pause_timer, rs, pending_trick_pause = _step_next_ai_or_human(manager)
                    if rs:
                        last_round_scores = rs

        # ---- DRAW ----
        screen.fill(BG_COLOR)

        state = manager.get_state()
        scores = manager.get_scores()
        all_bids = manager.get_all_bids()
        all_tricks = manager.get_all_tricks_won()

        renderer.draw_top_bar(manager.round_number, manager.max_rounds, scores)
        renderer.draw_info_bar(all_bids[0], all_tricks[0], all_bids, all_tricks)
        renderer.draw_ai_hands(manager.get_ai_hand_sizes(), manager.num_players)
        renderer.draw_trick_area(manager.get_trick_cards(), manager.num_players)

        if app_state == AppState.BIDDING and manager.is_human_turn():
            renderer.draw_hand(state.hand, [], mouse_pos)
            bid_rects = renderer.draw_bid_selector(manager.round_number, mouse_pos)
            if clicked:
                for rect, bid_val in bid_rects:
                    if rect.collidepoint(mouse_pos):
                        manager.human_bid(bid_val)
                        # Fast-forward remaining AI bids
                        _advance_ai_bids(manager)
                        if manager.phase == Phase.PLAYING:
                            if manager.is_human_turn():
                                app_state = AppState.PLAYING
                            else:
                                # Start stepping AI plays one by one
                                app_state, pause_timer, rs, pending_trick_pause = _step_next_ai_or_human(manager)
                                if rs:
                                    last_round_scores = rs
                        break

        elif app_state == AppState.PLAYING and manager.is_human_turn():
            hand_rects = renderer.draw_hand(state.hand, state.legal_actions, mouse_pos)
            if clicked:
                for rect, hand_idx in hand_rects:
                    if rect.collidepoint(mouse_pos) and hand_idx in state.legal_actions:
                        card = state.hand[hand_idx]
                        if card.is_tigress():
                            tigress_hand_index = hand_idx
                            app_state = AppState.TIGRESS_PROMPT
                        else:
                            result = manager.human_play(hand_idx)
                            app_state, pause_timer, rs, pending_trick_pause = _after_human_play(manager, result)
                            if rs:
                                last_round_scores = rs
                        break
        else:
            # AI_TURN, TRICK_PAUSE, etc. — show hand but non-interactive
            renderer.draw_hand(state.hand, [], mouse_pos)

        if app_state == AppState.TRICK_PAUSE and manager.trick_winner is not None:
            winner = manager.trick_winner
            winner_label = "You" if winner == 0 else f"AI{winner}"
            renderer.draw_message(f"{winner_label} won the trick!")

        if app_state == AppState.TIGRESS_PROMPT:
            renderer.draw_overlay_text(
                ["Play Tigress as:"],
                ["Press [P] for Pirate  or  [E] for Escape"],
            )

        if app_state == AppState.ROUND_OVER:
            rnd = manager.round_number - 1 if not manager.is_game_over() else manager.round_number
            lines = [f"Round {rnd} Complete!"]
            sub = []
            rs = last_round_scores or manager.last_round_scores
            for i, s in enumerate(scores):
                label = "You" if i == 0 else f"AI{i}"
                r = rs[i] if rs and i < len(rs) else 0
                sub.append(f"{label}: {s} (round: {'+' if r >= 0 else ''}{r})")
            sub.append("")
            sub.append("Click or press any key to continue")
            renderer.draw_overlay_text(lines, sub)

        if app_state == AppState.GAME_OVER:
            winner = manager.get_winner()
            winner_label = "You win!" if winner == 0 else f"AI{winner} wins!"
            lines = ["Game Over!", winner_label]
            sub = []
            for i, s in enumerate(scores):
                label = "You" if i == 0 else f"AI{i}"
                sub.append(f"{label}: {s}")
            sub.append("")
            sub.append("Click or press any key to play again")
            renderer.draw_overlay_text(lines, sub)

        pygame.display.flip()

    pygame.quit()


def _advance_ai_bids(manager: GameManager):
    """Fast-forward all AI bids (no per-card viz needed for bidding)."""
    while (not manager.is_game_over()
           and not manager.is_human_turn()
           and manager.phase == Phase.BIDDING):
        manager.advance_ai_single()


def _step_next_ai_or_human(manager):
    """Play one AI card and return state to pause on, or hand off to human.

    Returns (app_state, pause_timer, round_scores_or_none, pending_trick).
    """
    if manager.is_game_over():
        return AppState.GAME_OVER, 0, None, False
    if manager.round_just_ended:
        return AppState.ROUND_OVER, 0, manager.last_round_scores, False
    if manager.is_human_turn():
        return AppState.PLAYING, 0, None, False

    event = manager.advance_ai_single()
    if event is None:
        return AppState.PLAYING, 0, None, False

    if event.get("type") == "game_over":
        return AppState.GAME_OVER, 0, None, False

    round_scores = None
    if event.get("round_over"):
        round_scores = manager.last_round_scores
        # Show the card first, then trick pause later
        return AppState.AI_TURN, AI_CARD_PAUSE_MS, round_scores, True

    if event.get("trick_complete"):
        # Show the completing card first, then trick pause later
        return AppState.AI_TURN, AI_CARD_PAUSE_MS, None, True

    # Normal AI play — pause to show the card
    return AppState.AI_TURN, AI_CARD_PAUSE_MS, None, False


def _after_human_play(manager, trick_result):
    """Returns (app_state, pause_timer, round_scores_or_none, pending_trick)."""
    round_scores = None
    if manager.round_just_ended:
        round_scores = manager.last_round_scores
        return AppState.TRICK_PAUSE, TRICK_PAUSE_MS, round_scores, False
    if manager.is_game_over():
        return AppState.GAME_OVER, 0, round_scores, False
    if trick_result is not None:
        # Trick completed, pause to show all cards + winner
        return AppState.TRICK_PAUSE, TRICK_PAUSE_MS, round_scores, False
    if not manager.is_human_turn():
        # Pause to let human see their card before AI responds
        return AppState.AI_TURN, AI_CARD_PAUSE_MS, None, False
    return AppState.PLAYING, 0, round_scores, False
