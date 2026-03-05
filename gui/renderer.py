"""All pygame drawing logic for the Skull King GUI."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pygame

from skull_king.cards import Card, Suit, SpecialType
from skull_king.game import GameState, Phase
from gui.constants import *


def _suit_name(suit: Suit) -> str:
    return suit.name


def _card_color(card: Card) -> Tuple[int, int, int]:
    if card.is_special():
        return SPECIAL_COLORS.get(card.special.name, (130, 130, 130))
    return SUIT_COLORS.get(card.suit.name, (100, 100, 100))


def _card_label(card: Card) -> Tuple[str, str]:
    """Returns (main_text, sub_text) for a card."""
    if card.is_numbered():
        suit_short = card.suit.name[0]
        return str(card.number), suit_short
    if card.is_special():
        names = {
            SpecialType.ESCAPE: "ESC",
            SpecialType.PIRATE: "PIR",
            SpecialType.MERMAID: "MER",
            SpecialType.SKULL_KING: "S.K.",
            SpecialType.TIGRESS: "TIG",
        }
        if card.special == SpecialType.TIGRESS and card.tigress_as_pirate is not None:
            mode = "PIR" if card.tigress_as_pirate else "ESC"
            return "TIG", mode
        return names.get(card.special, "?"), ""
    return "?", ""


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_med = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_tiny = pygame.font.SysFont("monospace", 12)

    def draw_card(
        self,
        x: int, y: int,
        card: Card,
        w: int = CARD_WIDTH, h: int = CARD_HEIGHT,
        illegal: bool = False,
        hover: bool = False,
    ) -> pygame.Rect:
        """Draw a card rectangle. Returns its rect for click detection."""
        rect = pygame.Rect(x, y, w, h)
        color = _card_color(card)
        pygame.draw.rect(self.screen, color, rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(self.screen, (200, 200, 200), rect, 2, border_radius=CARD_RADIUS)

        main, sub = _card_label(card)
        font = self.font_large if w >= CARD_WIDTH else self.font_med
        sub_font = self.font_small if w >= CARD_WIDTH else self.font_tiny

        # Determine text color (dark bg needs light text, light bg needs dark)
        brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        txt_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

        main_surf = font.render(main, True, txt_color)
        mr = main_surf.get_rect(center=(rect.centerx, rect.centery - 5))
        self.screen.blit(main_surf, mr)

        if sub:
            sub_surf = sub_font.render(sub, True, txt_color)
            sr = sub_surf.get_rect(center=(rect.centerx, rect.centery + 20))
            self.screen.blit(sub_surf, sr)

        if illegal:
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill(ILLEGAL_OVERLAY)
            self.screen.blit(overlay, (x, y))

        if hover:
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill(HIGHLIGHT_COLOR)
            self.screen.blit(overlay, (x, y))

        return rect

    def draw_card_back(self, x: int, y: int, w: int = SMALL_CARD_WIDTH, h: int = SMALL_CARD_HEIGHT):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (60, 60, 90), rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(self.screen, (100, 100, 130), rect, 2, border_radius=CARD_RADIUS)
        # Draw pattern
        pattern = self.font_tiny.render("SK", True, (100, 100, 130))
        pr = pattern.get_rect(center=rect.center)
        self.screen.blit(pattern, pr)

    def draw_top_bar(self, round_num: int, max_rounds: int, scores: List[int]):
        bar = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, TOP_BAR_COLOR, bar)
        pygame.draw.line(self.screen, (60, 60, 80), (0, TOP_BAR_HEIGHT), (WINDOW_WIDTH, TOP_BAR_HEIGHT))

        # Round info
        round_text = self.font_med.render(f"Round {round_num}/{max_rounds}", True, TEXT_COLOR)
        self.screen.blit(round_text, (20, 12))

        # Scores
        parts = []
        for i, s in enumerate(scores):
            label = "You" if i == 0 else f"AI{i}"
            parts.append(f"{label}:{s}")
        score_text = self.font_small.render("  ".join(parts), True, TEXT_COLOR)
        self.screen.blit(score_text, (WINDOW_WIDTH - score_text.get_width() - 20, 16))

    def draw_info_bar(self, bid: int, tricks_won: int, all_bids: List[int], all_tricks: List[int]):
        y = WINDOW_HEIGHT - HAND_AREA_HEIGHT - INFO_BAR_HEIGHT
        bar = pygame.Rect(0, y, WINDOW_WIDTH, INFO_BAR_HEIGHT)
        pygame.draw.rect(self.screen, INFO_BAR_COLOR, bar)
        pygame.draw.line(self.screen, (60, 60, 80), (0, y), (WINDOW_WIDTH, y))

        # Your bid/tricks
        if bid >= 0:
            info = f"Your Bid: {bid}  Won: {tricks_won}/{bid}"
        else:
            info = "Bidding..."
        info_surf = self.font_med.render(info, True, TEXT_COLOR)
        self.screen.blit(info_surf, (20, y + 13))

        # All player bids and tricks won
        parts = []
        for i in range(len(all_bids)):
            label = "You" if i == 0 else f"AI{i}"
            b = all_bids[i] if all_bids[i] >= 0 else "?"
            t = all_tricks[i]
            parts.append(f"{label}: {t}/{b}")
        bids_surf = self.font_small.render("   ".join(parts), True, TEXT_COLOR)
        self.screen.blit(bids_surf, (WINDOW_WIDTH - bids_surf.get_width() - 20, y + 16))

    def draw_hand(
        self,
        hand: List[Card],
        legal_actions: List[int],
        mouse_pos: Tuple[int, int],
    ) -> List[Tuple[pygame.Rect, int]]:
        """Draw human hand. Returns list of (rect, hand_index) for click detection."""
        n = len(hand)
        if n == 0:
            return []
        total_w = n * CARD_WIDTH + (n - 1) * CARD_GAP
        start_x = (WINDOW_WIDTH - total_w) // 2
        y = HAND_Y

        rects = []
        for i, card in enumerate(hand):
            x = start_x + i * (CARD_WIDTH + CARD_GAP)
            legal = i in legal_actions
            hover = legal and pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT).collidepoint(mouse_pos)
            card_y = y - 8 if hover else y
            r = self.draw_card(x, card_y, card, illegal=not legal, hover=hover)
            if hover:
                r = pygame.Rect(x, card_y, CARD_WIDTH, CARD_HEIGHT)
            rects.append((r, i))
        return rects

    def draw_trick_area(self, trick_cards: List[Tuple[int, Card]], num_players: int):
        """Draw cards played in the current trick."""
        if not trick_cards:
            # Empty trick area indicator
            txt = self.font_small.render("Trick Area", True, DIM_TEXT_COLOR)
            self.screen.blit(txt, txt.get_rect(center=(TRICK_CENTER_X, TRICK_CENTER_Y)))
            return

        n = len(trick_cards)
        total_w = n * SMALL_CARD_WIDTH + (n - 1) * CARD_GAP
        start_x = TRICK_CENTER_X - total_w // 2

        for i, (pid, card) in enumerate(trick_cards):
            x = start_x + i * (SMALL_CARD_WIDTH + CARD_GAP)
            y = TRICK_CENTER_Y - SMALL_CARD_HEIGHT // 2

            # Player label above
            label = "You" if pid == 0 else f"AI{pid}"
            lbl_surf = self.font_tiny.render(label, True, DIM_TEXT_COLOR)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(centerx=x + SMALL_CARD_WIDTH // 2, bottom=y - 4))

            self.draw_card(x, y, card, w=SMALL_CARD_WIDTH, h=SMALL_CARD_HEIGHT)

    def draw_ai_hands(self, hand_sizes: List[int], num_players: int):
        """Draw face-down cards for AI players around the table."""
        n_ai = len(hand_sizes)
        if n_ai == 0:
            return

        # Position AI players: top and sides
        positions = self._ai_positions(n_ai)
        for i, (cx, cy, horizontal) in enumerate(positions):
            size = hand_sizes[i]
            if size == 0:
                continue
            if horizontal:
                total = size * (SMALL_CARD_WIDTH + 3)
                sx = cx - total // 2
                for j in range(size):
                    self.draw_card_back(sx + j * (SMALL_CARD_WIDTH + 3), cy, SMALL_CARD_WIDTH, SMALL_CARD_HEIGHT)
            else:
                total = size * 20
                sy = cy - total // 2
                for j in range(size):
                    self.draw_card_back(cx, sy + j * 20, SMALL_CARD_WIDTH, SMALL_CARD_HEIGHT)

            label = f"AI{i + 1}"
            lbl = self.font_small.render(label, True, TEXT_COLOR)
            if horizontal:
                self.screen.blit(lbl, lbl.get_rect(centerx=cx, top=cy + SMALL_CARD_HEIGHT + 5))
            else:
                self.screen.blit(lbl, lbl.get_rect(centerx=cx + SMALL_CARD_WIDTH // 2, top=cy + total // 2 + SMALL_CARD_HEIGHT // 2 + 5))

    def _ai_positions(self, n_ai: int) -> List[Tuple[int, int, bool]]:
        """Returns (center_x, top_y, horizontal) for each AI player."""
        positions = []
        if n_ai == 1:
            positions.append((WINDOW_WIDTH // 2, TOP_BAR_HEIGHT + 15, True))
        elif n_ai == 2:
            positions.append((150, TRICK_CENTER_Y - 80, False))
            positions.append((WINDOW_WIDTH - 150 - SMALL_CARD_WIDTH, TRICK_CENTER_Y - 80, False))
        elif n_ai == 3:
            positions.append((150, TRICK_CENTER_Y - 80, False))
            positions.append((WINDOW_WIDTH // 2, TOP_BAR_HEIGHT + 15, True))
            positions.append((WINDOW_WIDTH - 150 - SMALL_CARD_WIDTH, TRICK_CENTER_Y - 80, False))
        elif n_ai >= 4:
            positions.append((150, TRICK_CENTER_Y - 80, False))
            # Spread remaining across top
            top_count = n_ai - 2
            spacing = (WINDOW_WIDTH - 400) // (top_count + 1)
            for j in range(top_count):
                positions.append((200 + spacing * (j + 1), TOP_BAR_HEIGHT + 15, True))
            positions.append((WINDOW_WIDTH - 150 - SMALL_CARD_WIDTH, TRICK_CENTER_Y - 80, False))
        return positions

    def draw_bid_selector(self, max_bid: int, mouse_pos: Tuple[int, int]) -> List[Tuple[pygame.Rect, int]]:
        """Draw bid buttons. Returns (rect, bid_value) pairs."""
        n = max_bid + 1
        total_w = n * BID_BUTTON_SIZE + (n - 1) * BID_BUTTON_GAP
        start_x = (WINDOW_WIDTH - total_w) // 2
        y = TRICK_CENTER_Y - BID_BUTTON_SIZE // 2

        # Label
        label = self.font_med.render("Choose your bid:", True, TEXT_COLOR)
        self.screen.blit(label, label.get_rect(centerx=WINDOW_WIDTH // 2, bottom=y - 15))

        rects = []
        for i in range(n):
            x = start_x + i * (BID_BUTTON_SIZE + BID_BUTTON_GAP)
            rect = pygame.Rect(x, y, BID_BUTTON_SIZE, BID_BUTTON_SIZE)
            hover = rect.collidepoint(mouse_pos)
            color = (80, 80, 140) if hover else (50, 50, 80)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, (120, 120, 160), rect, 2, border_radius=6)
            txt = self.font_large.render(str(i), True, TEXT_COLOR)
            self.screen.blit(txt, txt.get_rect(center=rect.center))
            rects.append((rect, i))
        return rects

    def draw_overlay_text(self, lines: List[str], sub_lines: Optional[List[str]] = None):
        """Draw a centered overlay with text (for round end, game over, etc.)."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

        total_lines = len(lines) + (len(sub_lines) if sub_lines else 0)
        start_y = WINDOW_HEIGHT // 2 - total_lines * 20

        for i, line in enumerate(lines):
            surf = self.font_med.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, surf.get_rect(centerx=WINDOW_WIDTH // 2, y=start_y + i * 35))

        if sub_lines:
            sub_start = start_y + len(lines) * 35 + 15
            for i, line in enumerate(sub_lines):
                surf = self.font_small.render(line, True, DIM_TEXT_COLOR)
                self.screen.blit(surf, surf.get_rect(centerx=WINDOW_WIDTH // 2, y=sub_start + i * 25))

    def draw_message(self, text: str):
        """Draw a temporary message at center."""
        surf = self.font_med.render(text, True, TEXT_COLOR)
        bg = pygame.Rect(0, 0, surf.get_width() + 40, surf.get_height() + 20)
        bg.center = (WINDOW_WIDTH // 2, TRICK_CENTER_Y + 80)
        pygame.draw.rect(self.screen, (40, 40, 60), bg, border_radius=8)
        self.screen.blit(surf, surf.get_rect(center=bg.center))
