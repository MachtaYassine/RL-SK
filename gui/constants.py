"""Colors, sizes, and layout constants for the Skull King GUI."""

# Window
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
BG_COLOR = (30, 30, 40)
TOP_BAR_COLOR = (20, 20, 30)
INFO_BAR_COLOR = (20, 20, 30)
TEXT_COLOR = (240, 240, 240)
DIM_TEXT_COLOR = (150, 150, 150)
HIGHLIGHT_COLOR = (255, 255, 100, 80)
ILLEGAL_OVERLAY = (0, 0, 0, 128)

# Card suit colors
SUIT_COLORS = {
    "YELLOW": (255, 215, 0),
    "GREEN": (34, 139, 34),
    "PURPLE": (139, 0, 139),
    "BLACK": (51, 51, 51),
}

# Special card colors
SPECIAL_COLORS = {
    "PIRATE": (200, 30, 30),
    "SKULL_KING": (200, 30, 30),
    "MERMAID": (0, 200, 200),
    "ESCAPE": (130, 130, 130),
    "TIGRESS": (230, 130, 30),
}

# Card dimensions
CARD_WIDTH = 80
CARD_HEIGHT = 110
CARD_GAP = 10
CARD_RADIUS = 8

# Small card (for AI hands and trick area)
SMALL_CARD_WIDTH = 55
SMALL_CARD_HEIGHT = 75

# Layout positions
TOP_BAR_HEIGHT = 50
INFO_BAR_HEIGHT = 50
HAND_AREA_HEIGHT = 150
HAND_Y = WINDOW_HEIGHT - HAND_AREA_HEIGHT + 15

# Trick area
TRICK_CENTER_X = WINDOW_WIDTH // 2
TRICK_CENTER_Y = WINDOW_HEIGHT // 2 - 20

# Bid selector
BID_BUTTON_SIZE = 50
BID_BUTTON_GAP = 10

# Overlay
OVERLAY_COLOR = (0, 0, 0, 180)

# Animation
TRICK_PAUSE_MS = 1500
ROUND_PAUSE_MS = 2500
AI_CARD_PAUSE_MS = 1200
