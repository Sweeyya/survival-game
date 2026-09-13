"""Tune difficulty/content here. No code changes needed elsewhere."""

# --- Map generation (fixed at reset, not regenerated mid-episode) ---------
NUM_TREES = 5           # whole tree shapes never overlap or cover lava
NUM_LAVA_POOLS = 2      # a few contiguous blobs, not scattered single tiles
LAVA_POOL_SIZE = 6

# --- Movement --------------------------------------------------------------
MOVE_SPEED = 0.22    # cells/step, about 4-5 steps to cross one tile
ZOMBIE_SPEED = 0.09  # cells/step, much slower than the player
PLAYER_RADIUS = 0.3  # collision half-width, in cells (player and zombies)

# --- Day / night cycle -------------------------------------------------
# 360 steps = 30 seconds at play.py's human-mode tick rate (12 steps/sec).
DAY_LENGTH = 360
NIGHT_LENGTH = 360

# --- Zombies ---------------------------------------------------------------
ZOMBIE_SPAWN_INTERVAL = 60    # six arrivals over one 360-step night
MAX_ZOMBIES = 6

# --- Resources ---------------------------------------------------------
PLANKS_PER_LOG = 4   # one collected log yields this many usable planks

# --- Episode length ----------------------------------------------------
# Not enforced here, the training harness's TimeLimit wrapper owns it
# (time_limit: 8000); we only ever report a real death via discount=0.
MAX_STEPS = 8000

# --- Rewards -------------------------------------------------------------
# Staircase as adjacent solid sides go 0 -> 4: 1 (first block ever placed,
# see _try_place), 2, 3, 5.
REWARD_FIRST_BLOCK = 1.0
REWARD_TWO_ADJACENT = 2.0
REWARD_THREE_ADJACENT = 3.0
REWARD_FOUR_ADJACENT = 5.0
REWARD_ENCLOSED_TICK = 1.0
ENCLOSED_INTERVAL_STEPS = 10   # while all 4 sides solid, +1 every N steps
REWARD_SURVIVE_NIGHT = 5.0     # one-time reward at dawn, if still alive

# --- Observation -----------------------------------------------------------
WINDOW_RADIUS = 3      # 7x7 window
INVENTORY_CAP = 10.0   # log count normalizes against this
