"""Shared grid/tile constants. No logic here, just numbers game.py and
render.py both need, kept separate so neither has to import the other.
"""

GRID = 16
TILE = 32          # on-screen pixels per cell (2x the 16px source art)
CENTER = GRID // 2  # player spawn cell (both axes)

# --- Tile IDs -----------------------------------------------------------
# A tree isn't one tile, it's several cells built by game.py's _tree_cells.
# Walking onto an upper trunk segment collects a log and removes that
# segment (see game._apply_cell_effects); only a tree's base log is solid (see
# game._tree_bases), everything else is walk-through and depth-sorted at
# render time. ZOMBIE/OOB are observation-only stamps, never written to
# the map.
GRASS = 0
BLOCK = 1      # placed wall
LAVA = 2
TREE_LOG = 3   # walking onto an upper trunk segment collects that segment
LEAVES = 4     # tree canopy
SKY = 5        # top border strip; solid, just a backdrop
ZOMBIE = 6
OOB = 7

NUM_TILE_IDS = 8  # for observation normalization

SKY_ROWS = 3  # top rows reserved as an impassable "horizon" strip

# --- Directions -----------------------------------------------------------
# Index order also fixes the facing one-hot layout in game.py's observation().
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}
DEFAULT_FACING = DOWN


def in_bounds(x, y):
    return 0 <= x < GRID and 0 <= y < GRID
