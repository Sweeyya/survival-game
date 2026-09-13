"""Shared grid/tile constants. No logic here, just numbers game.py and
render.py both need, kept separate so neither has to import the other.
"""

GRID = 16
TILE = 32          # on-screen pixels per cell (2x the 16px source art)
CENTER = GRID // 2  # player spawn cell (both axes)

# --- Tile IDs -----------------------------------------------------------
# A tree isn't one tile, it's several cells built by game.py's
# _tree_cells. TREE_LOG segments are individually breakable; only a tree's
# base log is solid (see game._tree_bases), everything else is walk-through
# and depth-sorted at render time. ZOMBIE/OOB are observation-only stamps,
# never written to the map.
GRASS = 0
LOG = 1        # a pickup lying on the ground; walk over it for +1 inventory
BLOCK = 2      # placed wall
LAVA = 3
TREE_LOG = 4   # one segment of a tree's trunk; breakable -> +1 log
LEAVES = 5     # tree canopy; not breakable
SKY = 6        # top border strip; solid, unbreakable, just a backdrop
ZOMBIE = 7
OOB = 8

NUM_TILE_IDS = 9  # for observation normalization

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
