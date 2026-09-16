"""Core game logic. Pure Python: no pygame, no gym, no RL dependencies. This
module is imported inside ParallelEnv subprocesses, so it must stay
import-light (mirrors pig-runner/game.py).

Positions are real (x, y) floats, crossing one grid tile takes several
steps at config.MOVE_SPEED cells/step, not one full cell per action.
Collision is a small box (config.PLAYER_RADIUS half-width) checked against
the grid, axis-separated (X then Y) so bumping a wall at an angle slides
along it instead of fully stopping. Anything inherently cell-based (which
tile you're facing, the observation window, the enclosure-adjacency count)
reads "current cell" as floor(px), floor(py); only rendering uses the raw
float position.

Facing/movement uses the standard "bump to turn" idiom: any movement action
always updates facing to that direction, and only actually moves the player
if the target cell is free. This is what makes self-enclosure possible with
just 4 move actions + place, you can turn to face a blocked direction
without stepping into it, then place there.
"""

import math
import random

try:  # works both standalone and if this ever gets copied into r2dreamer's envs/
    from . import config as C
    from . import world as W
except ImportError:
    import config as C
    import world as W

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PLACE, ACTION_NOOP = 0, 1, 2, 3, 4, 5
NUM_ACTIONS = 6
_MOVE_ACTION_TO_DIR = {
    ACTION_UP: W.UP,
    ACTION_DOWN: W.DOWN,
    ACTION_LEFT: W.LEFT,
    ACTION_RIGHT: W.RIGHT,
}
# Tiles that unconditionally block movement. TREE_LOG/LEAVES are
# deliberately not here, only a tree's base log blocks (see
# self._tree_bases / _box_blocked); everything else about a tree is
# walk-through and purely depth-sorted at render time.
_SOLID_TILES = (W.BLOCK, W.SKY)

# A tree = a stack of TREE_LOGS_TALL log cells (the trunk, growing north/up
# from a base cell) capped with a leaf canopy. Walking onto an upper trunk
# segment collects four planks and removes that one segment.
TREE_LOGS_TALL = 2
# A 3-3-1 pyramid sitting directly on the top log: two full 3-wide rows,
# tapering to a single leaf at the apex.
_LEAF_OFFSETS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, -2), (0, -2), (1, -2),
    (0, -3),
)  # relative to the top log

# Rows a tree spans above its own base (trunk + however far the apex leaf
# reaches past the top log), used by _scatter_trees for sky clearance.
_TREE_CLEARANCE = (TREE_LOGS_TALL - 1) - min(dy for _, dy in _LEAF_OFFSETS)


def _tree_cells(x, y):
    """All (cx, cy) -> tile_id cells for a tree whose base (bottom) log sits
    at (x, y). Caller is responsible for bounds/overlap checking every cell
    this returns before committing any of them to the grid."""
    cells = {(x, y - i): W.TREE_LOG for i in range(TREE_LOGS_TALL)}
    top_y = y - (TREE_LOGS_TALL - 1)
    for dx, dy in _LEAF_OFFSETS:
        cells[(x + dx, top_y + dy)] = W.LEAVES
    return cells

# window(49) + inventory(1) + zombie dx/dy/count(3) + facing one-hot(4)
# + cycle phase/night flag(2) + adjacent-solid count(1)
OBS_DIM = (2 * C.WINDOW_RADIUS + 1) ** 2 + 1 + 3 + 4 + 2 + 1
NO_ZOMBIE_SENTINEL = 1.5  # matches pig-runner's DIST_CLIP_HI "nothing there" idiom


class SurvivalGame:
    """The game itself. Advance it with step(action); read state off the attrs."""

    def __init__(self, seed=0, curriculum=False, curriculum_step_offset=0):
        self._rng = random.Random(seed)
        self._curriculum = curriculum
        self._lifetime_steps = curriculum_step_offset
        self.reset()

    def _settings_for_episode(self):
        """Return this episode's training settings. The game itself owns a
        local lifetime counter because r2dreamer does not pass its global
        counter into environments. With 16 parallel environments, the two
        thresholds in config correspond roughly to 75k and 150k global
        training steps."""
        if not self._curriculum:
            return {
                "starting_planks": 0, "nearby_tree": False, "zombies": True,
                "lava_pools": C.NUM_LAVA_POOLS, "require_enclosure": False,
            }
        settings = C.CURRICULUM_STAGES[0]
        for candidate in C.CURRICULUM_STAGES:
            if self._lifetime_steps >= candidate["after_env_steps"]:
                settings = candidate
        return settings

    # lifecycle
    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)

        self._episode_settings = self._settings_for_episode()
        self.grid = [[W.GRASS] * W.GRID for _ in range(W.GRID)]
        for y in range(W.SKY_ROWS):
            for x in range(W.GRID):
                self.grid[y][x] = W.SKY
        # Centered in the cell (CENTER + 0.5), not at its corner, position
        # is a real float from here on, see the module docstring.
        self.px, self.py = W.CENTER + 0.5, W.CENTER + 0.5
        self.facing = W.DEFAULT_FACING
        self.inventory = self._episode_settings["starting_planks"]
        self.zombies = []  # list of [x, y] floats

        # A 3x3 clearance around spawn, not just the player's own cell.
        # Reserving only the center cell still let something land on 3 of
        # its 4 neighbors often enough to box the player in on step one.
        px0, py0 = int(self.px), int(self.py)
        occupied = {
            (px0 + dx, py0 + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        occupied.update((x, y) for y in range(W.SKY_ROWS) for x in range(W.GRID))
        self._tree_bases = set()  # which TREE_LOG cells are actually solid; see _box_blocked
        if self._episode_settings["nearby_tree"]:
            self._place_nearby_tree(occupied)
        self._scatter_trees(C.NUM_TREES, occupied)
        self._scatter_lava_pools(self._episode_settings["lava_pools"], C.LAVA_POOL_SIZE, occupied)

        self.steps = 0
        self.dead = False
        self.won = False  # set once a full night is survived; see step()
        self.score = 0  # logs collected + blocks placed, a simple HUD number
        self._placed_first_block = False
        self._rewarded_adjacent = set()
        self._enclosed_streak = 0
        self._next_zombie_spawn = C.ZOMBIE_SPAWN_INTERVAL
        return self.observation()

    def _place_nearby_tree(self, occupied):
        """Guarantee one reachable log during the collection lesson.
        Its upper trunk segment is a few moves right and down from spawn,
        while the base stays a real obstacle like every other tree."""
        x, y = W.CENTER + 3, W.CENTER + 2
        cells = _tree_cells(x, y)
        if any(not W.in_bounds(cx, cy) or (cx, cy) in occupied for cx, cy in cells):
            return
        for (cx, cy), tile in cells.items():
            self.grid[cy][cx] = tile
            occupied.add((cx, cy))
        self._tree_bases.add((x, y))

    @property
    def is_night(self):
        """Cycle = DAY_LENGTH + NIGHT_LENGTH steps; night is the back half."""
        cycle = C.DAY_LENGTH + C.NIGHT_LENGTH
        return (self.steps % cycle) >= C.DAY_LENGTH

    def _scatter_lava_pools(self, num_pools, pool_size, occupied):
        """Lava as a handful of contiguous blobs: each pool grows from a
        random seed cell by repeatedly annexing a random free neighbor of
        an already-claimed cell, giving an organic/irregular shape rather
        than a perfect square (render._draw_lava rounds its outer corners
        to read as one connected pool). Only spreads onto plain GRASS, not
        just cells absent from `occupied`. Occupied only tracks tree
        *bases* now (see _scatter_trees), so a grid-value check is what
        actually stops lava from overwriting a tree's leaves/logs outright.
        That's real tile replacement, not a rendering z-order question:
        no depth sort can put back a leaf that's been turned into lava."""
        placed_pools = 0
        attempts = 0
        while placed_pools < num_pools and attempts < num_pools * 50:
            attempts += 1
            sx, sy = self._rng.randrange(W.GRID), self._rng.randrange(W.SKY_ROWS, W.GRID)
            if (sx, sy) in occupied or self.grid[sy][sx] != W.GRASS:
                continue
            pool = {(sx, sy)}
            frontier = [(sx, sy)]
            while len(pool) < pool_size and frontier:
                cx, cy = frontier.pop(self._rng.randrange(len(frontier)))
                neighbors = [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)]
                self._rng.shuffle(neighbors)
                for nx, ny in neighbors:
                    if len(pool) >= pool_size:
                        break
                    if (
                        not W.in_bounds(nx, ny) or ny < W.SKY_ROWS
                        or (nx, ny) in occupied or (nx, ny) in pool
                        or self.grid[ny][nx] != W.GRASS
                    ):
                        continue
                    pool.add((nx, ny))
                    frontier.append((nx, ny))
            for x, y in pool:
                self.grid[y][x] = W.LAVA
                occupied.add((x, y))
            placed_pools += 1

    def _scatter_trees(self, count, occupied):
        """Drop `count` whole tree structures (see _tree_cells). Every cell
        of the shape must land on free grass, or the whole tree is skipped
        and another random spot is tried, letting trees overlap (an
        earlier version of this allowed it, checking only the base cell)
        let one tree's cells silently overwrite another's, corrupting both:
        a base could vanish under a neighboring tree's leaves, or vice
        versa. With NUM_TREES this low, requiring the whole shape to be
        free still fits comfortably."""
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 50:
            attempts += 1
            x = self._rng.randrange(1, W.GRID - 1)
            y = self._rng.randrange(W.SKY_ROWS + _TREE_CLEARANCE, W.GRID)
            cells = _tree_cells(x, y)
            if any(not W.in_bounds(cx, cy) or (cx, cy) in occupied for cx, cy in cells):
                continue
            for (cx, cy), tile in cells.items():
                self.grid[cy][cx] = tile
                occupied.add((cx, cy))
            self._tree_bases.add((x, y))
            placed += 1

    # simulation
    def step(self, action):
        """Advance one tick. Returns reward earned this tick."""
        if self.dead or self.won:
            return 0.0
        reward = 0.0
        was_night = self.is_night

        if action == ACTION_PLACE:
            reward += self._try_place()
        elif action in _MOVE_ACTION_TO_DIR:
            self._move(_MOVE_ACTION_TO_DIR[action])
        # ACTION_NOOP, and any defensive out-of-range value, leave the world
        # alone while its clock and zombies continue advancing.

        reward += self._apply_cell_effects()
        if self._touching_zombie():
            self.dead = True

        if not self.dead and self._episode_settings["zombies"]:
            self._spawn_zombies_if_due()
            self._move_zombies()
            if self._touching_zombie():
                self.dead = True

        reward += self._adjacency_reward()

        self.steps += 1
        self._lifetime_steps += 1
        # Night just ended (was_night flips to not-night at the top of a
        # new cycle) and the player's still alive to see it: that's a win.
        if was_night and not self.is_night and not self.dead:
            self.won = True
            if (
                not self._episode_settings["require_enclosure"]
                or self._adjacent_solid_count() >= 4
            ):
                reward += C.REWARD_SURVIVE_NIGHT
        return reward

    def _move(self, direction):
        self.facing = direction
        dx, dy = W.DIRS[direction]
        self.px, self.py = self._slide(
            self.px, self.py, dx * C.MOVE_SPEED, dy * C.MOVE_SPEED
        )

    def _slide(self, x, y, dx, dy, ignore_tree_base=False):
        """Move (x, y) by (dx, dy) with axis-separated collision: try X
        alone and keep it if the player-radius box at the new X doesn't
        overlap a solid/OOB cell, then the same for Y, lets bumping a
        wall at an angle slide along it instead of fully stopping. Shared
        by the player and every zombie (see ignore_tree_base)."""
        if dx and not self._box_blocked(x + dx, y, ignore_tree_base):
            x += dx
        if dy and not self._box_blocked(x, y + dy, ignore_tree_base):
            y += dy
        return x, y

    def _box_blocked(self, cx, cy, ignore_tree_base=False):
        """True if a PLAYER_RADIUS-wide box centered at (cx, cy) overlaps
        any solid tile or goes out of bounds. Uses floor (not int/truncate)
        so a box straying slightly negative still checks the right
        off-grid cell instead of silently rounding it up to cell 0.
        ignore_tree_base lets zombies float straight through trees (their
        one real obstacle is a wall you actually build, not the scenery)
        while the player still can't."""
        r = C.PLAYER_RADIUS
        x0, x1 = math.floor(cx - r), math.floor(cx + r)
        y0, y1 = math.floor(cy - r), math.floor(cy + r)
        for gy in range(y0, y1 + 1):
            for gx in range(x0, x1 + 1):
                if not W.in_bounds(gx, gy):
                    return True
                tile = self.grid[gy][gx]
                if tile in _SOLID_TILES:
                    return True
                # A tree's base log is its one real ground-level obstacle;
                # every other TREE_LOG/LEAVES cell is walk-through.
                if tile == W.TREE_LOG and not ignore_tree_base and (gx, gy) in self._tree_bases:
                    return True
        return False

    def _apply_cell_effects(self):
        """Whatever cell the player is currently standing in: LAVA kills;
        walking onto an upper tree-trunk tile collects one log and removes
        that segment, no mining, matching
        pig-runner's walk-over collectibles. One log yields
            C.PLANKS_PER_LOG usable planks and a small collection reward
            (placing still only costs 1, see
            _try_place). Checked every step, not just after a move, since with
            continuous position "did you move to a new cell" isn't the only
            way to still be standing in one."""
        x, y = int(self.px), int(self.py)
        if not W.in_bounds(x, y):
            return 0.0
        tile = self.grid[y][x]
        if tile == W.TREE_LOG:
            self.grid[y][x] = W.GRASS
            self.inventory += C.PLANKS_PER_LOG
            self.score += 1
            return C.REWARD_COLLECT_LOG
        elif tile == W.LAVA:
            self.dead = True
        return 0.0

    def _try_place(self):
        px, py = int(self.px), int(self.py)
        dx, dy = W.DIRS[self.facing]
        tx, ty = px + dx, py + dy
        if not W.in_bounds(tx, ty):
            return 0.0
        if self.inventory <= 0 or self.grid[ty][tx] != W.GRASS or self._zombie_at_cell(tx, ty):
            return 0.0

        self.grid[ty][tx] = W.BLOCK
        self.inventory -= 1
        self.score += 1
        reward = 0.0
        if not self._placed_first_block:
            self._placed_first_block = True
            reward += C.REWARD_FIRST_BLOCK
        return reward

    def _adjacent_solid_count(self):
        px, py = int(self.px), int(self.py)
        count = 0
        for direction in (W.UP, W.DOWN, W.LEFT, W.RIGHT):
            dx, dy = W.DIRS[direction]
            x, y = px + dx, py + dy
            # off the edge of the grid counts as solid, a corner is a
            # legitimate, cheaper place to build an enclosure
            if not W.in_bounds(x, y) or self.grid[y][x] == W.BLOCK:
                count += 1
        return count

    def _adjacency_reward(self):
        """Fires the one-time +2/+3/+5 threshold rewards (staircase:
        1 for the first block ever placed, see _try_place, then 2, 3, 5 as
        adjacent solid sides reach 2, 3, 4) and the recurring enclosed-tick
        reward. Adjacency only ever changes by +-1 per step (one placement
        at a time), so 0->1->2->3->4 transitions are never skipped."""
        count = self._adjacent_solid_count()
        reward = 0.0
        for threshold, bonus in (
            (2, C.REWARD_TWO_ADJACENT),
            (3, C.REWARD_THREE_ADJACENT),
            (4, C.REWARD_FOUR_ADJACENT),
        ):
            if count >= threshold and threshold not in self._rewarded_adjacent:
                self._rewarded_adjacent.add(threshold)
                reward += bonus

        if count >= 4:
            self._enclosed_streak += 1
            if self._enclosed_streak % C.ENCLOSED_INTERVAL_STEPS == 0:
                reward += C.REWARD_ENCLOSED_TICK
        else:
            self._enclosed_streak = 0
        return reward

    # zombies
    def _zombie_at_cell(self, x, y):
        """Cell-based zombie check, for things that are inherently about a
        discrete grid cell (place targeting, spawn-point selection) rather
        than continuous position (see _touching_zombie for that)."""
        return any(int(zx) == x and int(zy) == y for zx, zy in self.zombies)

    def _touching_zombie(self):
        """Continuous-position death check: player and zombies are both
        circles of radius PLAYER_RADIUS, so "touching" is a center-distance
        check against the sum of both radii, not an exact cell match."""
        r2 = (2 * C.PLAYER_RADIUS) ** 2
        return any((zx - self.px) ** 2 + (zy - self.py) ** 2 < r2 for zx, zy in self.zombies)

    def _spawn_zombies_if_due(self):
        # Zombies only spawn at night. While it's day we don't advance the
        # spawn timer either, so the moment night falls the first spawn
        # happens right away rather than waiting out a stale countdown.
        if not self.is_night:
            return
        if self.steps < self._next_zombie_spawn or len(self.zombies) >= C.MAX_ZOMBIES:
            if len(self.zombies) >= C.MAX_ZOMBIES:
                self._next_zombie_spawn = self.steps + C.ZOMBIE_SPAWN_INTERVAL
            return
        self._next_zombie_spawn = self.steps + C.ZOMBIE_SPAWN_INTERVAL

        px, py = int(self.px), int(self.py)
        edge_cells = [
            (x, y)
            for x in range(W.GRID)
            for y in range(W.GRID)
            if (x in (0, W.GRID - 1) or y in (0, W.GRID - 1))
            and self.grid[y][x] not in _SOLID_TILES
            and not self._zombie_at_cell(x, y)
            and (x, y) != (px, py)
        ]
        if edge_cells:
            ex, ey = self._rng.choice(edge_cells)
            self.zombies.append([ex + 0.5, ey + 0.5])  # centered, like the player

    def _move_zombies(self):
        """Steer straight toward the player's live continuous position at
        C.ZOMBIE_SPEED, real chasing, not a grid-snapped walk. Same
        box-collision as the player via _slide, except zombies float
        straight through trees (ignore_tree_base) so a tree can't be used
        to permanently block a chase, only a placed BLOCK wall can. Lava
        isn't a collision obstacle for anyone (see _apply_cell_effects),
        it's fatal instead, so a zombie that steps into it just dies
        there rather than getting stuck."""
        survivors = []
        for z in self.zombies:
            dx, dy = self.px - z[0], self.py - z[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                dx, dy = dx / dist * C.ZOMBIE_SPEED, dy / dist * C.ZOMBIE_SPEED
                z[0], z[1] = self._slide(z[0], z[1], dx, dy, ignore_tree_base=True)
            zx, zy = int(z[0]), int(z[1])
            if W.in_bounds(zx, zy) and self.grid[zy][zx] == W.LAVA:
                continue
            survivors.append(z)
        self.zombies = survivors

    # observation
    def local_window(self):
        """(2R+1)x(2R+1) tile IDs centered on the player's current cell
        (floor of its continuous position), row-major, with zombies stamped
        in (they aren't part of the static grid) and off-grid cells stamped
        W.OOB."""
        r = C.WINDOW_RADIUS
        px, py = int(self.px), int(self.py)
        window = []
        for wy in range(py - r, py + r + 1):
            for wx in range(px - r, px + r + 1):
                if not W.in_bounds(wx, wy):
                    window.append(W.OOB)
                elif self._zombie_at_cell(wx, wy):
                    window.append(W.ZOMBIE)
                else:
                    window.append(self.grid[wy][wx])
        return window

    def _nearest_zombie_offset(self):
        if not self.zombies:
            return NO_ZOMBIE_SENTINEL, NO_ZOMBIE_SENTINEL
        zx, zy = min(
            self.zombies,
            key=lambda z: (z[0] - self.px) ** 2 + (z[1] - self.py) ** 2,
        )
        dx = max(-1.0, min(1.0, (zx - self.px) / W.GRID))
        dy = max(-1.0, min(1.0, (zy - self.py) / W.GRID))
        return dx, dy

    def observation(self):
        """The numbers the agent sees. Never pixels."""
        window_norm = [t / (W.NUM_TILE_IDS - 1) for t in self.local_window()]
        inventory_norm = min(1.0, self.inventory / C.INVENTORY_CAP)
        zdx, zdy = self._nearest_zombie_offset()
        zombie_count_norm = min(1.0, len(self.zombies) / C.MAX_ZOMBIES)
        facing_onehot = [1.0 if self.facing == d else 0.0 for d in (W.UP, W.DOWN, W.LEFT, W.RIGHT)]
        cycle = C.DAY_LENGTH + C.NIGHT_LENGTH
        cycle_phase = (self.steps % cycle) / cycle
        night = 1.0 if self.is_night else 0.0
        adjacent_norm = self._adjacent_solid_count() / 4.0

        return (
            window_norm
            + [float(inventory_norm), float(zdx), float(zdy), float(zombie_count_norm)]
            + facing_onehot
            + [float(cycle_phase), night, float(adjacent_norm)]
        )
