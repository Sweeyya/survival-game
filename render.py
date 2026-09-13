"""Visual renderer. Imported only when something actually wants pixels;
never by the training path, which runs many envs in subprocesses.

Draws colored rectangles by default. Drop the named PNGs into assets/ and
they're picked up automatically, no renderer changes needed (mirrors
pig-runner/render.py's sprite-or-fallback pattern).
"""

import os

import numpy as np
import pygame

try:  # works both standalone and when copied into r2dreamer's envs/ package
    from . import config as C
    from . import world as W
except ImportError:
    import config as C
    import world as W

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

GRASS_COL = (106, 170, 75)
BLOCK_COL = (120, 120, 128)
BLOCK_EDGE = (80, 80, 88)
LAVA_COL = (214, 69, 33)
LAVA_GLOW = (255, 150, 60)
TREE_LOG_COL = (94, 62, 33)
LEAVES_COL = (46, 110, 46)
PLAYER_COL = (240, 210, 90)
ZOMBIE_COL = (76, 140, 76)
ZOMBIE_DARK = (40, 90, 40)
GRID_LINE = (0, 0, 0, 25)
TEXT = (250, 250, 250)
TEXT_BG = (0, 0, 0)
NIGHT_TINT = (10, 15, 45, 150)  # RGBA; alpha-blended over the whole scene
SKY_COL = (185, 235, 245)  # fallback if sky_bg.png isn't there

# The player, trunk logs and leaves all draw taller than one tile, anchored
# at the BOTTOM of their own cell and extending UPWARD. That direction is
# what makes the fence illusion work: something rising off its own cell
# covers a player standing in the row(s) above it (farther from the
# viewer, walking "behind" it) while a player below it overlaps its base
# and draws in front. Only a tree's base log is actually solid (see
# game._tree_bases), everything else about a tree is walk-through, with
# this height-plus-sort doing the work of making that still look right.
PLAYER_HEIGHT_TILES = 1.5
LEAF_HEIGHT_TILES = 1.1   # kept close to one cell, the 3-3-1 canopy gets
                          # its depth from 3 real grid rows, not overhang
TRUNK_HEIGHT_TILES = 1.8
LEAF_ALPHA = 215  # a little see-through, per Minecraft's own leaves

# Per-type depth, on top of ground-contact row: bigger = further from the
# camera (drawn first, so painted over); smaller = closer (drawn last, on
# top). Everything defaults to 0, a plain row/continuous-position
# comparison, and stays there. Ground/lava need no depth constant at
# all: they're drawn in the flat base layer before this sort ever runs, so
# they're always behind every depth-sorted thing unconditionally.
#
# Giving TREE_LOG or LEAVES a nonzero depth here was tried (DEPTH_LOG=-1,
# DEPTH_LEAVES=-2, to make a canopy sit in front of the trunk it caps) and
# reverted: any such shift isn't scoped to that one relationship, it
# applies to EVERY comparison that type takes part in. DEPTH_LOG=-1 meant
# a player had to stand a full extra row south of a tree's base before
# actually covering it, and DEPTH_LEAVES=-2 broke the same way against
# leaves. See _leaf_cap_key for how the leaf-caps-its-log look is achieved
# instead, without touching the general player/zombie comparisons.
DEPTH_LOG = 0
DEPTH_LEAVES = 0
DEPTH_PLAYER = 0
DEPTH_ZOMBIE = 0

# See _leaf_cap_key: a leaf directly above the log it caps needs a small,
# narrowly-scoped boost to draw after (in front of) that ONE log, despite
# being one row closer, the log is exactly one row south, which on pure
# position always outweighs an equal-depth leaf by exactly 1.
LEAF_CAP_BIAS = 1.2

# character_walk_side is authored facing right; mirrored horizontally for
# left-facing draws so only one asset is needed for both.
_FACING_SPRITE = {
    W.UP: "character_walk_back",
    W.DOWN: "character_walk_front",
    W.LEFT: "character_walk_side",
    W.RIGHT: "character_walk_side",
}
_TILE_SPRITE = {
    W.GRASS: "grass",
    W.BLOCK: "placed_block",
    W.LAVA: "lava",
    # TREE_LOG/LEAVES aren't here: they draw taller than one cell (see
    # _draw_trunk/_draw_leaves), never through this flat-tile path.
}

_sprite_cache = {}


def load_sprite(name):
    """Return a Surface for assets/<name>.png, or None if it isn't there yet."""
    if name in _sprite_cache:
        return _sprite_cache[name]
    path = os.path.join(ASSET_DIR, name + ".png")
    surf = pygame.image.load(path).convert_alpha() if os.path.exists(path) else None
    _sprite_cache[name] = surf
    return surf


_transform_cache = {}


def _cached(key, compute):
    if key not in _transform_cache:
        _transform_cache[key] = compute()
    return _transform_cache[key]


def _scaled(name, size):
    """Hard-edged scale to an exact (w, h) box. scale(), not smoothscale():
    the source art is tiny pixel art meant to blow up crisp, bilinear
    filtering would blur it into a smeared, un-pixel-art look."""
    sprite = load_sprite(name)
    if sprite is None:
        return None
    return _cached(("scaled", name, size), lambda: pygame.transform.scale(sprite, size))


def _scaled_by_height(name, target_h):
    """Like _scaled, but locks to a target height and derives width from
    the source's own aspect ratio instead of forcing a fixed box, the
    character sprites are narrow in their source art, and stretching them
    to a fixed width squashed them. Callers center the result themselves
    since its width now varies by sprite."""
    sprite = load_sprite(name)
    if sprite is None:
        return None
    w0, h0 = sprite.get_size()
    target_w = max(1, round(w0 * target_h / h0))
    return _cached(
        ("scaled_h", name, target_w, target_h),
        lambda: pygame.transform.scale(sprite, (target_w, target_h)),
    )


def _tall_sprite(sprite_name, h):
    """sprite_name.png built up to (W.TILE, h) without non-uniform
    stretching: the bottom W.TILE is the sprite at its own uniform scale
    (identical to a normal tile), and any extra height is filled by
    re-cropping a strip off the TOP of the source art at that same scale.
    Stretching the whole sprite taller instead would visibly warp the wood
    grain/leaf texture into an elongated shape."""
    def build():
        src = load_sprite(sprite_name)
        if src is None:
            return None
        sw, sh = src.get_size()
        base_square = pygame.transform.scale(src, (W.TILE, W.TILE))
        extra = h - W.TILE
        canvas = pygame.Surface((W.TILE, max(h, W.TILE)), pygame.SRCALPHA)
        if extra > 0:
            crop_h = min(sh, max(1, round(extra * sh / W.TILE)))
            crop = src.subsurface((0, 0, sw, crop_h))
            canvas.blit(pygame.transform.scale(crop, (W.TILE, extra)), (0, 0))
            canvas.blit(base_square, (0, extra))
        else:
            canvas.blit(base_square, (0, 0))
        return canvas

    return _cached(("tall", sprite_name, h), build)


def _draw_tile_fallback(surf, tile, x, y):
    # TREE_LOG/LEAVES never reach here, see _TILE_SPRITE.
    rect = (x, y, W.TILE, W.TILE)
    if tile == W.GRASS:
        pygame.draw.rect(surf, GRASS_COL, rect)
    elif tile == W.BLOCK:
        pygame.draw.rect(surf, BLOCK_COL, rect)
        pygame.draw.rect(surf, BLOCK_EDGE, rect, 3)
    elif tile == W.LAVA:
        pygame.draw.rect(surf, LAVA_COL, rect)
        pygame.draw.circle(surf, LAVA_GLOW, (x + W.TILE // 3, y + W.TILE // 2), 4)
        pygame.draw.circle(surf, LAVA_GLOW, (x + 2 * W.TILE // 3, y + W.TILE // 3), 3)


def _draw_tile(surf, tile, x, y):
    name = _TILE_SPRITE.get(tile)
    sprite = _scaled(name, (W.TILE, W.TILE)) if name else None
    if sprite is not None:
        surf.blit(sprite, (x, y))
    else:
        _draw_tile_fallback(surf, tile, x, y)


def _draw_lava(surf, g, gx, gy):
    """Lava as pools, not a grid of hard squares: a corner only stays
    square where it touches another lava cell (so adjacent tiles meet
    edge-to-edge), every other corner rounds off, an isolated tile reads
    as a blob, a cluster reads as one organic pool. A dark offset shadow
    and a darker rim (same rounded silhouette) sell it as a shallow
    depression instead of a flat-colored sticker on the grass."""
    x, y = gx * W.TILE, gy * W.TILE
    _draw_tile(surf, W.GRASS, x, y)  # ground the rounded-off corners reveal

    def is_lava(nx, ny):
        return W.in_bounds(nx, ny) and g.grid[ny][nx] == W.LAVA

    up, down = is_lava(gx, gy - 1), is_lava(gx, gy + 1)
    left, right = is_lava(gx - 1, gy), is_lava(gx + 1, gy)
    r = W.TILE // 3
    radii = dict(
        border_top_left_radius=0 if (up or left) else r,
        border_top_right_radius=0 if (up or right) else r,
        border_bottom_left_radius=0 if (down or left) else r,
        border_bottom_right_radius=0 if (down or right) else r,
    )

    shadow = pygame.Surface((W.TILE, W.TILE), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (20, 10, 5, 90), shadow.get_rect(), **radii)
    surf.blit(shadow, (x + 3, y + 3))

    lava_sprite = _scaled("lava", (W.TILE, W.TILE))
    if lava_sprite is None:
        lava_sprite = pygame.Surface((W.TILE, W.TILE), pygame.SRCALPHA)
        lava_sprite.fill(LAVA_COL)
        pygame.draw.circle(lava_sprite, LAVA_GLOW, (W.TILE // 3, W.TILE // 2), 4)
        pygame.draw.circle(lava_sprite, LAVA_GLOW, (2 * W.TILE // 3, W.TILE // 3), 3)

    mask = pygame.Surface((W.TILE, W.TILE), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), **radii)
    mask.blit(lava_sprite, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(mask, (x, y))

    # Rim as 4 independent edges, not one rect outline: a rect's outline
    # always strokes all 4 sides, which drew a visible seam at every
    # internal boundary between two adjacent lava tiles instead of just
    # around the pool's actual outer silhouette.
    rim = pygame.Surface((W.TILE, W.TILE), pygame.SRCALPHA)
    rim_col = (90, 30, 10, 160)
    if not up:
        pygame.draw.line(rim, rim_col, (0, 0), (W.TILE, 0), 2)
    if not down:
        pygame.draw.line(rim, rim_col, (0, W.TILE - 1), (W.TILE, W.TILE - 1), 2)
    if not left:
        pygame.draw.line(rim, rim_col, (0, 0), (0, W.TILE), 2)
    if not right:
        pygame.draw.line(rim, rim_col, (W.TILE - 1, 0), (W.TILE - 1, W.TILE), 2)
    surf.blit(rim, (x, y))


def _draw_tall(surf, cx, cy, sprite_name, height_tiles, alpha, fallback_col):
    """Shared by leaves and trunk logs: taller than one cell, anchored at
    the bottom of its own cell and extending upward (see _tall_sprite)."""
    x = cx * W.TILE
    base_y = (cy + 1) * W.TILE
    h = round(W.TILE * height_tiles)
    draw_y = base_y - h
    sprite = _tall_sprite(sprite_name, h)
    if sprite is not None:
        if alpha < 255:
            sprite = sprite.copy()
            sprite.set_alpha(alpha)
        surf.blit(sprite, (x, draw_y))
        return
    overlay = pygame.Surface((W.TILE, h), pygame.SRCALPHA)
    pygame.draw.rect(overlay, (*fallback_col, alpha), (2, h - W.TILE + 2, W.TILE - 4, W.TILE - 4))
    surf.blit(overlay, (x, draw_y))


def _draw_leaves(surf, cx, cy):
    _draw_tall(surf, cx, cy, "tree_leaves", LEAF_HEIGHT_TILES, LEAF_ALPHA, LEAVES_COL)


def _draw_trunk(surf, cx, cy, is_base=False):
    if is_base:
        # Only the base sits at actual ground level, one shadow per
        # tree, not one per log segment.
        _draw_shadow(surf, cx * W.TILE + W.TILE / 2, (cy + 1) * W.TILE, W.TILE * 0.55)
    _draw_tall(surf, cx, cy, "tree_trunk", TRUNK_HEIGHT_TILES, 255, TREE_LOG_COL)


def _depth_key(ground_contact_row, depth):
    """Sort key combining position with DEPTH_*: bigger depth pulls the
    key smaller (drawn earlier/further back), smaller depth pulls it
    bigger (drawn later/closer to camera). See the DEPTH_* comment."""
    return ground_contact_row - depth


def _leaf_cap_key(g, gx, gy):
    """Sort key for a LEAVES cell, plus whether it's capping a log
    directly below it (same column, one row south, see
    game._LEAF_OFFSETS' (0, -1)). Scoped to just that one pair instead of
    a general DEPTH_LEAVES: see the DEPTH_* comment for why a blanket
    per-type depth broke the general player/zombie comparisons."""
    is_cap = gy + 1 < W.GRID and g.grid[gy + 1][gx] == W.TREE_LOG
    key = _depth_key(gy + 1, DEPTH_LEAVES)
    if is_cap:
        key += LEAF_CAP_BIAS
    return key, is_cap


_sky_strip_cache = None


def _draw_sky_strip(surf):
    """The top W.SKY_ROWS rows, drawn once as a single stretched backdrop.
    sky_bg.png is one continuous gradient/glow image, so tiling it
    cell-by-cell would repeat the glow once per column. It's a soft
    painted image, not pixel art, so smoothscale is the right filter here."""
    global _sky_strip_cache
    size = (W.GRID * W.TILE, W.SKY_ROWS * W.TILE)
    if _sky_strip_cache is None or _sky_strip_cache.get_size() != size:
        src = load_sprite("sky_bg")
        if src is not None:
            _sky_strip_cache = pygame.transform.smoothscale(src, size)
        else:
            _sky_strip_cache = pygame.Surface(size)
            _sky_strip_cache.fill(SKY_COL)
    surf.blit(_sky_strip_cache, (0, 0))


HIT_FLASH_FRAMES = 8  # how many render frames the colliding zombie stays flashed


def draw_world(surf, g, dead_frames=0):
    _draw_sky_strip(surf)

    # Base layer: every cell draws itself flat, grass included. LEAVES and
    # TREE_LOG don't belong here, both draw taller than their cell as
    # part of the depth-sorted pass below, so GRASS stands in for them
    # (the ground they visually rise up off of). SKY is skipped entirely:
    # the strip above already covers that whole area in one piece.
    for gy in range(W.GRID):
        for gx in range(W.GRID):
            tile = g.grid[gy][gx]
            if tile == W.SKY:
                continue
            if tile == W.LAVA:
                _draw_lava(surf, g, gx, gy)
                continue
            _draw_tile(
                surf, W.GRASS if tile in (W.LEAVES, W.TREE_LOG) else tile,
                gx * W.TILE, gy * W.TILE,
            )

    # Grid lines only over the walkable ground, not the sky backdrop, the
    # row at W.SKY_ROWS still gets a line, marking the horizon.
    grid_overlay = pygame.Surface((W.GRID * W.TILE, W.GRID * W.TILE), pygame.SRCALPHA)
    top = W.SKY_ROWS * W.TILE
    for i in range(W.GRID + 1):
        pygame.draw.line(grid_overlay, GRID_LINE, (i * W.TILE, top), (i * W.TILE, W.GRID * W.TILE))
    for i in range(W.SKY_ROWS, W.GRID + 1):
        pygame.draw.line(grid_overlay, GRID_LINE, (0, i * W.TILE), (W.GRID * W.TILE, i * W.TILE))
    surf.blit(grid_overlay, (0, 0))

    # Depth sort: every drawable that visually reaches outside its own cell
    # is drawn in order of position (plain row for tree cells, continuous
    # position for the player/zombies), so whatever's further back draws
    # first and whatever's closer to the camera draws over it. The one
    # exception is a cap leaf, which gets a small scoped boost against the
    # specific log it caps (see _leaf_cap_key); the tuple's second element
    # breaks the resulting tie in a fixed order. Only a tree's base log
    # actually blocks movement, every other TREE_LOG/LEAVES cell is
    # walk-through, so this sort is doing real work, not just decorating
    # something already solid.
    hit_r2 = (2 * C.PLAYER_RADIUS) ** 2
    flashing = g.dead and dead_frames < HIT_FLASH_FRAMES
    drawables = []
    for gy in range(W.GRID):
        for gx in range(W.GRID):
            tile = g.grid[gy][gx]
            if tile == W.LEAVES:
                key, _ = _leaf_cap_key(g, gx, gy)
                drawables.append(((key, 1), lambda cx=gx, cy=gy: _draw_leaves(surf, cx, cy)))
            elif tile == W.TREE_LOG:
                is_base = (gx, gy) in g._tree_bases
                key = _depth_key(gy + 1, DEPTH_LOG)
                drawables.append(((key, 0), lambda cx=gx, cy=gy, base=is_base: _draw_trunk(surf, cx, cy, base)))
    for zx, zy in g.zombies:
        # Whichever zombie is actually touching the player when dead
        # flashes white, a "that's what got you" cue.
        hit = flashing and (zx - g.px) ** 2 + (zy - g.py) ** 2 < hit_r2
        key = _depth_key(zy, DEPTH_ZOMBIE)
        drawables.append(((key, 2), lambda zx=zx, zy=zy, hit=hit: _draw_zombie(surf, zx, zy, hit)))
    drawables.append(((_depth_key(g.py, DEPTH_PLAYER), 3), lambda: _draw_player(surf, g)))
    drawables.sort(key=lambda d: d[0])
    for _, draw_fn in drawables:
        draw_fn()

    draw_night_overlay(surf, g)


def _draw_shadow(surf, cx_px, cy_px, w):
    """A flat, square-edged shadow at a character's own ground-contact
    point, under its sprite. A soft ellipse read as out of place next to
    Minecraft's own blocky look, so this is a plain rect, not a circle."""
    h = w * 0.4
    shadow = pygame.Surface((round(w), round(h)), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 90))
    surf.blit(shadow, (cx_px - w / 2, cy_px - h / 2))


def _draw_zombie(surf, zx, zy, hit=False):
    # zx, zy are the zombie's real continuous (x, y), zx * TILE, zy *
    # TILE is where the zombie IS. zombie.png is taller than it is wide in
    # its source art, so it gets the same aspect-preserving height scale
    # as the player.
    _draw_shadow(surf, zx * W.TILE, zy * W.TILE, W.TILE * 0.45)
    x, y = zx * W.TILE - W.TILE / 2, zy * W.TILE - W.TILE / 2
    sprite = _scaled_by_height("zombie", W.TILE)
    if sprite is not None:
        if hit:
            # additive white flash for whichever zombie is touching the
            # player on the death frame.
            sprite = sprite.copy()
            flash = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            flash.fill((255, 255, 255, 190))
            sprite.blit(flash, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        surf.blit(sprite, (zx * W.TILE - sprite.get_width() / 2, y))
        return
    col = (255, 255, 255) if hit else ZOMBIE_COL
    pygame.draw.rect(surf, col, (x + 4, y + 4, W.TILE - 8, W.TILE - 8))
    pygame.draw.rect(surf, ZOMBIE_DARK, (x + 4, y + 4, W.TILE - 8, W.TILE - 8), 3)


def _draw_player(surf, g):
    # g.px, g.py are the player's real continuous (x, y), px * TILE, py *
    # TILE is the literal pixel point where the player IS. The sprite is
    # centered horizontally on that point and anchored there as the feet
    # line, extending upward by its own height.
    base_y = g.py * W.TILE
    h = round(W.TILE * PLAYER_HEIGHT_TILES)
    draw_y = base_y - h
    cx = g.px * W.TILE
    x = cx - W.TILE / 2
    _draw_shadow(surf, cx, base_y, W.TILE * 0.5)
    name = _FACING_SPRITE[g.facing]
    sprite = _scaled_by_height(name, h)
    if sprite is not None:
        if g.facing == W.LEFT:
            sprite = pygame.transform.flip(sprite, True, False)
        surf.blit(sprite, (cx - sprite.get_width() / 2, draw_y))
        return
    pygame.draw.rect(surf, PLAYER_COL, (x + 6, draw_y + 4, W.TILE - 12, h - 8))
    # facing wedge so the fallback art still communicates direction
    cx, cy = x + W.TILE // 2, draw_y + h // 2
    dx, dy = W.DIRS[g.facing]
    tip = (cx + dx * W.TILE // 2, cy + dy * h // 2)
    pygame.draw.polygon(surf, (200, 60, 60), [tip, (cx - 5, cy), (cx + 5, cy)])


_PIXEL_FONT_PT = 8  # rendered tiny, then hard-scaled up, see _pixel_text
_pixel_font_cache = None


def _pixel_font():
    global _pixel_font_cache
    if _pixel_font_cache is None:
        _pixel_font_cache = pygame.font.SysFont(
            "menlo,monaco,consolas,monospace", _PIXEL_FONT_PT, bold=True
        )
    return _pixel_font_cache


def _pixel_text(text, color, scale, bg=None):
    """Blocky, Minecraft-ish text: render tiny with antialiasing off (no
    smooth gray edges to begin with), then blow it up with a hard nearest-
    neighbor scale, same crisp-pixel-art trick _scaled uses for sprites,
    applied to text instead of an image file."""
    surf = _pixel_font().render(text, False, color, bg) if bg else _pixel_font().render(text, False, color)
    w, h = surf.get_size()
    return pygame.transform.scale(surf, (w * scale, h * scale))


def draw_night_overlay(surf, g):
    """Darken the whole scene at night. Folded into draw_world so
    play.py's live window and render_rgb's headless frames stay identical."""
    if not g.is_night:
        return
    overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    overlay.fill(NIGHT_TINT)
    surf.blit(overlay, (0, 0))


def draw_hud(surf, g):
    phase = "night, zombies spawning" if g.is_night else "day"
    lines = [
        f"step {g.steps}  score {g.score}  [{phase}]",
        f"zombies {len(g.zombies)}",
    ]
    y = 6
    for line in lines:
        img = _pixel_text(line, TEXT, scale=2, bg=TEXT_BG)
        surf.blit(img, (6, y))
        y += img.get_height() + 2
    _draw_inventory_slot(surf, g.inventory)


def _draw_inventory_slot(surf, planks):
    """One deliberately simple inventory slot: a blocky framed square, the
    plank texture used for placing, and its stack count. It communicates the
    only resource the game has without presenting a whole unused inventory."""
    size, margin = 42, 6
    x, y = surf.get_width() - size - margin, margin
    pygame.draw.rect(surf, (38, 38, 38), (x, y, size, size))
    pygame.draw.rect(surf, (120, 120, 120), (x + 2, y + 2, size - 4, size - 4), 2)
    pygame.draw.rect(surf, (18, 18, 18), (x + 5, y + 5, size - 10, size - 10))
    plank = _scaled("placed_block", (24, 24))
    if plank is not None:
        surf.blit(plank, (x + 9, y + 8))
    else:
        pygame.draw.rect(surf, BLOCK_COL, (x + 10, y + 9, 22, 22))
    count = _pixel_text(str(planks), TEXT, scale=2, bg=TEXT_BG)
    surf.blit(count, (x + size - count.get_width() - 4, y + size - count.get_height() - 3))


def draw_death_screen(surf, g, dead_frames=0):
    """Called by play.py once g.dead is True, on top of the final frame.
    Frame 0 is a bright white impact flash (text withheld so it doesn't
    compete for contrast); after that it settles into the steady dark-red
    tint with the title."""
    w, h = surf.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    if dead_frames == 0:
        overlay.fill((255, 255, 255, 210))
        surf.blit(overlay, (0, 0))
        return
    overlay.fill((30, 0, 0, 165))
    surf.blit(overlay, (0, 0))

    title = _pixel_text("YOU DIED", (230, 60, 60), scale=5)
    sub = _pixel_text(f"score {g.score}, {g.steps} steps, press R", TEXT, scale=2)
    surf.blit(title, title.get_rect(center=(w // 2, h // 2 - 16)))
    surf.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 26)))


def draw_win_screen(surf, g, win_frames=0):
    """Called by play.py once g.won is True (survived a full night), same
    beat structure as draw_death_screen but a bright-green success flash
    settling into a steady green tint instead of red."""
    w, h = surf.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    if win_frames == 0:
        overlay.fill((255, 255, 255, 210))
        surf.blit(overlay, (0, 0))
        return
    overlay.fill((10, 40, 10, 150))
    surf.blit(overlay, (0, 0))

    title = _pixel_text("SURVIVED", (90, 230, 100), scale=5)
    sub = _pixel_text(f"score {g.score}, {g.steps} steps, press R", TEXT, scale=2)
    surf.blit(title, title.get_rect(center=(w // 2, h // 2 - 16)))
    surf.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 26)))


def render_rgb(g):
    """Headless RGB frame (H, W, 3) for recording/observation-debug."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    size = W.GRID * W.TILE
    surf = pygame.Surface((size, size))
    draw_world(surf, g)
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
