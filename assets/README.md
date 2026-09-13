# Sprites

Drop these PNGs in here (any size — scaled to `world.TILE` at draw time)
and `render.py` picks them up automatically. Missing files fall back to
colored-rectangle placeholders, so the game runs fine before any art exists.

- `character_walk_front.png` — player facing down
- `character_walk_back.png` — player facing up
- `character_walk_side.png` — player facing right (mirrored automatically for left)
- `zombie.png`
- `grass.png`
- `log.png`
- `placed_block.png`
- `lava.png`
- `tree_leaves.png` — canopy of the composed tree structure (see render.py's `_draw_tree`)
- `tree_trunk.png` — trunk of the composed tree structure, drawn under the canopy
- `sky_bg.png` — not wired into the renderer yet (the game is full-bleed top-down grid,
  no sky-visible viewport currently exists); saved here for whenever that changes
