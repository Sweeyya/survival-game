"""Watch or play the game.

    python play.py                  # play it yourself: arrows/WASD move, SPACE place
    python play.py --mode random    # random policy, the untrained baseline
    python play.py --record run.gif --episodes 1

Keys: WASD/arrows move (also turns you to face that way) | SPACE place block
| R reset | ESC quit

Walk onto a tree's trunk to collect a log and clear the whole tree, no
mining, matching pig-runner's walk-over collectibles. Movement is sampled
every frame from whatever's currently held down
(pygame.key.get_pressed()), not one-shot key-press events, so holding a
direction walks continuously. Place stays a one-shot tap (KEYDOWN) since
one press should place exactly one block, not spam them for as long as
the key is held.

Zombies only spawn at night (see config.DAY_LENGTH / NIGHT_LENGTH); the
screen darkens and the HUD flags it once night falls.
"""

import argparse
import random

import pygame

import game as G
import world as W
from game import SurvivalGame
from render import draw_death_screen, draw_hud, draw_win_screen, draw_world

_KEY_TO_MOVE = {
    pygame.K_UP: G.ACTION_UP, pygame.K_w: G.ACTION_UP,
    pygame.K_DOWN: G.ACTION_DOWN, pygame.K_s: G.ACTION_DOWN,
    pygame.K_LEFT: G.ACTION_LEFT, pygame.K_a: G.ACTION_LEFT,
    pygame.K_RIGHT: G.ACTION_RIGHT, pygame.K_d: G.ACTION_RIGHT,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("human", "random"), default="human")
    ap.add_argument("--seed", type=int, default=None,
                     help="fixed map seed, for a reproducible layout (default: random each run)")
    ap.add_argument("--record", metavar="OUT.gif")
    ap.add_argument("--episodes", type=int, default=0, help="stop after N deaths (0 = forever)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    if args.seed is None:
        print(f"no --seed given, using random seed {seed} (pass --seed {seed} to replay this map)")

    pygame.init()
    pygame.display.set_caption("Survival v0")
    size = W.GRID * W.TILE
    win = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()

    g = SurvivalGame(seed)
    rng = random.Random(seed)
    frames = []
    episodes = 0
    place_pressed = False
    running = True
    episode_counted = False
    end_frames = 0

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    g.reset()
                    episode_counted = False
                    end_frames = 0
                elif ev.key == pygame.K_SPACE:
                    place_pressed = True

        over = g.dead or g.won
        if not over:
            # A real no-op action when nothing's pressed, not skipping
            # step() entirely: the game clock (and DAY_LENGTH/
            # NIGHT_LENGTH, calibrated to steps/sec) has to keep advancing
            # at a steady rate tied to real elapsed frames, or standing
            # idle would silently pause day/night instead of just pausing
            # the player.
            action = G.ACTION_NOOP
            if args.mode == "human":
                if place_pressed:
                    action = G.ACTION_PLACE
                else:
                    keys = pygame.key.get_pressed()
                    for key, act in _KEY_TO_MOVE.items():
                        if keys[key]:
                            action = act
                            break
            else:
                action = rng.randrange(G.NUM_ACTIONS)
            place_pressed = False
            g.step(action)

        draw_world(win, g, end_frames if g.dead else 0)
        draw_hud(win, g)
        if g.won:
            draw_win_screen(win, g, end_frames)
            end_frames += 1
        elif g.dead:
            draw_death_screen(win, g, end_frames)
            end_frames += 1
        pygame.display.flip()

        if args.record is not None:
            import numpy as np
            frames.append(np.transpose(pygame.surfarray.array3d(win), (1, 0, 2)).copy())

        if over and not episode_counted:
            episode_counted = True
            episodes += 1
            outcome = "survived" if g.won else "died"
            print(f"episode {episodes}: score {g.score} in {g.steps} steps ({outcome})")
            if bool(args.episodes) and episodes >= args.episodes:
                running = False
            elif args.mode != "human":
                # No one's there to read the end screen or press R.
                # Random/automated runs (recording, episode batches) just
                # keep going. Human mode waits for an R press instead (see
                # the KEYDOWN handler above) so the end screen is actually
                # visible on screen for a beat.
                g.reset()
                episode_counted = False
                end_frames = 0

        clock.tick(12 if args.mode == "human" else 30)

    if args.record and frames:
        import imageio
        imageio.mimsave(args.record, frames, fps=12, loop=0)
        print(f"wrote {len(frames)} frames -> {args.record}")
    pygame.quit()


if __name__ == "__main__":
    main()
