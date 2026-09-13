"""r2dreamer-compatible environment wrapper.

Matches the interface in NM512/r2dreamer (envs/crafter.py, envs/dmc.py) and
this repo's own pig-runner/env.py: subclasses gym.Env and uses gym.spaces,
but with old-gym call signatures -- reset() returns the obs dict alone,
step() returns a 4-tuple, and the obs dict carries is_first/is_last/is_terminal.

The 1000-step episode cap is NOT implemented here: r2dreamer's
wrappers.TimeLimit owns it (configure time_limit: 1000). We only report
real terminal outcomes (death or surviving the night), via
info["discount"]=0, so hitting the cap stays a truncation.

Deliberately never imports pygame -- ParallelEnv constructs this in
subprocesses.
"""

import gymnasium as gym
import numpy as np

try:  # works both standalone and when copied into r2dreamer's envs/ package
    from .game import NUM_ACTIONS, OBS_DIM, SurvivalGame
except ImportError:
    from game import NUM_ACTIONS, OBS_DIM, SurvivalGame


class SurvivalEnv(gym.Env):
    metadata = {}

    def __init__(self, task="v0", seed=0):
        assert task == "v0", f"only the v0 task exists so far, got {task!r}"
        self._game = SurvivalGame(seed=seed)
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {"state": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)}
        )

    @property
    def action_space(self):
        # make_env wraps this in wrappers.OneHotAction; we receive a plain int.
        return gym.spaces.Discrete(NUM_ACTIONS)

    def _obs(self, is_first=False, is_last=False, is_terminal=False):
        return {
            "state": np.asarray(self._game.observation(), dtype=np.float32),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def reset(self):
        self._game.reset()
        return self._obs(is_first=True)

    def step(self, action):
        reward = self._game.step(int(action))
        # Surviving a full night ends the episode too, same as dying does --
        # both are a real termination, not a truncation, so discount=0 either way.
        done = self._game.dead or self._game.won
        info = {"discount": np.array(0.0 if done else 1.0, dtype=np.float32)}
        obs = self._obs(is_last=done, is_terminal=done)
        return obs, np.float32(reward), done, info

    def render(self):
        try:  # works both standalone and when copied into r2dreamer's envs/ package
            from .render import render_rgb
        except ImportError:
            from render import render_rgb

        return render_rgb(self._game)
