# Add to envs/__init__.py in make_env(), alongside the crafter/pigrunner
# branches. config.task = "survival_v0" splits into suite="survival", task="v0".
#
# NOTE while wiring this in: envs/__init__.py's existing "pigrunner" branch
# is duplicated verbatim (two identical `elif suite == "pigrunner":` blocks) --
# the second one is dead code, Python's if/elif chain always matches the
# first. Worth deleting the duplicate while you're editing this file.

    elif suite == "survival":
        from envs.survival import SurvivalEnv

        env = SurvivalEnv(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)   # env exposes Discrete(6); wrapper one-hots it
