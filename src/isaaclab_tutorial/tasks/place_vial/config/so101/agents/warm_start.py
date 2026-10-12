"""Explicit weight-only PPO initialization using RSL-RL's public load interface."""

from rsl_rl.algorithms import PPO


class FreshOptimizerPPO(PPO):
    """Load actor, critic and iteration while retaining the newly configured optimizer.

    Select this algorithm only for the first warm-start block. Subsequent blocks use
    ordinary PPO so optimizer moments and learning-rate state resume normally.
    """

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if self.rnd:
            raise ValueError("FreshOptimizerPPO does not support RND warm starts")
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "iteration": True, "optimizer": False}
        return super().load(loaded_dict, load_cfg, strict)
