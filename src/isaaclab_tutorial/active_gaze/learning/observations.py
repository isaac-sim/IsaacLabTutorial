"""Shared frozen visual preprocessing for native manager-based tasks."""

from isaaclab.managers import ManagerTermBase

from .dino import FrozenDino


class DinoTokens(ManagerTermBase):
    """Encode the monocular eye once per observation, outside PPO minibatches."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene["eye"]
        self.encoder = FrozenDino(**{k: v for k, v in cfg.params.items() if k != "data_type"}).to(env.device).eval()
        self.cached = None

    def __call__(self, env, reference: str, checksum: str | None, batch_size: int, data_type: str = "rgb"):
        if not getattr(env, "refresh_policy_vision", True) and self.cached is not None:
            return self.cached
        image = self.camera.data.output[data_type]
        self.cached = self.encoder(image.torch if hasattr(image, "torch") else image)
        return self.cached
