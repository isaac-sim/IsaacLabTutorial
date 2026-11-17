"""Frozen DINO tokens of the foveated scene camera's retina."""

from isaaclab.managers import ManagerTermBase

from .dino import FrozenDino


class DinoTokens(ManagerTermBase):
    """Encode the retina once per policy decision, outside PPO minibatches.

    With ``photometric`` ranges (see :mod:`isaaclab_tutorial.active_gaze.sim2real`), the retina's appearance varies
    per episode before encoding.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene["eye"]
        encoder_args = {k: v for k, v in cfg.params.items() if k not in ("data_type", "photometric")}
        self.encoder = FrozenDino(**encoder_args).to(env.device).eval()
        self.photometric = None
        if cfg.params.get("photometric") is not None:
            from isaaclab_tutorial.active_gaze.sim2real import PhotometricVariation

            self.photometric = PhotometricVariation(env.num_envs, env.device, cfg.params["photometric"])
        self.cached = None

    def reset(self, env_ids=None):
        if self.photometric is not None:
            self.photometric.reset(env_ids)

    def __call__(
        self, env, reference: str, checksum: str | None, batch_size: int, data_type: str = "rgb", photometric=None
    ):
        if not getattr(env, "refresh_policy_vision", True) and self.cached is not None:
            return self.cached
        image = self.camera.data.output[data_type].torch
        if self.photometric is not None:
            image = self.photometric(image)
        self.cached = self.encoder(image)
        return self.cached
