"""Configuration for language-conditioned visual actors."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class LanguageVisionActorCfg(RslRlMLPModelCfg):
    class_name: str = "isaaclab_tutorial.active_gaze.learning.models:LanguageVisionActor"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="isaaclab_tutorial.active_gaze.learning.distributions:ClippedGaussianDistribution",
        init_std=0.5,
        std_type="log",
    )
    instructions: tuple = ()
    instruction_table_path: str | None = None
    language_device: str = "cpu"
    fusion_dim: int = 128
    fusion_width: int = 256
    fusion_heads: int = 4
    fusion_blocks: int = 2
    spatial_channels: int = 4
