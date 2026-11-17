"""PPO agents of the active-gaze variants, with gaze supervision by the teacher.

Presets: ``no_gaze_supervision`` turns the teacher's gaze loss off, ``chunking`` makes each decision a plan of three
control actions, ``finetune`` lowers the learning rate to warm-start from a checkpoint, and ``sim2real`` adds the
extrinsics head to the language agent.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg, preset

from .language import INSTRUCTIONS
from .learning.config import LanguageVisionActorCfg


@configclass
class GazeSupervisedPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO with a loss that pulls the gaze actions toward the teacher's."""

    class_name = "isaaclab_tutorial.active_gaze.learning.oracle_ppo:OraclePPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 8
    learning_rate = preset(default=3e-4, finetune=3e-5)
    schedule = "fixed"
    gamma = 0.995
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    oracle_weight: float = preset(default=1.0, no_gaze_supervision=0.0)
    """Weight of the teacher's gaze loss."""
    oracle_std: float = 1e-4
    action_chunk_size: int = preset(default=1, chunking=3)
    """Control actions planned by each policy decision."""


@configclass
class ExtrinsicsPPOCfg(GazeSupervisedPPOCfg):
    """Gaze-supervised PPO that also trains the actor's head to predict the camera mount's move."""

    class_name = "isaaclab_tutorial.active_gaze.pose_ppo:ExtrinsicsPPO"
    extrinsics_weight: float = 0.05


@configclass
class ActiveGazePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """A CNN actor on the retina, proprioception and gaze; an MLP critic on the full state."""

    seed = 42
    # Keeps the rollout near the same control horizon when each decision plans three actions.
    num_steps_per_env = preset(default=64, chunking=21)
    run_name = ""
    clip_actions = 1.0
    finetune: bool = preset(default=False, finetune=True)
    """Warm-start from a checkpoint, allowing the extrinsics head to be new."""
    experiment_name = "so101_vial_active_gaze"
    max_iterations = 5000
    save_interval = 100
    obs_groups = {"actor": ["scene_rgb", "proprioception", "gaze"], "critic": ["critic"]}
    algorithm = GazeSupervisedPPOCfg()
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu"
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )


@configclass
class LanguageActorCfg(LanguageVisionActorCfg):
    instructions = INSTRUCTIONS
    distribution_cfg = ActiveGazePPORunnerCfg().actor.distribution_cfg
    extrinsics_head: bool = preset(default=False, sim2real=True)


@configclass
class LanguagePPORunnerCfg(ActiveGazePPORunnerCfg):
    """A frozen-DINO and language actor; the extrinsics head and its loss come with ``sim2real``."""

    actor = LanguageActorCfg()
    algorithm = preset(
        default=GazeSupervisedPPOCfg(use_mixed_precision=True), sim2real=ExtrinsicsPPOCfg(use_mixed_precision=True)
    )
    obs_groups = {"actor": ["proprio", "vision", "instruction"], "critic": ["critic"]}


@configclass
class ActiveGazeAgentCfg(PresetCfg):
    """Agents of the task variants in :class:`~.env_cfg.ActiveGazeEnvCfg`, selected by the same names."""

    nearest = ActiveGazePPORunnerCfg()
    numbered = nearest
    dino_language = LanguagePPORunnerCfg()
    default = numbered
