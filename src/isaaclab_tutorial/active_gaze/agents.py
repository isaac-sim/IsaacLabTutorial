"""Class-based SO-101 policy, loss, and optimizer presets."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_tutorial.active_gaze.learning.config import LanguageVisionActorCfg

from .language import INSTRUCTIONS


@configclass
class LearningRateCfg(PresetCfg):
    default = 3e-4
    finetune = 3e-5


@configclass
class FineTuneCfg(PresetCfg):
    default = False
    finetune = True


@configclass
class PPOCfg(RslRlPpoAlgorithmCfg):
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 8
    learning_rate = LearningRateCfg()
    schedule = "adaptive"
    gamma = 0.995
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class ActionChunkSizeCfg(PresetCfg):
    """Number of control actions predicted by each policy invocation."""

    default = 1
    chunking = 3


@configclass
class RolloutLengthCfg(PresetCfg):
    """Keep the rollout near the default control horizon with three-action chunks."""

    default = 64
    chunking = 21


@configclass
class OraclePPOCfg(PPOCfg):
    class_name = "isaaclab_tutorial.active_gaze.learning.oracle_ppo:OraclePPO"
    oracle_weight: float = 1.0
    oracle_std: float = 1e-4
    action_chunk_size = ActionChunkSizeCfg()
    schedule = "fixed"


@configclass
class GazeWeightCfg(PresetCfg):
    grasp_gated = 1.0
    no_gaze_supervision = 0.0
    default = grasp_gated


@configclass
class ExtrinsicsPPOCfg(OraclePPOCfg):
    class_name = "isaaclab_tutorial.active_gaze.pose_ppo:ExtrinsicsPPO"
    oracle_weight = GazeWeightCfg()
    extrinsics_weight: float = 0.05


@configclass
class UnsupervisedAlgorithmCfg(PresetCfg):
    default = PPOCfg()
    chunking = OraclePPOCfg(oracle_weight=0.0)


@configclass
class GazeAlgorithmCfg(PresetCfg):
    grasp_gated = OraclePPOCfg()
    no_gaze_supervision = UnsupervisedAlgorithmCfg()
    default = grasp_gated


@configclass
class AlgorithmCfg(PresetCfg):
    default = GazeAlgorithmCfg()
    extrinsics = ExtrinsicsPPOCfg()


@configclass
class ExtrinsicsHeadCfg(PresetCfg):
    default = False
    extrinsics = True


@configclass
class SO101FoveatedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = RolloutLengthCfg()
    run_name = ""
    clip_actions = 1.0
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    finetune = FineTuneCfg()
    experiment_name = "so101_vial_foveated"
    max_iterations = 5000
    save_interval = 100
    obs_groups = {"actor": ["scene_rgb", "proprioception", "gaze"], "critic": ["critic"]}
    algorithm = AlgorithmCfg()
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="elu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(
            init_std=0.2,
            std_type="log",
            std_range=(0.05, 0.3),
        ),
    )


@configclass
class SO101LanguageActorCfg(LanguageVisionActorCfg):
    instructions = INSTRUCTIONS
    distribution_cfg = SO101FoveatedPPORunnerCfg().actor.distribution_cfg
    extrinsics_head = ExtrinsicsHeadCfg()


@configclass
class LanguageUnsupervisedAlgorithmCfg(PresetCfg):
    default = PPOCfg(use_mixed_precision=True)
    chunking = OraclePPOCfg(oracle_weight=0.0, use_mixed_precision=True)


@configclass
class LanguageGazeAlgorithmCfg(GazeAlgorithmCfg):
    grasp_gated = OraclePPOCfg(use_mixed_precision=True)
    no_gaze_supervision = LanguageUnsupervisedAlgorithmCfg()
    default = grasp_gated


@configclass
class LanguageAlgorithmCfg(AlgorithmCfg):
    default = LanguageGazeAlgorithmCfg()
    extrinsics = ExtrinsicsPPOCfg(use_mixed_precision=True)


@configclass
class SO101LanguagePPORunnerCfg(SO101FoveatedPPORunnerCfg):
    actor = SO101LanguageActorCfg()
    algorithm = LanguageAlgorithmCfg()
    obs_groups = {"actor": ["proprio", "vision", "instruction"], "critic": ["critic"]}


@configclass
class SO101Sim2RealPPORunnerCfg(SO101LanguagePPORunnerCfg):
    actor = SO101LanguageActorCfg(extrinsics_head=True)
    algorithm = ExtrinsicsPPOCfg(use_mixed_precision=True)


@configclass
class SO101AgentCfg(PresetCfg):
    nearest = SO101FoveatedPPORunnerCfg()
    numbered = nearest
    dino_language = SO101LanguagePPORunnerCfg()
    planar = dino_language
    sim2real = SO101Sim2RealPPORunnerCfg()
    default = numbered
