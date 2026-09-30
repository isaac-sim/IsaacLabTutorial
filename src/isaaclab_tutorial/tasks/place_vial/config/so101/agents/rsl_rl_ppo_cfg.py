"""RSL-RL PPO configurations for the state and wrist-camera tasks."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian policy head whose standard deviation is clamped to ``std_range``.

    With an entropy bonus and no cap, PPO inflates the std without bound once actions are clipped to [-1, 1]: the
    state policy drifted past 1.5 and the visual policy past 10. Noise that large random-walks the gripper by
    centimetres over the approach, so a policy that has not yet learned to reach the vial never can. The cap keeps
    the exploration bonus while holding the noise at a level where the approach is still learnable.
    """

    std_range: tuple[float, float] = (0.05, 0.3)


WRIST_CAMERA_CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[16, 32, 32],
    kernel_size=[5, 3, 3],
    stride=[2, 2, 2],
    activation="elu",
)
"""Compact encoder for the 64 x 48 wrist image, shared by the visual PPO actor and the distillation student."""

PPO_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=8,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class SO101StatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """State teacher: fully observed actor, privileged critic."""

    seed = 42
    num_steps_per_env = 64
    max_iterations = 800
    save_interval = 50
    experiment_name = "so101_vial_state"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianDistributionCfg(init_std=0.2, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = PPO_ALGORITHM_CFG


@configclass
class SO101CameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Visual policy trained from scratch: wrist RGB and proprioception actor, privileged state critic."""

    seed = 42
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 100
    experiment_name = "so101_vial_camera"
    run_name = ""
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlCNNModelCfg(
        cnn_cfg=WRIST_CAMERA_CNN_CFG,
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianDistributionCfg(init_std=0.2, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = PPO_ALGORITHM_CFG
