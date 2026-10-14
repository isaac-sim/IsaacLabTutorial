"""RSL-RL PPO configurations for the state teacher and the from-scratch wrist-camera policy."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


# BEGIN lesson-05-ppo
@configclass
class SO101StatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    pass
# END lesson-05-ppo


@configclass
class SO101CameraPPORunnerCfg(SO101StatePPORunnerCfg):
    """Wrist-camera policy trained from scratch: wrist RGB and proprioception actor, privileged state critic."""

    max_iterations = 2500
    save_interval = 100
    experiment_name = "so101_vial_camera"
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    actor = RslRlCNNModelCfg(
        # Same encoder as the distillation student.
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu"
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )
