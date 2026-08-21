"""PPO configurations for state and dual-camera policies."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _algorithm(*, minibatches: int, learning_rate: float, schedule: str) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=minibatches,
        learning_rate=learning_rate,
        schedule=schedule,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SO101StatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 5
    experiment_name = "so101_vial_state"
    run_name = "newton_mjwarp"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = _algorithm(minibatches=4, learning_rate=1.0e-3, schedule="adaptive")


@configclass
class SO101CameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "so101_vial_camera"
    run_name = "newton_renderer"
    obs_groups = {"actor": ["ego_rgb", "external_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="avg",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
