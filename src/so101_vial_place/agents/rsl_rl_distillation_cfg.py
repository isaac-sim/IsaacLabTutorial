"""Wrist-policy distillation from the state-policy teacher."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class SO101CameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """On-policy DAgger-style distillation using privileged state only in the teacher."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "so101_vial_camera_distillation"
    run_name = "wrist_from_state_teacher"
    obs_groups = {
        "student": ["wrist_rgb", "proprioception"],
        "teacher": ["teacher_state"],
    }
    clip_actions = 1.0
    student = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.03, std_type="log"),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
