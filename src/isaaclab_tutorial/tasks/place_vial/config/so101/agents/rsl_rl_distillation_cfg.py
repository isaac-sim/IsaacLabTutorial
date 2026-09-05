"""State-teacher to wrist-camera-student distillation configuration."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    WRIST_CAMERA_CNN_CFG,
    BoundedGaussianDistributionCfg,
)


@configclass
class SO101CameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """DAgger-style distillation: the student acts, the loaded state teacher labels every visited state."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 1600
    save_interval = 100
    experiment_name = "so101_vial_camera_distillation"
    run_name = ""
    obs_groups = {"student": ["wrist_rgb", "proprioception"], "teacher": ["teacher_state"]}
    clip_actions = 1.0
    student = RslRlCNNModelCfg(
        cnn_cfg=WRIST_CAMERA_CNN_CFG,
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.05, std_type="log"),
    )
    # Must mirror the state actor so its checkpoint loads; the teacher always acts deterministically.
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianDistributionCfg(init_std=0.2, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="isaaclab_tutorial.tasks.place_vial.config.so101.agents.distillation:BoundedTeacherDistillation",
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
