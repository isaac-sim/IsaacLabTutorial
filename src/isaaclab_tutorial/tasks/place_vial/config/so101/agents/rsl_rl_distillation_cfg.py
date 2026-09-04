"""State-to-wrist-vision distillation configuration."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class SpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    """Compact CNN that retains feature locations as learned keypoints."""

    class_name: str = "isaaclab_tasks.core.lift.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


@configclass
class GeometrySpatialSoftmaxCNNModelCfg(SpatialSoftmaxCNNModelCfg):
    """Spatial-softmax actor with a training-only geometry prediction head."""

    class_name: str = (
        "isaaclab_tutorial.tasks.place_vial.config.so101.agents.models:GeometrySpatialSoftmaxCNNModel"
    )
    geometry_dim: int = 9


@configclass
class ReplayDAggerAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Bounded dataset aggregation across task phases."""

    class_name: str = "isaaclab_tutorial.tasks.place_vial.config.so101.agents.distillation:ReplayDAggerDistillation"
    teacher_steps: int = 100
    anneal_steps: int = 300
    min_teacher_probability: float = 0.25
    raw_loss_coef: float = 0.1
    bounded_loss_coef: float = 1.0
    gripper_loss_coef: float = 1.0
    replay_capacity: int = 65_536
    replay_insert_per_step: int = 128
    replay_batch_size: int = 1024
    replay_batches_per_update: int = 64
    auxiliary_group: str = "visual_geometry"
    auxiliary_loss_coef: float = 10.0
    swa_start: int = 601
    swa_interval: int = 50


@configclass
class SO101CameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the state teacher into wrist RGB and proprioception."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 800
    save_interval = 25
    experiment_name = "so101_vial_camera_distillation"
    run_name = "replay_dagger_geometry"
    obs_groups = {
        "student": ["wrist_rgb", "proprioception"],
        "teacher": ["teacher_state"],
    }
    clip_actions = 1.0
    student = GeometrySpatialSoftmaxCNNModelCfg(
        cnn_cfg=GeometrySpatialSoftmaxCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
            global_pool="none",
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=GeometrySpatialSoftmaxCNNModelCfg.GaussianDistributionCfg(
            init_std=0.03, std_type="log"
        ),
        init_temperature=1.0,
        geometry_dim=9,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0, std_type="log"),
    )
    algorithm = ReplayDAggerAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
