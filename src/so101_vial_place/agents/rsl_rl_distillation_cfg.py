"""Wrist-policy distillation from the state-policy teacher."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from .rsl_rl_ppo_cfg import _geometry_spatial_camera_actor, _spatial_camera_actor


@configclass
class RslRlGeometryDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration fields for the auxiliary geometry loss."""

    geometry_loss_coef: float = 1.0
    geometry_group: str = "visual_geometry"


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


@configclass
class SO101WideCameraDistillationRunnerCfg(SO101CameraDistillationRunnerCfg):
    """Capacity ablation for the same compact 64x64 vision boundary."""

    experiment_name = "so101_vial_camera_distillation_wide"
    run_name = "wide_wrist_from_state_teacher"
    student = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.03, std_type="log"),
    )


@configclass
class SO101WideTeacherRolloutDistillationRunnerCfg(SO101WideCameraDistillationRunnerCfg):
    """Wide student trained first on coherent teacher-controlled rollouts."""

    experiment_name = "so101_vial_camera_distillation_teacher_rollout"
    run_name = "wide_teacher_rollout"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:TeacherRolloutDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101SpatialCameraDistillationRunnerCfg(SO101CameraDistillationRunnerCfg):
    """DAgger student using spatial-softmax image coordinates."""

    experiment_name = "so101_vial_camera_distillation_spatial"
    run_name = "spatial_keypoints_from_state_teacher"
    student = _spatial_camera_actor(init_std=0.03)


@configclass
class SO101SpatialTeacherRolloutDistillationRunnerCfg(SO101SpatialCameraDistillationRunnerCfg):
    """Coherent teacher-rollout warm start for the spatial student."""

    experiment_name = "so101_vial_camera_distillation_spatial_teacher_rollout"
    run_name = "spatial_keypoints_teacher_rollout"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:TeacherRolloutDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101SpatialFineDistillationRunnerCfg(SO101SpatialCameraDistillationRunnerCfg):
    """Checkpoint-dense low-rate DAgger for the spatial student."""

    save_interval = 10
    experiment_name = "so101_vial_camera_distillation_spatial_fine"
    run_name = "spatial_keypoints_low_rate"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FineTuneDistillation",
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101SpatialDenseDistillationRunnerCfg(SO101SpatialCameraDistillationRunnerCfg):
    """Unchanged spatial DAgger with denser checkpoints for exact selection."""

    save_interval = 10
    experiment_name = "so101_vial_camera_distillation_spatial_dense"
    run_name = "spatial_keypoints_dense_checkpoints"


@configclass
class SO101SpatialPeakSearchDistillationRunnerCfg(SO101SpatialDenseDistillationRunnerCfg):
    """Short one-save-per-update search around an exact-evaluated peak."""

    save_interval = 1
    experiment_name = "so101_vial_camera_distillation_spatial_peak_search"
    run_name = "spatial_keypoints_peak_search"


@configclass
class SO101GeometrySpatialDistillationRunnerCfg(SO101SpatialCameraDistillationRunnerCfg):
    """Spatial DAgger with one compact gripper-frame geometry target."""

    experiment_name = "so101_vial_camera_distillation_geometry_spatial"
    run_name = "geometry_spatial_from_state_teacher"
    student = _geometry_spatial_camera_actor(init_std=0.03)
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:GeometryDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=1.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometrySpatialTeacherRolloutRunnerCfg(SO101GeometrySpatialDistillationRunnerCfg):
    """Teacher-rollout warm start with the same auxiliary geometry target."""

    experiment_name = "so101_vial_camera_distillation_geometry_spatial_teacher_rollout"
    run_name = "geometry_spatial_teacher_rollout"
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:GeometryTeacherRolloutDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=1.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101StrongGeometrySpatialDistillationRunnerCfg(SO101GeometrySpatialDistillationRunnerCfg):
    """Geometry distillation with a task-scale localization contribution."""

    experiment_name = "so101_vial_camera_distillation_geometry_spatial_strong"
    run_name = "strong_geometry_spatial_from_state_teacher"
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:GeometryDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101StrongGeometrySpatialDenseDistillationRunnerCfg(
    SO101StrongGeometrySpatialDistillationRunnerCfg
):
    """Strong geometry DAgger with dense checkpoints for exact selection."""

    save_interval = 10
    experiment_name = "so101_vial_camera_distillation_geometry_spatial_strong_dense"
    run_name = "strong_geometry_spatial_dense"


@configclass
class SO101FrozenGeometrySpatialDenseDistillationRunnerCfg(
    SO101StrongGeometrySpatialDenseDistillationRunnerCfg
):
    """Dense DAgger with the pretrained spatial-keypoint encoder fixed."""

    experiment_name = "so101_vial_camera_distillation_geometry_spatial_frozen"
    run_name = "frozen_geometry_spatial_dense"
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FrozenEncoderGeometryDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101LowRateGeometrySpatialDistillationRunnerCfg(
    SO101StrongGeometrySpatialDistillationRunnerCfg
):
    """Short low-rate continuation around an exact-evaluated policy peak."""

    save_interval = 1
    experiment_name = "so101_vial_camera_distillation_geometry_spatial_low_rate"
    run_name = "geometry_spatial_low_rate"
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FineTuneGeometryDistillation",
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101UltraLowRateGeometrySpatialDistillationRunnerCfg(
    SO101LowRateGeometrySpatialDistillationRunnerCfg
):
    """Very short conservative continuation from an exact-evaluated peak."""

    experiment_name = "so101_vial_camera_distillation_geometry_spatial_ultra_low_rate"
    run_name = "geometry_spatial_ultra_low_rate"
    algorithm = RslRlGeometryDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FineTuneGeometryDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-5,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101WideTemporalCameraDistillationRunnerCfg(SO101WideCameraDistillationRunnerCfg):
    """Student-visited distillation with two consecutive 64x64 RGB frames."""

    experiment_name = "so101_vial_camera_distillation_temporal"
    run_name = "wide_two_frame_wrist_from_state_teacher"


@configclass
class SO101WideTemporalTeacherRolloutDistillationRunnerCfg(SO101WideTemporalCameraDistillationRunnerCfg):
    """Teacher-controlled warm start for the two-frame vision student."""

    experiment_name = "so101_vial_camera_distillation_temporal_teacher_rollout"
    run_name = "wide_two_frame_teacher_rollout"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:TeacherRolloutDistillation",
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101WideCameraFineDistillationRunnerCfg(SO101WideCameraDistillationRunnerCfg):
    """Low-rate DAgger refinement for checkpoint-stable canonical behavior."""

    save_interval = 10
    experiment_name = "so101_vial_camera_distillation_fine"
    run_name = "wide_low_rate_dagger"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FineTuneDistillation",
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101WideCameraFineTeacherRolloutRunnerCfg(SO101WideCameraFineDistillationRunnerCfg):
    """Low-rate canonical teacher rollouts from a capable DAgger student."""

    experiment_name = "so101_vial_camera_distillation_fine_teacher_rollout"
    run_name = "wide_low_rate_teacher_rollout"
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="so101_vial_place.agents.distillation:FineTuneTeacherRolloutDistillation",
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
