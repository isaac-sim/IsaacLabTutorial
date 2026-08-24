"""SO-101 vial-placement task registrations."""

import os

# Isaac Lab's source sub-wheels expect its monorepo-level Kit file to provide this
# shared default. The task itself uses only local assets, but importing core spawners
# still initializes the constant. Respect an operator-provided mirror when present.
os.environ.setdefault(
    "ISAACSIM_ASSET_ROOT",
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1",
)

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.env_cfg:SO101VialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Canonical",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.env_cfg:SO101VialCanonicalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "rsl_rl_scratch_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraScratchPPORunnerCfg"),
        "rsl_rl_wide_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraPPORunnerCfg",
        "rsl_rl_wide_exploration_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraExplorationPPORunnerCfg"
        ),
        "rsl_rl_wide_scratch_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraScratchPPORunnerCfg"),
        "rsl_rl_spatial_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101SpatialCameraPPORunnerCfg"),
        "rsl_rl_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101SpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_bottleneck_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg"
        ),
        "rsl_rl_wide_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraDistillationRunnerCfg"
        ),
        "rsl_rl_wide_fine_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraFineDistillationRunnerCfg"
        ),
        "rsl_rl_wide_fine_teacher_rollout_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraFineTeacherRolloutRunnerCfg"
        ),
        "rsl_rl_teacher_rollout_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideTeacherRolloutDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialCameraDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_teacher_rollout_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialTeacherRolloutDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_fine_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialFineDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_peak_search_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialPeakSearchDistillationRunnerCfg"
        ),
        "rsl_rl_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101GeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_geometry_spatial_teacher_rollout_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101GeometrySpatialTeacherRolloutRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_frozen_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101FrozenGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_low_rate_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101LowRateGeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_ultra_low_rate_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101UltraLowRateGeometrySpatialDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Canonical",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraCanonicalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "rsl_rl_wide_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraPPORunnerCfg",
        "rsl_rl_spatial_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101SpatialCameraPPORunnerCfg"),
        "rsl_rl_wide_fine_ppo_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraFinePPORunnerCfg"),
        "rsl_rl_wide_shared_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraSharedScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_exploration_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ExplorationGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101FineGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_bottleneck_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_augmented_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_bottleneck_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckFineScratchPPORunnerCfg"
        ),
        "rsl_rl_wide_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraDistillationRunnerCfg"
        ),
        "rsl_rl_wide_fine_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraFineDistillationRunnerCfg"
        ),
        "rsl_rl_wide_fine_teacher_rollout_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideCameraFineTeacherRolloutRunnerCfg"
        ),
        "rsl_rl_teacher_rollout_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideTeacherRolloutDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialCameraDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_teacher_rollout_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialTeacherRolloutDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_fine_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialFineDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_peak_search_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialPeakSearchDistillationRunnerCfg"
        ),
        "rsl_rl_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101GeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_frozen_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101FrozenGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Canonical-Long",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraLongCanonicalEnvCfg",
        "rsl_rl_geometry_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101FineGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_augmented_low_noise_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Acquisition",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraAcquisitionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "rsl_rl_wide_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraPPORunnerCfg",
        "rsl_rl_wide_exploration_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraExplorationPPORunnerCfg"
        ),
        "rsl_rl_wide_scratch_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraScratchPPORunnerCfg"),
        "rsl_rl_wide_shared_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101WideCameraSharedScratchPPORunnerCfg"
        ),
        "rsl_rl_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101SpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialCameraDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_fine_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialFineDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_spatial_peak_search_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101SpatialPeakSearchDistillationRunnerCfg"
        ),
        "rsl_rl_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101GeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_strong_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101StrongGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_frozen_geometry_spatial_dense_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101FrozenGeometrySpatialDenseDistillationRunnerCfg"
        ),
        "rsl_rl_low_rate_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101LowRateGeometrySpatialDistillationRunnerCfg"
        ),
        "rsl_rl_ultra_low_rate_geometry_spatial_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101UltraLowRateGeometrySpatialDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraScratchReachEnvCfg",
        "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

for canonical_percent, env_cfg_name in (
    (75, "SO101VialCameraScratchReach75EnvCfg"),
    (90, "SO101VialCameraScratchReach90EnvCfg"),
):
    gym.register(
        id=f"IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Reach{canonical_percent}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"so101_vial_place.camera_env_cfg:{env_cfg_name}",
            "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_pretrained_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryPretrainedCameraScratchPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )

for canonical_percent, env_cfg_name in (
    (75, "SO101VialCameraScratchGated75EnvCfg"),
    (90, "SO101VialCameraScratchGated90EnvCfg"),
):
    gym.register(
        id=f"IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Gated{canonical_percent}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"so101_vial_place.camera_env_cfg:{env_cfg_name}",
            "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_pretrained_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryPretrainedCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_exploration_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ExplorationGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_bottleneck_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckCameraScratchPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Scratch-GatedAcquisition",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ("so101_vial_place.camera_env_cfg:SO101VialCameraScratchGatedAcquisitionEnvCfg"),
        "rsl_rl_geometry_bottleneck_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_exploration_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ExplorationGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101FineGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_spatial_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometrySpatialCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_augmented_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_bottleneck_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckFineScratchPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Canonical",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraScratchCanonicalEnvCfg",
        "rsl_rl_geometry_bottleneck_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryBottleneckCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_exploration_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ExplorationGeometryCameraScratchPPORunnerCfg"
        ),
        "rsl_rl_geometry_fine_scratch_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SO101FineGeometryCameraScratchPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)

for task_suffix, env_cfg_name in (
    ("HeldGoal", "SO101VialCameraScratchHeldGoalEnvCfg"),
    ("HeldGoal-NoMargin", "SO101VialCameraScratchHeldGoalNoMarginEnvCfg"),
    ("Clearance", "SO101VialCameraScratchClearanceEnvCfg"),
    ("HeldGoal-Clearance", "SO101VialCameraScratchHeldGoalClearanceEnvCfg"),
    ("StrongHeldGoal-Clearance", "SO101VialCameraScratchStrongHeldGoalClearanceEnvCfg"),
    (
        "StrongHeldGoal-Clearance-Horizon",
        "SO101VialCameraScratchStrongHeldGoalClearanceHorizonEnvCfg",
    ),
    ("GoalProgress", "SO101VialCameraScratchGoalProgressEnvCfg"),
    ("GoalProgress-Canonical", "SO101VialCameraScratchGoalProgressCanonicalEnvCfg"),
    ("StrongGoalProgress-Canonical", "SO101VialCameraScratchStrongGoalProgressCanonicalEnvCfg"),
    ("NarrowGoalProgress-Canonical", "SO101VialCameraScratchNarrowGoalProgressCanonicalEnvCfg"),
    ("GoalError-Canonical", "SO101VialCameraScratchGoalErrorCanonicalEnvCfg"),
    ("WeakGoalError-Canonical", "SO101VialCameraScratchWeakGoalErrorCanonicalEnvCfg"),
    ("StrongGoalError-Canonical", "SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg"),
    ("GoalError90", "SO101VialCameraScratchGoalError90EnvCfg"),
    ("GoalError75", "SO101VialCameraScratchGoalError75EnvCfg"),
    ("GoalError-Horizon", "SO101VialCameraScratchGoalErrorHorizonEnvCfg"),
    ("StrongGoalError-Horizon", "SO101VialCameraScratchStrongGoalErrorHorizonEnvCfg"),
    ("StrongGoalError90", "SO101VialCameraScratchStrongGoalError90EnvCfg"),
    ("StrongGoalError75", "SO101VialCameraScratchStrongGoalError75EnvCfg"),
    ("StrongGoalError-Prefix5", "SO101VialCameraScratchStrongGoalErrorPrefix5EnvCfg"),
    ("StrongGoalError-Prefix6", "SO101VialCameraScratchStrongGoalErrorPrefix6EnvCfg"),
    ("StrongGoalError-Prefix7", "SO101VialCameraScratchStrongGoalErrorPrefix7EnvCfg"),
    ("GoalBasin-Canonical", "SO101VialCameraScratchGoalBasinCanonicalEnvCfg"),
    ("WideGoalBasin-Canonical", "SO101VialCameraScratchWideGoalBasinCanonicalEnvCfg"),
    ("StrongGoalBasin-Canonical", "SO101VialCameraScratchStrongGoalBasinCanonicalEnvCfg"),
    ("GoalBasin-Prefix6", "SO101VialCameraScratchGoalBasinPrefix6EnvCfg"),
    ("GoalBasin-Prefix7", "SO101VialCameraScratchGoalBasinPrefix7EnvCfg"),
    ("GoalBasin-Horizon", "SO101VialCameraScratchGoalBasinHorizonEnvCfg"),
    ("StrongGoalBasin-Horizon", "SO101VialCameraScratchStrongGoalBasinHorizonEnvCfg"),
    ("UprightGoal-Canonical", "SO101VialCameraScratchUprightGoalCanonicalEnvCfg"),
    ("WeakUprightGoal-Canonical", "SO101VialCameraScratchWeakUprightGoalCanonicalEnvCfg"),
    ("StrongUprightGoal-Canonical", "SO101VialCameraScratchStrongUprightGoalCanonicalEnvCfg"),
    ("EarlyGoal-Canonical", "SO101VialCameraScratchEarlyGoalCanonicalEnvCfg"),
    ("EarlyGoal-Horizon", "SO101VialCameraScratchEarlyGoalHorizonEnvCfg"),
    ("EarlyGoal-HorizonRelease", "SO101VialCameraScratchEarlyGoalHorizonReleaseEnvCfg"),
    ("EarlyGoal-CanonicalRelease", "SO101VialCameraScratchEarlyGoalCanonicalReleaseEnvCfg"),
    ("EarlyGoal-Release", "SO101VialCameraScratchEarlyGoalReleaseEnvCfg"),
    ("EarlyGoal-InsertionRelease", "SO101VialCameraScratchEarlyGoalInsertionReleaseEnvCfg"),
    (
        "EarlyGoal-CanonicalInsertionRelease",
        "SO101VialCameraScratchEarlyGoalCanonicalInsertionReleaseEnvCfg",
    ),
    ("EarlyGoal-ReleaseShaping", "SO101VialCameraScratchEarlyGoalReleaseShapingEnvCfg"),
    (
        "EarlyGoal-CanonicalReleaseShaping",
        "SO101VialCameraScratchEarlyGoalCanonicalReleaseShapingEnvCfg",
    ),
    ("EarlyGoal-ReleaseProgress", "SO101VialCameraScratchEarlyGoalReleaseProgressEnvCfg"),
    ("EarlyGoal-InsertionTransition", "SO101VialCameraScratchEarlyGoalInsertionTransitionEnvCfg"),
    ("ClosedInsertion", "SO101VialCameraScratchClosedInsertionEnvCfg"),
    ("CanonicalClosedInsertion", "SO101VialCameraScratchCanonicalClosedInsertionEnvCfg"),
    ("RetainClosedInsertion", "SO101VialCameraScratchRetainClosedInsertionEnvCfg"),
    (
        "EarlyGoal-CanonicalReleaseProgress",
        "SO101VialCameraScratchEarlyGoalCanonicalReleaseProgressEnvCfg",
    ),
    ("EarlyGoal-Radial", "SO101VialCameraScratchEarlyGoalRadialEnvCfg"),
    ("EarlyGoal-StrongRadial", "SO101VialCameraScratchEarlyGoalStrongRadialEnvCfg"),
    ("Unified", "SO101VialCameraScratchUnifiedEnvCfg"),
    ("Unified-Horizon", "SO101VialCameraScratchUnifiedHorizonEnvCfg"),
    ("Discovery", "SO101VialCameraScratchDiscoveryEnvCfg"),
    ("Discovery-Release", "SO101VialCameraScratchDiscoveryReleaseEnvCfg"),
    ("Discovery-ReleaseOnly", "SO101VialCameraScratchDiscoveryReleaseOnlyEnvCfg"),
    ("Discovery-StrongReleaseOnly", "SO101VialCameraScratchDiscoveryStrongReleaseOnlyEnvCfg"),
    ("Discovery-StableReleaseOnly", "SO101VialCameraScratchDiscoveryStableReleaseOnlyEnvCfg"),
    ("Discovery-AnnealedReleaseOnly", "SO101VialCameraScratchDiscoveryAnnealedReleaseOnlyEnvCfg"),
    ("Discovery-BoostedReleaseOnly", "SO101VialCameraScratchDiscoveryBoostedReleaseOnlyEnvCfg"),
    ("Discovery-AlignedReleaseOnly", "SO101VialCameraScratchDiscoveryAlignedReleaseOnlyEnvCfg"),
    ("Discovery-FinalReleaseOnly", "SO101VialCameraScratchDiscoveryFinalReleaseOnlyEnvCfg"),
    ("Discovery-SuccessReleaseOnly", "SO101VialCameraScratchDiscoverySuccessReleaseOnlyEnvCfg"),
    (
        "Discovery-StableBoostedReleaseOnly",
        "SO101VialCameraScratchDiscoveryStableBoostedReleaseOnlyEnvCfg",
    ),
    ("Discovery-Canonical", "SO101VialCameraScratchDiscoveryCanonicalEnvCfg"),
    ("Discovery-CanonicalClose", "SO101VialCameraScratchDiscoveryCanonicalCloseEnvCfg"),
    ("Discovery-CanonicalLift", "SO101VialCameraScratchDiscoveryCanonicalLiftEnvCfg"),
    ("Discovery-CanonicalCloseLift", "SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg"),
    ("Discovery-CloseLift", "SO101VialCameraScratchDiscoveryCloseLiftEnvCfg"),
    ("Discovery-CanonicalTransport", "SO101VialCameraScratchDiscoveryCanonicalTransportEnvCfg"),
    ("Discovery-Transport", "SO101VialCameraScratchDiscoveryTransportEnvCfg"),
    ("Discovery-CanonicalStrongLift", "SO101VialCameraScratchDiscoveryCanonicalStrongLiftEnvCfg"),
    ("Discovery-CanonicalLiftProgress", "SO101VialCameraScratchDiscoveryCanonicalLiftProgressEnvCfg"),
    (
        "Discovery-CanonicalProgressTransport",
        "SO101VialCameraScratchDiscoveryCanonicalProgressTransportEnvCfg",
    ),
    ("Discovery-ProgressTransport", "SO101VialCameraScratchDiscoveryProgressTransportEnvCfg"),
    ("Discovery-CanonicalUpright", "SO101VialCameraScratchDiscoveryCanonicalUprightEnvCfg"),
    ("Discovery-CanonicalStrongUpright", "SO101VialCameraScratchDiscoveryCanonicalStrongUprightEnvCfg"),
    ("Discovery-UprightPair", "SO101VialCameraScratchDiscoveryUprightPairEnvCfg"),
    ("Discovery-CanonicalRadial", "SO101VialCameraScratchDiscoveryCanonicalRadialEnvCfg"),
    ("Discovery-RadialPair", "SO101VialCameraScratchDiscoveryRadialPairEnvCfg"),
    ("Discovery-StrongRadialPair", "SO101VialCameraScratchDiscoveryStrongRadialPairEnvCfg"),
    ("Discovery-TransportOnly", "SO101VialCameraScratchDiscoveryTransportOnlyEnvCfg"),
    ("Discovery-TransportGoal", "SO101VialCameraScratchDiscoveryTransportGoalEnvCfg"),
    ("Discovery-TransportAlignment", "SO101VialCameraScratchDiscoveryTransportAlignmentEnvCfg"),
    (
        "Discovery-TransportAlignmentHold",
        "SO101VialCameraScratchDiscoveryTransportAlignmentHoldEnvCfg",
    ),
    ("Discovery-TransportStable", "SO101VialCameraScratchDiscoveryTransportStableEnvCfg"),
    (
        "Discovery-TransportStrongStable",
        "SO101VialCameraScratchDiscoveryTransportStrongStableEnvCfg",
    ),
    ("Discovery-TransportVeryStable", "SO101VialCameraScratchDiscoveryTransportVeryStableEnvCfg"),
    ("Discovery-TransportInsertion", "SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg"),
    (
        "Discovery-TransportStrongInsertion",
        "SO101VialCameraScratchDiscoveryTransportStrongInsertionEnvCfg",
    ),
    (
        "Discovery-TransportStableInsertion",
        "SO101VialCameraScratchDiscoveryTransportStableInsertionEnvCfg",
    ),
    (
        "Discovery-TransportUnifiedGoal",
        "SO101VialCameraScratchDiscoveryTransportUnifiedGoalEnvCfg",
    ),
    (
        "Discovery-TransportFullGoal",
        "SO101VialCameraScratchDiscoveryTransportFullGoalEnvCfg",
    ),
    (
        "Discovery-TransportUpright",
        "SO101VialCameraScratchDiscoveryTransportUprightEnvCfg",
    ),
    (
        "Discovery-TransportUprightProgress",
        "SO101VialCameraScratchDiscoveryTransportUprightProgressEnvCfg",
    ),
    (
        "Discovery-TransportStrongUprightProgress",
        "SO101VialCameraScratchDiscoveryTransportStrongUprightProgressEnvCfg",
    ),
    ("Discovery-TransportCenter", "SO101VialCameraScratchDiscoveryTransportCenterEnvCfg"),
    (
        "Discovery-TransportStrongCenter",
        "SO101VialCameraScratchDiscoveryTransportStrongCenterEnvCfg",
    ),
    (
        "Discovery-TransportCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportCenterBasinEnvCfg",
    ),
    (
        "Discovery-TransportStrongCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportStrongCenterBasinEnvCfg",
    ),
    (
        "Discovery-TransportNarrowCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg",
    ),
    (
        "Discovery-TransportVeryStrongCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportVeryStrongCenterBasinEnvCfg",
    ),
    (
        "Discovery-TransportFreeCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportFreeCenterBasinEnvCfg",
    ),
    ("Discovery-TransportCenterGoal", "SO101VialCameraScratchDiscoveryTransportCenterGoalEnvCfg"),
    (
        "Discovery-TransportPreciseCenterBasin",
        "SO101VialCameraScratchDiscoveryTransportPreciseCenterBasinEnvCfg",
    ),
    (
        "Discovery-TransportInsertionMilestone",
        "SO101VialCameraScratchDiscoveryTransportInsertionMilestoneEnvCfg",
    ),
    (
        "Discovery-TransportStrongInsertionMilestone",
        "SO101VialCameraScratchDiscoveryTransportStrongInsertionMilestoneEnvCfg",
    ),
    (
        "Discovery-TransportInsertionEpisode",
        "SO101VialCameraScratchDiscoveryTransportInsertionEpisodeEnvCfg",
    ),
    (
        "Discovery-TransportStrongInsertionEpisode",
        "SO101VialCameraScratchDiscoveryTransportStrongInsertionEpisodeEnvCfg",
    ),
    (
        "Discovery-TransportInsertionCenter500",
        "SO101VialCameraScratchDiscoveryTransportInsertionCenter500EnvCfg",
    ),
    (
        "Discovery-TransportInsertionAlignment100",
        "SO101VialCameraScratchDiscoveryTransportInsertionAlignment100EnvCfg",
    ),
    (
        "Discovery-TransportInsertionBalanced",
        "SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg",
    ),
    (
        "Discovery-InsertionReleaseBalanced",
        "SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg",
    ),
    (
        "Discovery-InsertionReleaseBoosted",
        "SO101VialCameraScratchDiscoveryInsertionReleaseBoostedEnvCfg",
    ),
    (
        "Discovery-InsertionReleaseCompletion",
        "SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg",
    ),
    (
        "Discovery-ReadyReleaseCompletion",
        "SO101VialCameraScratchDiscoveryReadyReleaseCompletionEnvCfg",
    ),
    (
        "Discovery-TransportReleaseCompletion",
        "SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg",
    ),
    (
        "Discovery-TransportReleaseDirect",
        "SO101VialCameraScratchDiscoveryTransportReleaseDirectEnvCfg",
    ),
    (
        "Discovery-HorizonReleaseDirect",
        "SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg",
    ),
    (
        "Discovery-Acquisition90ReleaseDirect",
        "SO101VialCameraScratchDiscoveryAcquisition90ReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBalancedReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBalancedReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalStrongDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalStrongDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-TransportDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryTransportDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalTransportDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalTransportDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeStrongDenseReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeStrongDenseReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeRadialReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeRadialNoClearanceReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeRadialNoClearanceReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialUprightReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialUprightReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeStrongRadialReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeStrongRadialReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialVeryUprightReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialVeryUprightReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialUprightProgressReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialStrictUprightReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialStrictUprightReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialStrongUprightProgressReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialStrongUprightProgressReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeRadialModerateUprightReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeRadialModerateUprightReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightOnlyReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightStrongProgressReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightNormalizedReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightNormalizedReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightEfficientReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightEfficientReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightLocalSafeReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightLocalSafeReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightBalancedSafeReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightBalancedSafeReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightConjunctiveReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightConjunctiveOnlyReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightConjunctiveMediumReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveMediumReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightConjunctiveTightReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveTightReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightLiftReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightLiftReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeLiftCompensationReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeLiftCompensationReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeUprightLiftReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridge75UprightLiftReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridge75UprightLiftReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridgeStrongUprightLiftReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridgeStrongUprightLiftReleaseDirectEnvCfg",
    ),
    (
        "Discovery-CanonicalBridge75StrongUprightLiftReleaseDirect",
        "SO101VialCameraScratchDiscoveryCanonicalBridge75StrongUprightLiftReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightStrictReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightStrictReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightStrictStrongProgressReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightStrictStrongProgressReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightSafeReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightSafeReleaseDirectEnvCfg",
    ),
    (
        "Discovery-BridgeUprightStrictSafeReleaseDirect",
        "SO101VialCameraScratchDiscoveryBridgeUprightStrictSafeReleaseDirectEnvCfg",
    ),
    (
        "Discovery-TransportDenseAlignedReleaseDirect",
        "SO101VialCameraScratchDiscoveryTransportDenseAlignedReleaseDirectEnvCfg",
    ),
    (
        "Discovery-TransportReleaseOnly",
        "SO101VialCameraScratchDiscoveryTransportReleaseOnlyEnvCfg",
    ),
    (
        "Discovery-TransportInsertionGeometry10cm",
        "SO101VialCameraScratchDiscoveryTransportInsertionGeometry10cmEnvCfg",
    ),
    (
        "Discovery-TransportInsertionHighQuality",
        "SO101VialCameraScratchDiscoveryTransportInsertionHighQualityEnvCfg",
    ),
    (
        "Discovery-TransportInsertionGoalRefine",
        "SO101VialCameraScratchDiscoveryTransportInsertionGoalRefineEnvCfg",
    ),
    (
        "Discovery-TransportInsertionStrongGoalRefine",
        "SO101VialCameraScratchDiscoveryTransportInsertionStrongGoalRefineEnvCfg",
    ),
    (
        "Discovery-TransportInsertionDepthRefine",
        "SO101VialCameraScratchDiscoveryTransportInsertionDepthRefineEnvCfg",
    ),
    (
        "Discovery-TransportInsertionGateRefine",
        "SO101VialCameraScratchDiscoveryTransportInsertionGateRefineEnvCfg",
    ),
    (
        "Discovery-TransportInsertion86",
        "SO101VialCameraScratchDiscoveryTransportInsertion86EnvCfg",
    ),
    (
        "Discovery-TransportInsertionCenter1000",
        "SO101VialCameraScratchDiscoveryTransportInsertionCenter1000EnvCfg",
    ),
    (
        "Discovery-TransportInsertionRadialCost",
        "SO101VialCameraScratchDiscoveryTransportInsertionRadialCostEnvCfg",
    ),
    (
        "Discovery-TransportInsertionNominal",
        "SO101VialCameraScratchDiscoveryTransportInsertionNominalEnvCfg",
    ),
    ("Discovery-Lift", "SO101VialCameraScratchDiscoveryLiftEnvCfg"),
    ("Discovery-WeakLift", "SO101VialCameraScratchDiscoveryWeakLiftEnvCfg"),
    (
        "EarlyGoal-StrongReleaseShaping",
        "SO101VialCameraScratchEarlyGoalStrongReleaseShapingEnvCfg",
    ),
    ("EarlyGoal90", "SO101VialCameraScratchEarlyGoal90EnvCfg"),
    ("EarlyGoal75", "SO101VialCameraScratchEarlyGoal75EnvCfg"),
):
    gym.register(
        id=f"IsaacTutorial-Place-Vial-SO101-Camera-Scratch-{task_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"so101_vial_place.camera_env_cfg:{env_cfg_name}",
            "rsl_rl_geometry_exploration_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ExplorationGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_moderate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedModerateSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_stable_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedStableSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_stable_finetune_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedStableFineTuneScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_finetune_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFineTuneScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_slow_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonSlowRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_finetune_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_resume_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenResumeScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_fast_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenFastScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_aggressive_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenAggressiveScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_output_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_output_fast_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenOutputFastScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_output_midrate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenOutputMidrateScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_frozen_output_resume_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFrozenOutputResumeScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_proximal_compensation_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonProximalCompensationScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedRefineSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_slow_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedSlowResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_midrate_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedMidrateResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_fast_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFastResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_search_residual_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonSearchResidualScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_post_lift_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonPostLiftScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_fast_post_lift_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonFastPostLiftScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_search_post_lift_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonSearchPostLiftScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_aggressive_post_lift_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonAggressivePostLiftScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_long_horizon_aggressive_post_lift_resume_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedLongHorizonAggressivePostLiftResumeScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_ultra_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedUltraRefineSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_midrate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryMidrateScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_rate5_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedRate5ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_search_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedSearchScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_stable_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedStableScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_stable_rate10_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedStableRate10ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_long_rollout_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedLongRolloutScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_augmented_long_rollout_rate30_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryAugmentedLongRolloutRate30ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_ultra_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckUltraRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_output_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckOutputRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_output_rate30_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckOutputRate30ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_controller_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckControllerScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_encoder_refine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckEncoderRefineScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_encoder_rate30_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckEncoderRate30ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_gripper_release_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckGripperReleaseScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_gripper_release_rate30_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckGripperReleaseRate30ScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_geometry_bottleneck_split_gripper_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedGeometryBottleneckSplitGripperScratchPPORunnerCfg"
            ),
            "rsl_rl_vision_tuned_kuka_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101VisionTunedKukaSpatialScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_resume_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ResumeGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_midrate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101MidRateGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_fine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101FineGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_low_noise_fine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LowNoiseFineGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_moderate_noise_fine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ModerateNoiseFineGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_low_noise_midrate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LowNoiseMidRateGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_moderate_noise_midrate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101ModerateNoiseMidRateGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_high_noise_resume_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101HighNoiseResumeGeometryCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_low_noise_fine_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedCameraScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_high_noise_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedHighNoiseScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_moderate_noise_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedModerateNoiseScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_gripper_explore_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedGripperExploreScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_precise_moderate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedPreciseModerateScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_precise_low_noise_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedPreciseLowNoiseScratchPPORunnerCfg"
            ),
            "rsl_rl_geometry_augmented_target_precise_moderate_scratch_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:SO101GeometryAugmentedTargetPreciseModerateScratchPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera-Temporal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialTemporalCameraEnvCfg",
        "rsl_rl_wide_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideTemporalCameraDistillationRunnerCfg"
        ),
        "rsl_rl_teacher_rollout_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101WideTemporalTeacherRolloutDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
