"""Configuration contract tests for both public task variants."""

import inspect

import pytest
import torch
from rsl_rl.algorithms import Distillation

from so101_vial_place import evaluation
from so101_vial_place.agents.distillation import FineTuneDistillation
from so101_vial_place.agents.ppo import (
    FineTunePPO,
    FrozenStatsFineTunePPO,
    FrozenStatsResumePPO,
    OutputLayerFrozenStatsFineTunePPO,
    OutputLayerFrozenStatsResumePPO,
    ProximalOutputCompensationPPO,
    ResidualResumePPO,
)
from so101_vial_place.agents.rsl_rl_distillation_cfg import (
    SO101CameraDistillationRunnerCfg,
    SO101GeometrySpatialDistillationRunnerCfg,
    SO101GeometrySpatialTeacherRolloutRunnerCfg,
    SO101SpatialCameraDistillationRunnerCfg,
    SO101SpatialDenseDistillationRunnerCfg,
    SO101SpatialFineDistillationRunnerCfg,
    SO101SpatialTeacherRolloutDistillationRunnerCfg,
    SO101WideCameraDistillationRunnerCfg,
    SO101WideCameraFineDistillationRunnerCfg,
    SO101WideCameraFineTeacherRolloutRunnerCfg,
    SO101WideTeacherRolloutDistillationRunnerCfg,
    SO101WideTemporalCameraDistillationRunnerCfg,
    SO101WideTemporalTeacherRolloutDistillationRunnerCfg,
)
from so101_vial_place.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101CameraScratchPPORunnerCfg,
    SO101SpatialCameraPPORunnerCfg,
    SO101SpatialCameraScratchPPORunnerCfg,
    SO101StatePPORunnerCfg,
    SO101VisionTunedLongHorizonAggressivePostLiftResumeScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonAggressivePostLiftScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenFastScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenOutputFastScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenOutputMidrateScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenOutputResumeScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonFrozenResumeScratchPPORunnerCfg,
    SO101VisionTunedLongHorizonProximalCompensationScratchPPORunnerCfg,
    SO101WideCameraExplorationPPORunnerCfg,
    SO101WideCameraFinePPORunnerCfg,
    SO101WideCameraPPORunnerCfg,
    SO101WideCameraSharedScratchPPORunnerCfg,
)
from so101_vial_place.camera_env_cfg import (
    SO101VialCameraAcquisitionEnvCfg,
    SO101VialCameraCanonicalEnvCfg,
    SO101VialCameraEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeLiftCompensationReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialModerateUprightReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialStrictUprightReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialStrongUprightProgressReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialUprightReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeRadialVeryUprightReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeStrongRadialReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightBalancedSafeReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveMediumReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveTightReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightEfficientReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightLiftReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightLocalSafeReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightNormalizedReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightSafeReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightStrictReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightStrictSafeReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightStrictStrongProgressReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridge75StrongUprightLiftReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridge75UprightLiftReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridgeRadialNoClearanceReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridgeStrongUprightLiftReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg,
    SO101VialCameraScratchDiscoveryInsertionReleaseBoostedEnvCfg,
    SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg,
    SO101VialCameraScratchDiscoveryReadyReleaseCompletionEnvCfg,
    SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg,
    SO101VialCameraScratchDiscoveryTransportReleaseDirectEnvCfg,
    SO101VialCameraScratchDiscoveryTransportReleaseOnlyEnvCfg,
    SO101VialCameraScratchHeldGoalClearanceEnvCfg,
    SO101VialCameraScratchHeldGoalEnvCfg,
    SO101VialCameraScratchHeldGoalNoMarginEnvCfg,
    SO101VialTemporalCameraEnvCfg,
)
from so101_vial_place.control import (
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_VIAL_HEADING_RANGE,
    TABLETOP_VIAL_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from so101_vial_place.env_cfg import (
    ARM_JOINTS,
    JOINTS,
    InitialEventsCfg,
    ResetJointActionsCfg,
    SO101VialCanonicalEnvCfg,
    SO101VialEnvCfg,
    SO101VialGeneratorEnvCfg,
)
from so101_vial_place.mdp.actions import SoftLimitRelativeGripperAction, SoftLimitRelativeJointPositionAction
from so101_vial_place.reset.curriculum import (
    RESET_CURRICULA,
    reset_curriculum_maximum_difficulty,
    reset_curriculum_minimum_difficulty,
    reset_curriculum_weights,
)


def test_state_task_control_and_action_contract():
    cfg = SO101VialEnvCfg()

    assert cfg.scene.num_envs == 4096
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.episode_length_s == 20.0
    physics = cfg.sim.physics.newton_mjwarp
    # Match Isaac Lab's Newton manipulation defaults while retaining the
    # elliptic friction cone needed for a stable two-pad grasp.
    assert physics.num_substeps == 2
    assert physics.solver_cfg.use_mujoco_contacts is False
    assert physics.solver_cfg.solver == "newton"
    assert physics.solver_cfg.njmax == 300
    assert physics.solver_cfg.nconmax == 200
    assert physics.solver_cfg.iterations == 100
    assert physics.solver_cfg.ls_iterations == 15
    assert physics.solver_cfg.impratio == pytest.approx(10.0)
    assert physics.solver_cfg.update_data_interval == 2
    assert physics.solver_cfg.integrator == "implicitfast"
    assert physics.collision_cfg.__class__.__name__ == "NewtonCollisionPipelineCfg"

    assert physics.collision_cfg.broad_phase == "explicit"
    assert physics.collision_cfg.rigid_contact_max is None
    # Primitive rack/vial contacts need no mesh-heavy collision allocation.
    assert physics.collision_cfg.max_triangle_pairs == 1_000_000
    assert cfg.actions.arm_action.__class__.__name__ == "RelativeJointPositionActionCfg"
    assert cfg.actions.arm_action.joint_names == ARM_JOINTS
    assert cfg.actions.arm_action.scale == pytest.approx(0.033)
    assert cfg.actions.arm_action.use_zero_offset is True
    assert cfg.actions.gripper_action.joint_names == ["gripper"]
    assert cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is True
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.scene.robot.init_state.joint_pos["gripper"] == pytest.approx(PREGRASP_GRIPPER_POSITION)
    assert cfg.scene.vial.init_state.pos == pytest.approx(TABLETOP_VIAL_POSITION)
    assert tuple(cfg.scene.robot.init_state.joint_pos.values()) == pytest.approx(WORKSHOP_INITIAL_JOINT_POSITION)
    assert cfg.sim.default_visualizer_cfg.eye == pytest.approx((0.64, 0.0, 0.36))
    assert cfg.sim.default_visualizer_cfg.lookat == pytest.approx((0.19, 0.02, 0.075))
    assert len(JOINTS) == 6
    assert cfg.actions.gripper_action.scale == pytest.approx(0.02)
    assert cfg.rewards.object_goal.func.__name__ == "object_goal_reward"
    assert cfg.rewards.object_goal.weight == pytest.approx(0.1)
    reset_action = ResetJointActionsCfg().joint_delta
    assert reset_action.joint_names == JOINTS
    assert reset_action.scale["gripper"] == pytest.approx(1.0)
    assert reset_action.scale["shoulder_lift|elbow_flex"] == pytest.approx(0.04)
    assert reset_action.scale["shoulder_pan|wrist_.*"] == pytest.approx(0.03)
    assert "vial_rack_contact" in cfg.scene.__dict__
    assert set(cfg.events.__dict__) >= {
        "vial_material",
        "vial_mass",
        "reset_from_dataset",
    }
    assert "fingertip_material" not in cfg.events.__dict__
    assert "servo_gains" not in cfg.events.__dict__
    usd_sysid = cfg.scene.robot.actuators["usd_sysid"]
    for field in (
        "stiffness",
        "damping",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
        "effort_limit",
        "velocity_limit",
        "effort_limit_sim",
        "velocity_limit_sim",
    ):
        assert getattr(usd_sysid, field) is None


def test_balanced_visual_release_stage_keeps_insertion_contract():
    cfg = SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg()

    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["insertion"]
    assert cfg.events.reset_from_dataset.params["minimum_difficulty"] is None
    assert cfg.terminations.success.func.__name__ == "PlacementHistoryTerm"
    assert cfg.rewards.held_radial_center.weight == pytest.approx(500.0)
    assert cfg.rewards.held_upright_alignment.weight == pytest.approx(100.0)
    assert cfg.rewards.release_opening.weight == pytest.approx(100.0)
    assert cfg.rewards.release_opening_progress.weight == pytest.approx(50.0)
    assert cfg.rewards.success.weight == pytest.approx(2000.0)
    boosted = SO101VialCameraScratchDiscoveryInsertionReleaseBoostedEnvCfg()
    assert boosted.rewards.release_opening.weight == pytest.approx(1000.0)
    completion = SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg()
    assert completion.rewards.release_opening.weight == pytest.approx(0.0)
    assert completion.rewards.release_opening_progress.weight == pytest.approx(1000.0)
    assert completion.rewards.success.weight == pytest.approx(10000.0)
    ready = SO101VialCameraScratchDiscoveryReadyReleaseCompletionEnvCfg()
    assert ready.events.reset_from_dataset.params["minimum_difficulty"] == ((6, 1.0),)
    transport = SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg()
    assert transport.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["transport"]
    assert transport.terminations.success.func.__name__ == "PlacementHistoryTerm"
    assert transport.episode_length_s == pytest.approx(30.0)
    direct = SO101VialCameraScratchDiscoveryTransportReleaseDirectEnvCfg()
    assert direct.rewards.release_action.weight == pytest.approx(1000.0)
    horizon_direct = SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg()
    assert horizon_direct.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["horizon"]
    release_only = SO101VialCameraScratchDiscoveryTransportReleaseOnlyEnvCfg()
    nonzero = {
        name: term.weight
        for name, term in release_only.rewards.__dict__.items()
        if hasattr(term, "weight") and term.weight != 0.0
    }
    assert nonzero == {
        "success": 10000.0,
        "release_opening_progress": 1000.0,
        "vial_lost": -1000.0,
    }


def test_play_mode_is_fixed_to_canonical_initial_resets(monkeypatch):
    monkeypatch.setattr(evaluation, "PLAY_RESETS_SEQUENTIAL", True)
    cfg = SO101VialEnvCfg()

    cfg.play_mode()

    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["initial"]
    assert cfg.events.reset_from_dataset.params["minimum_difficulty"] is None


def test_video_play_can_randomize_across_canonical_rows(monkeypatch):
    monkeypatch.setattr(evaluation, "PLAY_RESETS_SEQUENTIAL", False)
    cfg = SO101VialEnvCfg()

    cfg.play_mode()

    assert cfg.events.reset_from_dataset.params["sequential"] is False
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["initial"]


def test_state_canonical_refinement_has_an_explicit_home_training_contract():
    cfg = SO101VialCanonicalEnvCfg()

    assert cfg.events.reset_from_dataset.params["sequential"] is False
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["initial"]
    assert cfg.events.reset_from_dataset.params["minimum_difficulty"] is None
    assert cfg.events.reset_from_dataset.params["maximum_difficulty"] is None


def test_tabletop_episodes_reset_every_dynamic_scene_object():
    events = InitialEventsCfg()

    assert events.reset_vial.params["asset_cfg"].name == "vial"
    assert events.reset_vial.params["pose_range"]["yaw"] == TABLETOP_VIAL_HEADING_RANGE
    assert events.reset_rack.params["asset_cfg"].name == "rack"
    assert set(events.reset_rack.params["pose_range"]) == {"x", "y", "z", "roll", "pitch", "yaw"}
    assert all(bounds == (0.0, 0.0) for bounds in events.reset_rack.params["pose_range"].values())


def test_exact_evaluation_retains_requested_batch(monkeypatch):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    monkeypatch.setattr(evaluation, "PLAY_EVALUATION_EPISODES", 1024)
    state = SO101VialEnvCfg()
    camera = SO101VialCameraEnvCfg()

    state.play_mode()
    camera.play_mode()

    assert state.scene.num_envs == 1024
    assert camera.scene.num_envs == 1024


def test_camera_actor_is_unprivileged_and_has_only_wrist_image():
    cfg = SO101VialCameraEnvCfg()
    critic_terms = set(cfg.observations.critic.__dict__)

    assert cfg.scene.num_envs == 1024
    assert (cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height) == (64, 64)
    assert cfg.scene.wrist_camera.prim_path.endswith("/gripper/wrist_camera")
    assert cfg.scene.wrist_camera.offset.pos == pytest.approx((-0.055, 0.052, -0.035))
    assert cfg.scene.wrist_camera.offset.rot == pytest.approx((-0.09871531, 0.59943614, 0.78375556, -0.12906908))
    assert cfg.scene.wrist_camera.offset.convention == "opengl"
    calibration = cfg.scene.wrist_camera.spawn.distortion
    assert calibration.image_size == (64, 64)
    assert calibration.fx == pytest.approx(339.26593 / 10.0)
    assert calibration.fy == pytest.approx(338.8201 / 10.0)
    assert cfg.scene.wrist_camera.update_period == pytest.approx(1.0 / 30.0)
    assert cfg.scene.wrist_camera.update_latest_camera_pose is True
    assert set(cfg.observations.__dict__) >= {"wrist_rgb", "proprioception", "critic"}
    assert set(cfg.observations.proprioception.__dict__) >= {
        "joint_pos",
        "joint_vel",
        "joint_target",
        "previous_action",
    }
    assert not {"vial", "rack_target", "progress"} & set(cfg.observations.proprioception.__dict__)
    assert {"vial", "rack_target", "progress"} <= critic_terms
    assert cfg.observations.wrist_rgb.enable_corruption is True
    assert cfg.observations.proprioception.enable_corruption is True
    assert cfg.observations.wrist_rgb.image.func.__name__ == "DomainRandomizedCameraImage"
    assert cfg.observations.wrist_rgb.image.params["exposure_range"] == pytest.approx((0.75, 1.25))
    assert cfg.observations.wrist_rgb.image.params["white_balance_range"] == pytest.approx((0.90, 1.10))
    assert cfg.observations.wrist_rgb.image.noise is not None
    assert cfg.observations.proprioception.joint_pos.noise is not None
    camera_names = {name for name in cfg.scene.__dict__ if "camera" in name}
    assert camera_names == {"wrist_camera"}


def test_canonical_camera_refinement_has_an_explicit_phase_zero_training_contract():
    cfg = SO101VialCameraCanonicalEnvCfg()

    assert cfg.events.reset_from_dataset.params["sequential"] is False
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["initial"]
    assert cfg.events.reset_from_dataset.params["minimum_difficulty"] is None
    assert cfg.events.reset_from_dataset.params["maximum_difficulty"] is None


def test_camera_acquisition_ablation_is_one_fixed_half_canonical_distribution():
    cfg = SO101VialCameraAcquisitionEnvCfg()
    weights = cfg.events.reset_from_dataset.params["phase_weights"]

    assert weights == RESET_CURRICULA["acquisition"]
    assert weights[0] / sum(weights) == pytest.approx(0.5)
    assert len(set(weights[1:])) == 1
    assert cfg.events.reset_from_dataset.params["sequential"] is False


def test_held_goal_ablation_changes_only_post_grasp_goal_and_joint_margin():
    margin = SO101VialCameraScratchHeldGoalEnvCfg()
    no_margin = SO101VialCameraScratchHeldGoalNoMarginEnvCfg()

    assert margin.rewards.object_goal.params["use_live_grasp_goal"] is False
    assert margin.rewards.object_goal.params["require_lift_for_goal"] is False
    assert margin.rewards.joint_limit_margin.weight == pytest.approx(-0.1)
    assert no_margin.rewards.object_goal.params == margin.rewards.object_goal.params
    assert no_margin.rewards.joint_limit_margin.weight == 0.0
    assert no_margin.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["acquisition_pair"]

    clearance = SO101VialCameraScratchHeldGoalClearanceEnvCfg()
    assert clearance.rewards.object_goal.params == no_margin.rewards.object_goal.params
    assert clearance.rewards.joint_limit_margin.weight == 0.0
    assert clearance.rewards.rack_clearance.weight == pytest.approx(-10.0)


def test_action_term_can_only_command_articulation_targets():
    source = inspect.getsource(SoftLimitRelativeJointPositionAction) + inspect.getsource(SoftLimitRelativeGripperAction)

    assert "set_joint_position_target" in source
    for forbidden in ("write_root_pose", "write_root_velocity", 'scene["vial"]', 'scene["rack"]'):
        assert forbidden not in source


def test_canonical_bridge_radial_transport_has_long_range_gradient_and_no_depth_target():
    radial = SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg()
    no_clearance = SO101VialCameraScratchDiscoveryCanonicalBridgeRadialNoClearanceReleaseDirectEnvCfg()
    bridge = SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg()
    upright = SO101VialCameraScratchDiscoveryBridgeRadialUprightReleaseDirectEnvCfg()
    strong = SO101VialCameraScratchDiscoveryBridgeStrongRadialReleaseDirectEnvCfg()
    very_upright = SO101VialCameraScratchDiscoveryBridgeRadialVeryUprightReleaseDirectEnvCfg()
    progress = SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg()
    strict = SO101VialCameraScratchDiscoveryBridgeRadialStrictUprightReleaseDirectEnvCfg()
    strong_progress = SO101VialCameraScratchDiscoveryBridgeRadialStrongUprightProgressReleaseDirectEnvCfg()
    moderate = SO101VialCameraScratchDiscoveryBridgeRadialModerateUprightReleaseDirectEnvCfg()
    upright_only = SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg()
    upright_strict = SO101VialCameraScratchDiscoveryBridgeUprightStrictReleaseDirectEnvCfg()
    upright_strict_progress = SO101VialCameraScratchDiscoveryBridgeUprightStrictStrongProgressReleaseDirectEnvCfg()
    upright_progress = SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg()
    upright_normalized = SO101VialCameraScratchDiscoveryBridgeUprightNormalizedReleaseDirectEnvCfg()
    upright_efficient = SO101VialCameraScratchDiscoveryBridgeUprightEfficientReleaseDirectEnvCfg()
    upright_local_safe = SO101VialCameraScratchDiscoveryBridgeUprightLocalSafeReleaseDirectEnvCfg()
    upright_balanced_safe = SO101VialCameraScratchDiscoveryBridgeUprightBalancedSafeReleaseDirectEnvCfg()
    upright_conjunctive = SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveReleaseDirectEnvCfg()
    upright_conjunctive_only = SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg()
    upright_conjunctive_medium = SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveMediumReleaseDirectEnvCfg()
    upright_conjunctive_tight = SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveTightReleaseDirectEnvCfg()
    upright_lift = SO101VialCameraScratchDiscoveryBridgeUprightLiftReleaseDirectEnvCfg()
    lift_compensation = SO101VialCameraScratchDiscoveryBridgeLiftCompensationReleaseDirectEnvCfg()
    canonical_bridge_upright_lift = SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg()
    canonical_bridge_75_upright_lift = SO101VialCameraScratchDiscoveryCanonicalBridge75UprightLiftReleaseDirectEnvCfg()
    canonical_bridge_strong_upright_lift = (
        SO101VialCameraScratchDiscoveryCanonicalBridgeStrongUprightLiftReleaseDirectEnvCfg()
    )
    canonical_bridge_75_strong_upright_lift = (
        SO101VialCameraScratchDiscoveryCanonicalBridge75StrongUprightLiftReleaseDirectEnvCfg()
    )
    upright_safe = SO101VialCameraScratchDiscoveryBridgeUprightSafeReleaseDirectEnvCfg()
    upright_strict_safe = SO101VialCameraScratchDiscoveryBridgeUprightStrictSafeReleaseDirectEnvCfg()

    assert radial.rewards.held_radial_error.params["scale"] == pytest.approx(0.10)
    assert radial.rewards.object_goal.weight == 0.0
    assert radial.rewards.held_goal_progress.weight == 0.0
    assert radial.rewards.held_goal_error.weight == 0.0
    assert radial.rewards.held_clearance_error.weight == pytest.approx(-100.0)
    assert no_clearance.rewards.held_clearance_error.weight == 0.0
    assert bridge.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["bridge"]
    assert upright.rewards.held_upright_alignment.weight == pytest.approx(100.0)
    assert strong.rewards.held_radial_progress.weight == pytest.approx(500.0)
    assert strong.rewards.held_radial_error.weight == pytest.approx(-500.0)
    assert very_upright.rewards.held_upright_alignment.weight == pytest.approx(500.0)
    assert progress.rewards.held_upright_progress.weight == pytest.approx(100.0)
    assert strict.rewards.held_upright_alignment.weight == pytest.approx(1000.0)
    assert strong_progress.rewards.held_upright_progress.weight == pytest.approx(500.0)
    assert moderate.rewards.held_upright_alignment.weight == pytest.approx(300.0)
    assert upright_only.rewards.held_radial_progress.weight == 0.0
    assert upright_only.rewards.held_radial_center.weight == 0.0
    assert upright_only.rewards.held_radial_error.weight == 0.0
    assert upright_strict.rewards.held_upright_alignment.weight == pytest.approx(1000.0)
    assert upright_progress.rewards.held_upright_progress.weight == pytest.approx(500.0)
    assert upright_normalized.rewards.held_upright_alignment.weight == pytest.approx(5.0)
    assert upright_normalized.rewards.held_upright_progress.weight == pytest.approx(5.0)
    assert upright_normalized.rewards.held_clearance_error.weight == pytest.approx(-1.0)
    assert upright_efficient.rewards.arm_action_magnitude.weight == pytest.approx(-50.0)
    assert upright_local_safe.rewards.held_clearance_error.weight == pytest.approx(-500.0)
    assert upright_local_safe.rewards.held_clearance_error.params["scale"] == pytest.approx(0.02)
    assert upright_local_safe.rewards.vial_lost.weight == pytest.approx(-1000.0)
    assert upright_balanced_safe.rewards.held_clearance_error.weight == pytest.approx(-300.0)
    assert upright_conjunctive.rewards.held_upright_alignment.weight == 0.0
    assert upright_lift.rewards.held_upright_alignment.weight == 0.0
    assert upright_lift.rewards.held_upright_progress.weight == 0.0
    assert upright_lift.rewards.held_clearance_error.weight == 0.0
    assert upright_lift.rewards.held_upright_lift.weight == pytest.approx(500.0)
    assert upright_lift.rewards.vial_lost.weight == pytest.approx(-1000.0)
    assert lift_compensation.rewards.held_upright_alignment.weight == 0.0
    assert lift_compensation.rewards.held_upright_progress.weight == 0.0
    assert lift_compensation.rewards.held_lift_clearance.weight == pytest.approx(500.0)
    assert (
        canonical_bridge_upright_lift.events.reset_from_dataset.params["phase_weights"]
        == RESET_CURRICULA["canonical_bridge_pair"]
    )
    assert canonical_bridge_upright_lift.rewards.held_upright_lift.weight == pytest.approx(5.0)
    assert canonical_bridge_upright_lift.rewards.held_radial_error.weight == 0.0
    assert (
        canonical_bridge_75_upright_lift.events.reset_from_dataset.params["phase_weights"]
        == RESET_CURRICULA["canonical_bridge_75"]
    )
    assert canonical_bridge_strong_upright_lift.rewards.held_upright_lift.weight == pytest.approx(20.0)
    assert canonical_bridge_75_strong_upright_lift.rewards.held_upright_lift.weight == pytest.approx(20.0)
    assert upright_conjunctive.rewards.held_clearance_error.weight == 0.0
    assert upright_conjunctive.rewards.held_upright_progress.weight == pytest.approx(100.0)
    assert upright_conjunctive.rewards.held_upright_clearance.weight == pytest.approx(500.0)
    assert upright_conjunctive.rewards.held_upright_clearance.params["height_std"] == pytest.approx(0.01)
    assert upright_conjunctive_only.rewards.held_upright_progress.weight == 0.0
    assert upright_conjunctive_medium.rewards.held_upright_clearance.params["height_std"] == pytest.approx(0.0075)
    assert upright_conjunctive_tight.rewards.held_upright_clearance.params["height_std"] == pytest.approx(0.005)
    assert upright_strict_progress.rewards.held_upright_alignment.weight == pytest.approx(1000.0)
    assert upright_strict_progress.rewards.held_upright_progress.weight == pytest.approx(500.0)
    assert upright_safe.rewards.held_clearance_error.weight == pytest.approx(-1000.0)
    assert upright_safe.rewards.held_clearance_error.params["scale"] == pytest.approx(0.10)
    assert upright_safe.rewards.vial_lost.weight == pytest.approx(-1000.0)
    assert upright_strict_safe.rewards.held_clearance_error.weight == pytest.approx(-5000.0)


def test_ppo_contracts():
    state = SO101StatePPORunnerCfg()
    camera = SO101CameraPPORunnerCfg()
    scratch = SO101CameraScratchPPORunnerCfg()

    assert state.num_steps_per_env == 64
    assert camera.num_steps_per_env == 32
    assert state.actor.hidden_dims == state.critic.hidden_dims == [256, 256, 128]
    assert state.algorithm.num_learning_epochs == camera.algorithm.num_learning_epochs == 5
    assert state.algorithm.num_mini_batches == 8
    assert camera.algorithm.num_mini_batches == 8
    assert state.algorithm.gamma == pytest.approx(0.995)
    assert camera.algorithm.gamma == pytest.approx(0.99)
    assert state.algorithm.lam == camera.algorithm.lam == pytest.approx(0.95)
    assert scratch.obs_groups == {
        "actor": ["wrist_rgb", "proprioception"],
        "critic": ["wrist_rgb", "proprioception"],
    }
    assert isinstance(scratch.critic, type(camera.actor))
    assert scratch.critic.distribution_cfg is None
    assert scratch.experiment_name != camera.experiment_name
    assert state.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert state.algorithm.schedule == "adaptive"
    assert camera.algorithm.learning_rate == pytest.approx(2.5e-4)
    assert camera.algorithm.schedule == "adaptive"
    assert camera.obs_groups["actor"] == ["wrist_rgb", "proprioception"]
    aggressive = SO101VisionTunedLongHorizonAggressivePostLiftScratchPPORunnerCfg()
    assert aggressive.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert aggressive.actor.distribution_cfg.init_std == pytest.approx(0.03)
    resume = SO101VisionTunedLongHorizonAggressivePostLiftResumeScratchPPORunnerCfg()
    assert resume.algorithm.class_name.endswith(":ResidualResumePPO")


def test_residual_resume_preserves_optimizer_state(monkeypatch):
    captured = {}

    def fake_load(self, loaded_dict, load_cfg, strict):
        captured.update(load_cfg)
        return True

    monkeypatch.setattr(FineTunePPO, "load", fake_load)
    algorithm = ResidualResumePPO.__new__(ResidualResumePPO)
    algorithm.actor = type("Actor", (), {"named_parameters": lambda self: []})()
    algorithm.critic = type("Critic", (), {})()

    assert algorithm.load({}, None, True) is True
    assert captured["optimizer"] is True


def test_frozen_stats_finetune_stops_loaded_normalizers(monkeypatch):
    monkeypatch.setattr(FineTunePPO, "load", lambda self, loaded, cfg, strict: True)
    count = type("Count", (), {"item": lambda self: 37})()
    normalizer = type("Normalizer", (), {"count": count})()
    actor = type("Actor", (), {"obs_normalizer": normalizer})()
    critic = type("Critic", (), {"obs_normalizer": normalizer})()
    algorithm = FrozenStatsFineTunePPO.__new__(FrozenStatsFineTunePPO)
    algorithm.actor = actor
    algorithm.critic = critic

    assert algorithm.load({}, None, True) is True
    assert normalizer.until == 37

    cfg = SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg()
    assert cfg.algorithm.class_name.endswith(":FrozenStatsFineTunePPO")
    assert SO101VisionTunedLongHorizonFrozenFastScratchPPORunnerCfg().algorithm.learning_rate == pytest.approx(3.0e-4)


def test_output_layer_finetune_changes_only_the_existing_action_map(monkeypatch):
    monkeypatch.setattr(FrozenStatsFineTunePPO, "load", lambda self, loaded, cfg, strict: True)
    encoder = torch.nn.Parameter(torch.ones(1))
    output = torch.nn.Parameter(torch.ones(1))
    actor = type(
        "Actor",
        (),
        {"named_parameters": lambda self: [("cnn.weight", encoder), ("mlp.6.weight", output)]},
    )()
    algorithm = OutputLayerFrozenStatsFineTunePPO.__new__(OutputLayerFrozenStatsFineTunePPO)
    algorithm.actor = actor

    assert algorithm.load({}, None, True) is True
    assert encoder.requires_grad is False
    assert output.requires_grad is True

    selected = SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg()
    fast = SO101VisionTunedLongHorizonFrozenOutputFastScratchPPORunnerCfg()
    midrate = SO101VisionTunedLongHorizonFrozenOutputMidrateScratchPPORunnerCfg()
    assert selected.algorithm.class_name.endswith(":OutputLayerFrozenStatsFineTunePPO")
    assert selected.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert fast.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert midrate.algorithm.learning_rate == pytest.approx(2.0e-4)


def test_proximal_compensation_freezes_only_wrist_output_rows(monkeypatch):
    monkeypatch.setattr(OutputLayerFrozenStatsFineTunePPO, "load", lambda self, loaded, cfg, strict: True)
    weight = torch.nn.Parameter(torch.ones((6, 2)))
    bias = torch.nn.Parameter(torch.ones(6))
    actor = type(
        "Actor",
        (),
        {"named_parameters": lambda self: [("mlp.6.weight", weight), ("mlp.6.bias", bias)]},
    )()
    algorithm = ProximalOutputCompensationPPO.__new__(ProximalOutputCompensationPPO)
    algorithm.actor = actor

    assert algorithm.load({}, None, True) is True
    (weight.sum() + bias.sum()).backward()
    assert torch.all(weight.grad[:3] == 1.0)
    assert torch.all(weight.grad[3:5] == 0.0)
    assert torch.all(weight.grad[5:] == 1.0)
    assert torch.all(bias.grad[:3] == 1.0)
    assert torch.all(bias.grad[3:5] == 0.0)
    assert torch.all(bias.grad[5:] == 1.0)
    cfg = SO101VisionTunedLongHorizonProximalCompensationScratchPPORunnerCfg()
    assert cfg.algorithm.class_name.endswith(":ProximalOutputCompensationPPO")


def test_output_layer_resume_retains_optimizer_state(monkeypatch):
    captured = {}

    def fake_load(self, loaded_dict, load_cfg, strict):
        captured.update(load_cfg)
        return True

    monkeypatch.setattr(OutputLayerFrozenStatsFineTunePPO, "load", fake_load)
    algorithm = OutputLayerFrozenStatsResumePPO.__new__(OutputLayerFrozenStatsResumePPO)

    assert algorithm.load({}, None, True) is True
    assert captured["optimizer"] is True
    assert SO101VisionTunedLongHorizonFrozenOutputResumeScratchPPORunnerCfg().algorithm.class_name.endswith(
        ":OutputLayerFrozenStatsResumePPO"
    )


def test_full_policy_resume_retains_optimizer_state(monkeypatch):
    captured = {}

    def fake_load(self, loaded_dict, load_cfg, strict):
        captured.update(load_cfg)
        return True

    monkeypatch.setattr(FrozenStatsFineTunePPO, "load", fake_load)
    algorithm = FrozenStatsResumePPO.__new__(FrozenStatsResumePPO)

    assert algorithm.load({}, None, True) is True
    assert captured["optimizer"] is True
    assert SO101VisionTunedLongHorizonFrozenResumeScratchPPORunnerCfg().algorithm.class_name.endswith(
        ":FrozenStatsResumePPO"
    )


def test_distillation_keeps_privileged_state_out_of_student():
    cfg = SO101CameraDistillationRunnerCfg()

    assert cfg.obs_groups["student"] == ["wrist_rgb", "proprioception"]
    assert cfg.obs_groups["teacher"] == ["teacher_state"]
    assert cfg.student.cnn_cfg.output_channels == [16, 32, 32]
    wide = SO101WideCameraDistillationRunnerCfg()
    assert wide.student.cnn_cfg.output_channels == [32, 64, 64]
    assert wide.student.hidden_dims == [512, 256]
    teacher_rollout = SO101WideTeacherRolloutDistillationRunnerCfg()
    assert teacher_rollout.obs_groups["student"] == ["wrist_rgb", "proprioception"]
    assert teacher_rollout.algorithm.class_name.endswith(":TeacherRolloutDistillation")

    temporal = SO101WideTemporalCameraDistillationRunnerCfg()
    temporal_teacher = SO101WideTemporalTeacherRolloutDistillationRunnerCfg()
    assert temporal.obs_groups["student"] == ["wrist_rgb", "proprioception"]
    assert temporal_teacher.algorithm.class_name.endswith(":TeacherRolloutDistillation")

    fine = SO101WideCameraFineDistillationRunnerCfg()
    assert fine.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert fine.algorithm.class_name.endswith(":FineTuneDistillation")
    assert fine.save_interval == 10
    fine_teacher = SO101WideCameraFineTeacherRolloutRunnerCfg()
    assert fine_teacher.algorithm.class_name.endswith(":FineTuneTeacherRolloutDistillation")
    assert fine_teacher.algorithm.learning_rate == fine.algorithm.learning_rate


def test_fine_distillation_resets_optimizer_but_retains_models_and_iteration(monkeypatch):
    captured = {}

    def fake_load(self, loaded_dict, load_cfg, strict):
        captured.update(load_cfg)
        return load_cfg["iteration"]

    monkeypatch.setattr(Distillation, "load", fake_load)
    algorithm = FineTuneDistillation.__new__(FineTuneDistillation)

    assert algorithm.load({"student_state_dict": {}}, None, True) is True
    assert captured == {"student": True, "teacher": True, "optimizer": False, "iteration": True}


def test_temporal_camera_keeps_resolution_and_adds_only_one_rgb_frame():
    cfg = SO101VialTemporalCameraEnvCfg()

    assert (cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height) == (64, 64)
    assert cfg.observations.wrist_rgb.image.params["history_length"] == 2
    assert cfg.observations.wrist_rgb.image.func.__name__ == "TemporalDomainRandomizedCameraImage"
    assert set(cfg.observations.proprioception.__dict__) >= {
        "joint_pos",
        "joint_vel",
        "joint_target",
        "previous_action",
    }


def test_scratch_vision_uses_camera_actor_without_a_teacher():
    cfg = SO101CameraScratchPPORunnerCfg()

    assert cfg.experiment_name == "so101_vial_camera_scratch"
    assert cfg.actor.cnn_cfg.output_channels == [16, 32, 32]
    assert SO101WideCameraPPORunnerCfg().actor.cnn_cfg.output_channels == [32, 64, 64]
    assert cfg.obs_groups["actor"] == ["wrist_rgb", "proprioception"]


def test_wide_camera_exploration_ablation_changes_only_exploration_pressure():
    baseline = SO101WideCameraPPORunnerCfg()
    exploration = SO101WideCameraExplorationPPORunnerCfg()

    assert exploration.actor.cnn_cfg == baseline.actor.cnn_cfg
    assert exploration.actor.hidden_dims == baseline.actor.hidden_dims
    assert exploration.actor.distribution_cfg.init_std == pytest.approx(0.2)
    assert baseline.actor.distribution_cfg.init_std == pytest.approx(0.1)
    assert exploration.algorithm.entropy_coef == pytest.approx(0.005)
    assert baseline.algorithm.entropy_coef == pytest.approx(0.0)


def test_shared_scratch_uses_one_vision_encoder_for_actor_and_critic_learning():
    shared = SO101WideCameraSharedScratchPPORunnerCfg()

    assert shared.obs_groups["actor"] == shared.obs_groups["critic"] == ["wrist_rgb", "proprioception"]
    assert shared.algorithm.share_cnn_encoders is True
    assert shared.actor.cnn_cfg == shared.critic.cnn_cfg


def test_spatial_softmax_uses_upstream_keypoint_model_and_proprioception():
    distilled = SO101SpatialCameraDistillationRunnerCfg()
    warm_start = SO101SpatialTeacherRolloutDistillationRunnerCfg()
    fine = SO101SpatialFineDistillationRunnerCfg()
    dense = SO101SpatialDenseDistillationRunnerCfg()
    geometry = SO101GeometrySpatialDistillationRunnerCfg()
    geometry_teacher = SO101GeometrySpatialTeacherRolloutRunnerCfg()
    ppo = SO101SpatialCameraPPORunnerCfg()
    scratch = SO101SpatialCameraScratchPPORunnerCfg()

    expected_model = "isaaclab_tasks.core.lift.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    assert distilled.student.class_name == expected_model
    assert distilled.student.cnn_cfg.output_channels == [16, 32, 32]
    assert distilled.student.cnn_cfg.kernel_size == [8, 4, 3]
    assert distilled.student.cnn_cfg.stride == [4, 2, 1]
    assert distilled.obs_groups["student"] == ["wrist_rgb", "proprioception"]
    assert warm_start.algorithm.class_name.endswith(":TeacherRolloutDistillation")
    assert fine.algorithm.class_name.endswith(":FineTuneDistillation")
    assert fine.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert fine.save_interval == 10
    assert dense.algorithm == distilled.algorithm
    assert dense.save_interval == 10
    assert geometry.student.class_name.endswith(":GeometrySpatialSoftmaxCNNModel")
    assert geometry.student.geometry_dim == 9
    assert geometry.algorithm.geometry_group == "visual_geometry"
    assert geometry.algorithm.geometry_loss_coef == pytest.approx(1.0)
    assert geometry_teacher.algorithm.class_name.endswith(":GeometryTeacherRolloutDistillation")
    assert ppo.algorithm.learning_rate == pytest.approx(7.0e-5)
    assert ppo.algorithm.schedule == "fixed"
    assert scratch.obs_groups["actor"] == ["wrist_rgb", "proprioception"]
    assert scratch.obs_groups["critic"] == ["critic"]
    assert scratch.actor.class_name == expected_model
    assert scratch.critic.class_name == "MLPModel"


def test_camera_fine_ppo_uses_fixed_low_rate_and_fresh_optimizer(monkeypatch):
    cfg = SO101WideCameraFinePPORunnerCfg()
    assert cfg.algorithm.learning_rate == pytest.approx(5.0e-5)
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.class_name.endswith(":FineTunePPO")
    assert cfg.save_interval == 10

    captured = {}

    def fake_load(self, loaded_dict, load_cfg, strict):
        captured.update(load_cfg)
        return load_cfg["iteration"]

    monkeypatch.setattr("rsl_rl.algorithms.PPO.load", fake_load)
    algorithm = FineTunePPO.__new__(FineTunePPO)
    assert algorithm.load({}, None, True) is True
    assert captured["optimizer"] is False


def test_reset_generator_instantiates_without_a_reward_manager():
    cfg = SO101VialGeneratorEnvCfg()

    assert cfg.rewards is None


def test_reset_distributions_are_named_and_training_is_fixed_to_horizon():
    for weights in RESET_CURRICULA.values():
        assert len(weights) == 8
        assert all(weight >= 0.0 for weight in weights)
        assert any(weight > 0.0 for weight in weights)

    assert reset_curriculum_weights() == RESET_CURRICULA["horizon"]
    assert reset_curriculum_minimum_difficulty() is None
    assert reset_curriculum_maximum_difficulty() is None
