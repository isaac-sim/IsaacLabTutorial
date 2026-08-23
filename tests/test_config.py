"""Configuration contract tests for both public task variants."""

import inspect

import pytest

from so101_vial_place.agents.rsl_rl_distillation_cfg import SO101CameraDistillationRunnerCfg
from so101_vial_place.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101CameraScratchPPORunnerCfg,
    SO101StatePPORunnerCfg,
)
from so101_vial_place.camera_env_cfg import SO101VialCameraEnvCfg
from so101_vial_place.control import (
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_VIAL_HEADING_RANGE,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from so101_vial_place.env_cfg import (
    ARM_JOINTS,
    JOINTS,
    InitialEventsCfg,
    ResetJointActionsCfg,
    SO101VialEnvCfg,
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
    assert cfg.actions.arm_action.scale == pytest.approx(0.03)
    assert cfg.actions.arm_action.use_zero_offset is True
    assert cfg.actions.gripper_action.joint_names == ["gripper"]
    assert cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is True
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.scene.robot.init_state.joint_pos["gripper"] == pytest.approx(PREGRASP_GRIPPER_POSITION)
    assert tuple(cfg.scene.robot.init_state.joint_pos.values()) == pytest.approx(WORKSHOP_INITIAL_JOINT_POSITION)
    assert cfg.sim.default_visualizer_cfg.eye == pytest.approx((0.64, 0.0, 0.36))
    assert cfg.sim.default_visualizer_cfg.lookat == pytest.approx((0.19, 0.02, 0.075))
    assert len(JOINTS) == 6
    assert cfg.actions.gripper_action.scale == pytest.approx(0.02)
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


def test_play_mode_can_evaluate_a_filtered_reset_curriculum(monkeypatch):
    monkeypatch.setenv("SO101_RESET_CURRICULUM", "horizon")
    cfg = SO101VialEnvCfg()

    cfg.play_mode()

    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["horizon"]
    assert cfg.events.reset_from_dataset.params["minimum_difficulty"] is None


def test_play_mode_defaults_to_validated_initial_phase(monkeypatch):
    monkeypatch.delenv("SO101_RESET_CURRICULUM", raising=False)
    monkeypatch.delenv("SO101_EVAL_RAW_TABLETOP", raising=False)
    cfg = SO101VialEnvCfg()

    cfg.play_mode()

    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == RESET_CURRICULA["initial"]


def test_tabletop_episodes_reset_every_dynamic_scene_object():
    events = InitialEventsCfg()

    assert events.reset_vial.params["asset_cfg"].name == "vial"
    assert events.reset_vial.params["pose_range"]["yaw"] == TABLETOP_VIAL_HEADING_RANGE
    assert events.reset_rack.params["asset_cfg"].name == "rack"
    assert set(events.reset_rack.params["pose_range"]) == {"x", "y", "z", "roll", "pitch", "yaw"}
    assert all(bounds == (0.0, 0.0) for bounds in events.reset_rack.params["pose_range"].values())


def test_exact_evaluation_retains_requested_batch(monkeypatch):
    monkeypatch.setenv("SO101_EVAL_EPISODES", "1000")
    state = SO101VialEnvCfg()
    camera = SO101VialCameraEnvCfg()

    state.play_mode()
    camera.play_mode()

    assert state.scene.num_envs == 128
    assert camera.scene.num_envs == 128


def test_camera_actor_is_unprivileged_and_has_only_wrist_image():
    cfg = SO101VialCameraEnvCfg()
    critic_terms = set(cfg.observations.critic.__dict__)

    assert cfg.scene.num_envs == 1024
    assert (cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height) == (64, 48)
    assert cfg.scene.wrist_camera.prim_path.endswith("/gripper/wrist_camera")
    assert cfg.scene.wrist_camera.offset.pos == pytest.approx((-0.055, 0.052, -0.035))
    assert cfg.scene.wrist_camera.offset.rot == pytest.approx((-0.09871531, 0.59943614, 0.78375556, -0.12906908))
    assert cfg.scene.wrist_camera.offset.convention == "opengl"
    calibration = cfg.scene.wrist_camera.spawn.distortion
    assert calibration.image_size == (64, 48)
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


def test_action_term_can_only_command_articulation_targets():
    source = inspect.getsource(SoftLimitRelativeJointPositionAction) + inspect.getsource(SoftLimitRelativeGripperAction)

    assert "set_joint_position_target" in source
    for forbidden in ("write_root_pose", "write_root_velocity", 'scene["vial"]', 'scene["rack"]'):
        assert forbidden not in source


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


def test_distillation_keeps_privileged_state_out_of_student():
    cfg = SO101CameraDistillationRunnerCfg()

    assert cfg.obs_groups["student"] == ["wrist_rgb", "proprioception"]
    assert cfg.obs_groups["teacher"] == ["teacher_state"]


def test_scratch_vision_uses_camera_actor_without_a_teacher():
    cfg = SO101CameraScratchPPORunnerCfg()

    assert cfg.experiment_name == "so101_vial_camera_scratch"
    assert cfg.obs_groups["actor"] == ["wrist_rgb", "proprioception"]


def test_reset_curriculum_is_named_and_covers_every_phase(monkeypatch):
    for weights in RESET_CURRICULA.values():
        assert len(weights) == 8
        assert all(weight >= 0.0 for weight in weights)
        assert any(weight > 0.0 for weight in weights)

    monkeypatch.setenv("SO101_RESET_CURRICULUM", "horizon")
    assert reset_curriculum_weights() == RESET_CURRICULA["horizon"]
    for stage in RESET_CURRICULA:
        monkeypatch.setenv("SO101_RESET_CURRICULUM", stage)
        assert reset_curriculum_minimum_difficulty() is None
        assert reset_curriculum_maximum_difficulty() is None

    monkeypatch.setenv("SO101_RESET_CURRICULUM", "not-a-stage")
    with pytest.raises(ValueError, match="Unknown SO101_RESET_CURRICULUM"):
        reset_curriculum_weights()
