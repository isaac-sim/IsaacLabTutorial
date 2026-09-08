"""Configuration contracts for the public tutorial tasks."""

import math

import pytest
from isaaclab_assets.robots.so101 import SO101_CFG

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    SO101CameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101StatePPORunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import (
    SO101VialCameraDistillationEnvCfg,
    SO101VialCameraEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    ARM_JOINTS,
    JOINTS,
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_VIAL_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
    SO101VialEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START
from isaaclab_tutorial.utils import evaluation


def test_state_task_control_and_physics_contract():
    cfg = SO101VialEnvCfg()
    physics = cfg.sim.physics.newton_mjwarp

    assert cfg.scene.num_envs == 4096
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.episode_length_s == 20.0
    assert physics.num_substeps == 2
    assert physics.solver_cfg.solver == "newton"
    assert physics.solver_cfg.njmax == 300
    assert physics.solver_cfg.nconmax == 200
    assert physics.collision_cfg.broad_phase == "explicit"
    assert cfg.actions.arm_action.joint_names == ARM_JOINTS
    assert cfg.actions.arm_action.scale == pytest.approx(0.033)
    assert cfg.actions.gripper_action.joint_names == ["gripper"]
    assert cfg.actions.gripper_action.scale == pytest.approx(0.02)
    assert cfg.scene.robot.init_state.joint_pos["gripper"] == pytest.approx(PREGRASP_GRIPPER_POSITION)
    assert cfg.scene.vial.init_state.pos == pytest.approx(TABLETOP_VIAL_POSITION)
    assert tuple(cfg.scene.robot.init_state.joint_pos.values()) == pytest.approx(WORKSHOP_INITIAL_JOINT_POSITION)
    assert len(JOINTS) == 6

    assert cfg.scene.robot.spawn.usd_path == SO101_CFG.spawn.usd_path
    assert cfg.scene.robot.spawn.activate_contact_sensors is True
    # Newton uses the USD-authored Sys-ID dynamics; PhysX re-applies them through the actuator configuration.
    actuator = cfg.scene.robot.actuators["arm"]
    gripper = cfg.scene.robot.actuators["gripper"]
    assert gripper.joint_names_expr == ["gripper"]
    assert gripper.stiffness.newton_mjwarp is None and gripper.stiffness.default is None
    assert gripper.stiffness.physx["gripper"] == pytest.approx(0.6 * 68.2508, abs=1e-3)
    assert gripper.friction.physx["gripper"] == pytest.approx(2.0 * 0.083458, abs=1e-6)
    assert actuator.stiffness is None and actuator.damping is None
    for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        assert getattr(actuator, field).newton_mjwarp is None
        assert set(getattr(actuator, field).physx) | set(getattr(gripper, field).physx) == set(JOINTS)
    assert actuator.viscous_friction.physx["shoulder_pan"] == pytest.approx(2.0 * 1.5907, abs=2e-3)
    assert actuator.friction.physx["elbow_flex"] == pytest.approx(0.41119, abs=1e-6)
    assert actuator.friction.physx["shoulder_lift"] == pytest.approx(1.5 * 0.344793, abs=1e-6)
    assert actuator.viscous_friction.physx["wrist_flex"] == pytest.approx(3.0 * math.degrees(0.0187076781), abs=1e-4)
    assert cfg.scene.robot.spawn.variants["Physics"].newton_mjwarp == "physics"
    assert cfg.scene.robot.spawn.variants["Physics"].physx == "physx"
    assert cfg.sim.physics.physx.ovphysx is not None
    # PhysX needs more solver iterations to hold the jaw/vial pinch; Newton keeps the asset defaults.
    from isaaclab_tutorial.tasks.place_vial.config.so101 import env_cfg as so101_env_cfg

    iterations = so101_env_cfg.PHYSX_SOLVER_POSITION_ITERATIONS
    assert iterations >= 128
    ovphysx = cfg.sim.physics.physx.ovphysx
    assert ovphysx.rigid_body_position_iteration_count == iterations
    assert ovphysx.articulation_position_iteration_count == iterations
    assert cfg.scene.vial.spawn.rigid_props is None
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    # The shared contact material carries Newton's contact stiffness/damping as a PhysX compliant contact.
    friction = cfg.events.vial_material.params["static_friction_range"]
    assert friction.default == so101_env_cfg.VIAL_FRICTION_RANGE == (0.7, 1.3)
    assert friction.physx == so101_env_cfg.PHYSX_VIAL_FRICTION_RANGE
    assert friction.physx[1] < friction.default[0]  # the PhysX range sits below Newton's
    material = so101_env_cfg.WORKSHOP_CONTACT_MATERIAL
    assert material.compliant_contact_stiffness == pytest.approx(1.57e5)
    assert material.compliant_contact_damping == pytest.approx(1.12e3)
    assert material.friction_combine_mode == "max"
    assert material.static_friction == material.dynamic_friction == pytest.approx(0.7)


def test_training_samples_every_phase_and_play_uses_canonical_starts(monkeypatch):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", False)
    cfg = SO101VialEnvCfg()
    training = dict(cfg.events.reset_from_dataset.params)

    cfg.play_mode()

    play = cfg.events.reset_from_dataset.params
    assert training["sequential"] is False
    assert training["phase_weights"] == ALL_PHASES
    assert play["sequential"] is True
    assert play["phase_weights"] == CANONICAL_START
    assert cfg.scene.num_envs == 16


def test_exact_evaluation_retains_requested_batch(monkeypatch):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    state = SO101VialEnvCfg()
    camera = SO101VialCameraEnvCfg()

    state.play_mode()
    camera.play_mode()

    assert state.scene.num_envs == 1024
    assert camera.scene.num_envs == 1024


def test_camera_actor_observation_boundary():
    cfg = SO101VialCameraEnvCfg()

    assert cfg.scene.num_envs == 1024
    assert (cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height) == (64, 48)
    assert cfg.scene.wrist_camera.prim_path == "{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera"
    assert cfg.scene.wrist_camera.spawn is None
    assert cfg.scene.wrist_camera.offset.pos == (0.0, 0.0, 0.0)
    assert cfg.scene.wrist_camera.offset.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.scene.wrist_camera.offset.convention == "ros"
    assert cfg.scene.wrist_camera.data_types == ["rgb"]
    assert cfg.scene.wrist_camera.update_period == pytest.approx(1.0 / 30.0)
    assert cfg.scene.wrist_camera.update_latest_camera_pose is True
    assert cfg.scene.robot.spawn.variants["Physics"].default == "physics"
    assert set(cfg.observations.__dict__) >= {"wrist_rgb", "proprioception", "critic"}
    assert "teacher_state" not in cfg.observations.__dict__
    assert set(cfg.observations.proprioception.__dict__) >= {
        "joint_pos",
        "joint_vel",
        "joint_target",
        "previous_action",
    }
    assert not {"vial", "rack_target", "progress"} & set(cfg.observations.proprioception.__dict__)
    assert cfg.observations.wrist_rgb.enable_corruption is True
    assert cfg.observations.proprioception.enable_corruption is True


def test_distillation_task_only_adds_the_teacher_observation():
    cfg = SO101VialCameraDistillationEnvCfg()
    camera = SO101VialCameraEnvCfg()

    assert type(cfg.scene) is type(camera.scene)
    assert cfg.events.reset_from_dataset.params == camera.events.reset_from_dataset.params
    assert set(cfg.observations.__dict__) - set(camera.observations.__dict__) == {"teacher_state"}


def test_agent_configs_match_task_observation_groups():
    state = SO101StatePPORunnerCfg()
    camera = SO101CameraPPORunnerCfg()
    distillation = SO101CameraDistillationRunnerCfg()

    assert state.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert camera.obs_groups == {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    assert distillation.obs_groups == {
        "student": ["wrist_rgb", "proprioception"],
        "teacher": ["teacher_state"],
    }
    assert distillation.clip_actions == pytest.approx(1.0)
    assert distillation.algorithm.class_name.endswith(":BoundedTeacherDistillation")
    # The teacher must mirror the state actor so the PPO checkpoint loads into it.
    assert distillation.teacher.hidden_dims == state.actor.hidden_dims
    assert distillation.teacher.distribution_cfg.std_type == state.actor.distribution_cfg.std_type
    # The student and the from-scratch visual actor share one encoder definition.
    assert distillation.student.cnn_cfg == camera.actor.cnn_cfg
    assert camera.actor.obs_normalization is True


def test_record_task_keeps_state_observations():
    import gymnasium as gym

    from isaaclab_tutorial.tasks.place_vial.config.so101.record_env_cfg import (
        RECORD_CAMERA_EYE,
        RECORD_CAMERA_TARGET,
        SO101VialRecordEnvCfg,
        look_at_quaternion,
    )

    assert "IsaacTutorial-Place-Vial-SO101-Record" in gym.registry
    cfg = SO101VialRecordEnvCfg()
    assert cfg.scene.record_camera.width == 320 and cfg.scene.record_camera.height == 240
    assert cfg.observations.policy.__class__ is SO101VialEnvCfg().observations.policy.__class__
    q = look_at_quaternion(RECORD_CAMERA_EYE, RECORD_CAMERA_TARGET)
    assert sum(v * v for v in q) == pytest.approx(1.0)
    x, y, z, w = q
    forward = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
    direction = [t - e for t, e in zip(RECORD_CAMERA_TARGET, RECORD_CAMERA_EYE, strict=True)]
    norm = math.sqrt(sum(v * v for v in direction))
    assert all(abs(f - d / norm) < 1e-6 for f, d in zip(forward, direction, strict=True))


def test_domain_randomization_variants():
    from isaaclab_tutorial.tasks.place_vial.config.so101.dr_env_cfg import (
        JOINT_FRICTION_SCALE_RANGE,
        JOINT_VISCOUS_SCALE_RANGE,
        SO101VialDREnvCfg,
        SO101VialDRWideEnvCfg,
    )

    narrow = SO101VialDREnvCfg()
    assert narrow.events.robot_joint_parameters.params["friction_distribution_params"] == JOINT_FRICTION_SCALE_RANGE
    assert narrow.events.robot_joint_parameters.params["operation"] == "scale"
    assert narrow.events.robot_viscous_friction.params["scale_range"] == JOINT_VISCOUS_SCALE_RANGE
    assert narrow.events.robot_viscous_friction.mode == "reset"
    # The randomized ranges bracket the measured Newton -> OV PhysX actuator gap (friction x1.5-2, viscous x1.5-3).
    assert JOINT_FRICTION_SCALE_RANGE[0] < 1.0 < 2.0 <= JOINT_FRICTION_SCALE_RANGE[1]
    assert JOINT_VISCOUS_SCALE_RANGE[0] < 1.0 < 3.0 <= JOINT_VISCOUS_SCALE_RANGE[1]
    wide = SO101VialDRWideEnvCfg()
    assert wide.events.gripper_gains.params["asset_cfg"].joint_names == ["gripper"]
    assert wide.events.vial_scale.mode == "usd"
    # Observations and actions are untouched: any DR checkpoint plays on the plain task and vice versa.
    assert type(wide.observations) is type(SO101VialEnvCfg().observations)
    assert type(wide.actions) is type(SO101VialEnvCfg().actions)
