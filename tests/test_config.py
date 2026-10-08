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
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import (
    ALL_PHASES,
    CANONICAL_START,
)
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
    assert cfg.sim.physics.physx.ovphysx.enable_external_forces_every_iteration is False
    assert cfg.scene.vial.spawn.rigid_props.physx.solver_position_iteration_count == iterations
    assert cfg.scene.vial.spawn.rigid_props.default is None
    assert cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 8
    # The shared contact material carries Newton's contact stiffness/damping as a PhysX compliant contact.
    friction = cfg.events.vial_material.params["static_friction_range"]
    assert friction.default == so101_env_cfg.VIAL_FRICTION_RANGE == (0.7, 1.3)
    assert friction.physx == so101_env_cfg.PHYSX_VIAL_FRICTION_RANGE
    assert friction.physx[1] < friction.default[0]  # the PhysX range sits below Newton's
    material = so101_env_cfg.WORKSHOP_CONTACT_MATERIAL
    assert material.compliant_contact_stiffness == pytest.approx(1.57e5)
    assert material.compliant_contact_damping == pytest.approx(1.12e3)
    assert material.compliant_contact_acceleration_spring is True
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
    assert cfg.events.vial_material.params["static_friction_range"] == (0.7, 0.7)
    assert cfg.events.vial_material.params["dynamic_friction_range"] == (0.7, 0.7)
    assert cfg.events.vial_material.params["restitution_range"] == (0.0, 0.0)
    assert cfg.events.vial_mass.params["mass_distribution_params"] == (0.020, 0.020)


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
    assert cfg.scene.wrist_camera.prim_path == "{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera/pinhole"
    assert cfg.scene.wrist_camera.spawn.distortion is None
    assert cfg.scene.wrist_camera.spawn.focal_length == pytest.approx(13.6)
    assert cfg.scene.wrist_camera.spawn.horizontal_aperture == pytest.approx(20.955)
    assert cfg.scene.wrist_camera.offset.pos == (0.0, 0.0, 0.0)
    assert cfg.scene.wrist_camera.offset.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.scene.wrist_camera.offset.convention == "opengl"
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


def test_wrist_camera_hides_only_its_visual_housing(monkeypatch):
    from pxr import Usd, UsdGeom

    from isaaclab_tutorial.tasks.place_vial.config.so101 import camera_env_cfg

    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    UsdGeom.Xform.Define(stage, "/Robot/gripper/visuals/camera_mount")
    housing = UsdGeom.Cube.Define(stage, "/Robot/gripper/visuals/camera_mount/lens")
    collision = UsdGeom.Cube.Define(stage, "/Robot/gripper/collisions/camera_mount")
    jaw = UsdGeom.Cube.Define(stage, "/Robot/gripper/visuals/jaw")
    monkeypatch.setattr(camera_env_cfg, "_spawn_so101_with_camera_overrides", lambda *args, **kwargs: robot)
    camera_env_cfg._spawn_so101_for_wrist_camera.__wrapped__("/Robot", None)
    assert housing.ComputeVisibility() == UsdGeom.Tokens.invisible
    assert collision.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert jaw.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert SO101VialCameraEnvCfg().scene.robot.spawn.func is camera_env_cfg._spawn_so101_for_wrist_camera
    assert SO101VialEnvCfg().scene.robot.spawn.func is not camera_env_cfg._spawn_so101_for_wrist_camera
