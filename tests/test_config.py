"""Configuration contracts for the public tutorial tasks."""

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
    assert cfg.scene.robot.actuators == SO101_CFG.actuators


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
    assert cfg.scene.robot.spawn.variants == {"Robot": "robot", "Sensor": "sensors", "Physics": "physics"}
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
