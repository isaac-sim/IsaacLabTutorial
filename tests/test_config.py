"""Configuration contracts for the public tutorial tasks."""

import pytest

from so101_vial_place.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101StatePPORunnerCfg,
)
from so101_vial_place.tasks.place_vial.config.so101.camera_env_cfg import SO101VialCameraEnvCfg
from so101_vial_place.tasks.place_vial.config.so101.control import (
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_VIAL_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from so101_vial_place.tasks.place_vial.config.so101.state_env_cfg import ARM_JOINTS, JOINTS, SO101VialEnvCfg
from so101_vial_place.tasks.place_vial.reset.curriculum import RESET_CURRICULA
from so101_vial_place.utils import evaluation


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

    actuator = cfg.scene.robot.actuators["usd_sysid"]
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
        assert getattr(actuator, field) is None


def test_play_mode_uses_canonical_resets(monkeypatch):
    monkeypatch.setattr(evaluation, "PLAY_RESETS_SEQUENTIAL", True)
    cfg = SO101VialEnvCfg()

    cfg.play_mode()

    reset = cfg.events.reset_from_dataset.params
    assert reset["sequential"] is True
    assert reset["phase_weights"] == RESET_CURRICULA["initial"]
    assert reset["minimum_difficulty"] is None


def test_exact_evaluation_retains_requested_batch(monkeypatch):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    monkeypatch.setattr(evaluation, "PLAY_EVALUATION_EPISODES", 1024)
    state = SO101VialEnvCfg()
    camera = SO101VialCameraEnvCfg()

    state.play_mode()
    camera.play_mode()

    assert state.scene.num_envs == 1024
    assert camera.scene.num_envs == 1024


def test_camera_actor_observation_boundary():
    cfg = SO101VialCameraEnvCfg()

    assert cfg.scene.num_envs == 1024
    assert (cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height) == (64, 64)
    assert cfg.scene.wrist_camera.prim_path.endswith("/gripper/wowrobo_2MP_camera")
    assert cfg.scene.wrist_camera.spawn is None
    assert cfg.scene.wrist_camera.update_period == pytest.approx(1.0 / 30.0)
    assert set(cfg.observations.__dict__) >= {"wrist_rgb", "proprioception", "critic"}
    assert set(cfg.observations.proprioception.__dict__) >= {
        "joint_pos",
        "joint_vel",
        "joint_target",
        "previous_action",
    }
    assert not {"vial", "rack_target", "progress"} & set(cfg.observations.proprioception.__dict__)
    assert cfg.observations.wrist_rgb.enable_corruption is True
    assert cfg.observations.proprioception.enable_corruption is True


def test_agent_configs_match_task_observation_groups():
    state = SO101StatePPORunnerCfg()
    camera = SO101CameraPPORunnerCfg()

    assert state.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert camera.obs_groups == {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    assert state.actor.distribution_cfg.init_std == pytest.approx(0.2)
    assert camera.actor.distribution_cfg.init_std == pytest.approx(0.1)
