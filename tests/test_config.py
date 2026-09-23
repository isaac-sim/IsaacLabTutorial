"""Configuration contracts for the public tutorial tasks."""

import pytest
from isaaclab_assets.robots.so101 import SO101_CFG

from so101_place_vial.tasks.place_vial.config.so101.env_cfg import (
    ARM_JOINTS,
    JOINTS,
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_VIAL_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
    SO101VialEnvCfg,
)
from so101_place_vial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START
from so101_place_vial.utils import evaluation


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
