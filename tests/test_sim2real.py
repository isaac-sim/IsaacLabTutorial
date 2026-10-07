"""Acceptance audits must retain the physics distribution used to train both policies."""

import pytest

from isaaclab_tutorial.tasks.place_vial.config.so101.sim2real_env_cfg import (
    SO101VialCameraSim2RealEnvCfg,
    SO101VialSim2RealEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import CANONICAL_START


@pytest.mark.parametrize("config", [SO101VialSim2RealEnvCfg, SO101VialCameraSim2RealEnvCfg])
def test_home_audit_retains_randomized_physics(config):
    cfg = config()
    material = cfg.events.vial_material.params.copy()
    mass = cfg.events.vial_mass.params.copy()
    cfg.play_mode()
    assert cfg.episode_length_s == 30.0
    assert cfg.events.vial_material.params == material
    assert cfg.events.vial_mass.params == mass
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    assert cfg.events.reset_from_dataset.params["home_position_noise"] == 0.02
    assert cfg.events.robot_joint_parameters.mode == "reset"
    assert cfg.events.robot_actuator_gains.params["asset_cfg"].joint_names != ["gripper"]


def test_state_and_student_share_physics():
    state, camera = SO101VialSim2RealEnvCfg(), SO101VialCameraSim2RealEnvCfg()
    assert state.events.to_dict() == camera.events.to_dict()
    assert not hasattr(camera.observations, "policy")
