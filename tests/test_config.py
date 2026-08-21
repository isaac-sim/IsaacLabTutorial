"""Configuration contract tests for both public task variants."""

import pytest

from so101_vial_lift.agents.rsl_rl_ppo_cfg import SO101CameraPPORunnerCfg, SO101StatePPORunnerCfg
from so101_vial_lift.camera_env_cfg import SO101VialCameraEnvCfg
from so101_vial_lift.env_cfg import JOINTS, SO101VialEnvCfg


def test_state_task_control_and_action_contract():
    cfg = SO101VialEnvCfg()

    assert cfg.scene.num_envs == 4096
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.episode_length_s == 12.0
    assert cfg.sim.physics.newton_mjwarp.num_substeps == 12
    assert cfg.actions.joint_delta.joint_names == JOINTS
    assert len(JOINTS) == 6
    assert cfg.actions.joint_delta.scale["gripper"] == pytest.approx(0.10)
    assert cfg.actions.joint_delta.scale["shoulder_.*|elbow_flex|wrist_.*"] == pytest.approx(0.05)


def test_camera_actor_is_unprivileged_and_has_two_images():
    cfg = SO101VialCameraEnvCfg()
    critic_terms = set(cfg.observations.critic.__dict__)

    assert cfg.scene.num_envs == 4096
    assert cfg.scene.ego_camera.width == cfg.scene.external_camera.width == 64
    assert cfg.scene.ego_camera.height == cfg.scene.external_camera.height == 64
    assert cfg.scene.ego_camera.update_period == pytest.approx(1.0 / 30.0)
    assert set(cfg.observations.__dict__) >= {"ego_rgb", "external_rgb", "proprioception", "critic"}
    assert set(cfg.observations.proprioception.__dict__) >= {"joint_pos", "joint_vel", "previous_action"}
    assert not {"vial", "rack_target", "progress"} & set(cfg.observations.proprioception.__dict__)
    assert {"vial", "rack_target", "progress"} <= critic_terms


def test_ppo_contracts():
    state = SO101StatePPORunnerCfg()
    camera = SO101CameraPPORunnerCfg()

    assert state.num_steps_per_env == camera.num_steps_per_env == 32
    assert state.actor.hidden_dims == state.critic.hidden_dims == [256, 256, 128]
    assert state.algorithm.num_learning_epochs == camera.algorithm.num_learning_epochs == 5
    assert state.algorithm.num_mini_batches == 4
    assert camera.algorithm.num_mini_batches == 8
    assert state.algorithm.gamma == camera.algorithm.gamma == pytest.approx(0.99)
    assert state.algorithm.lam == camera.algorithm.lam == pytest.approx(0.95)
    assert state.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert state.algorithm.schedule == "adaptive"
    assert camera.algorithm.learning_rate == pytest.approx(1.0e-4)
    assert camera.algorithm.schedule == "fixed"
    assert camera.obs_groups["actor"] == ["ego_rgb", "external_rgb", "proprioception"]
