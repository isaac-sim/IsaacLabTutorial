# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Check the course's scientific comparison and joint-control contracts."""

import pytest

from isaaclab.utils import to_dict

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401


def config(name):
    return resolve_task_config(name, "rsl_rl_cfg_entry_point", overrides=[])[0]


def test_experiment_changes_only_slip_weight():
    before = to_dict(config("Course-Microduck-Walk-v0"))
    after = to_dict(config("Course-Microduck-LessSlip-v0"))
    before["rewards"]["foot_slip"]["weight"] = -0.5
    assert before == after


def test_fixture_is_not_present_in_learning_tasks():
    assert config("Course-Microduck-Inspect-v0").scene.robot.spawn.fix_root_link is True
    for name in ("Balance", "BalanceHead", "Walk", "LessSlip"):
        assert config(f"Course-Microduck-{name}-v0").scene.robot.spawn.fix_root_link is not True
    balance = config("Course-Microduck-Balance-v0")
    assert balance.commands.base_velocity.rel_standing_envs == 1.0
    assert balance.commands.base_velocity.rel_forward_envs == 0.0
    assert balance.commands.base_velocity.rel_turn_in_place_envs == 0.0
    assert config("Course-Microduck-Walk-v0").commands.base_velocity.ranges.lin_vel_x[1] > 0


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_tracking_builds_on_balance_without_duplicate_velocity_rewards(world_size, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    track, agent = resolve_task_config("Course-Microduck-Track-v0", "rsl_rl_cfg_entry_point", overrides=[])
    walk = config("Course-Microduck-Walk-v0")
    assert track.rewards.still_xy is None
    assert track.rewards.still_yaw is None
    for name in ("track_lin_vel", "track_ang_vel", "air_time", "foot_slip"):
        assert to_dict(getattr(track.rewards, name)) == to_dict(getattr(walk.rewards, name)), name
    assert track.commands.base_velocity.ranges.lin_vel_x == walk.commands.base_velocity.ranges.lin_vel_x
    assert to_dict(track.rewards.head_pose) == to_dict(walk.rewards.head_pose_tracking)
    stages = track.curriculum.action_rate_weight.params["modify_params"]["stages"]
    assert stages == walk.curriculum.action_rate_weight.params["modify_params"]["stages"]
    assert stages == [
        (iteration * agent.num_steps_per_env // world_size, weight)
        for iteration, weight in ((3000, -0.2), (4000, -0.4), (5000, -0.6), (6000, -0.8), (7000, -1.0))
    ]
    assert track.curriculum.action_rate_weight.params["address"] == "rewards.action_rate.weight"
    assert agent.max_iterations == 50000 // world_size
    assert stages[-1][0] < agent.max_iterations * agent.num_steps_per_env


def test_head_balance_adds_only_the_head_reward():
    base = to_dict(config("Course-Microduck-Balance-v0"))
    head = to_dict(config("Course-Microduck-BalanceHead-v0"))
    term = head["rewards"].pop("head_pose")
    assert base == head
    assert term["params"]["asset_cfg"]["joint_names"] == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    assert term["params"]["asset_cfg"]["preserve_order"] is True
    assert term["params"]["command_name"] == "head_pose"


def test_matched_comparison_uses_the_same_symmetry_runner():
    key = "rsl_rl_with_symmetry_cfg_entry_point"
    _, before = resolve_task_config("Course-Microduck-Walk-v0", key, overrides=[])
    _, after = resolve_task_config("Course-Microduck-LessSlip-v0", key, overrides=[])
    before, after = to_dict(before), to_dict(after)
    before.pop("experiment_name")
    after.pop("experiment_name")
    assert before == after
    assert before["algorithm"]["symmetry_cfg"]["use_data_augmentation"] is True
    assert before["algorithm"]["symmetry_cfg"]["use_mirror_loss"] is False


@pytest.mark.parametrize("task", ["Walk", "LessSlip"])
def test_comparison_can_disable_curriculum_without_losing_runner_schedules(task, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    env, agent = resolve_task_config(
        f"Course-Microduck-{task}-v0", "rsl_rl_with_symmetry_cfg_entry_point",
        overrides=["env.curriculum=null", "agent.algorithm.learning_rate=0.0001"],
    )
    assert env.curriculum is None
    assert agent.max_iterations == 25000
    assert agent.save_interval == 125
    assert agent.algorithm.learning_rate == 0.0001
    assert agent.algorithm.symmetry_cfg.use_data_augmentation is True

def test_rough_terrain_adds_height_sensors_for_the_critic():
    rough = config("Course-Microduck-Rough-v0")
    assert set(rough.scene.terrain.terrain_generator.sub_terrains) == {
        "flat", "pyramid_stairs", "random_grid", "pyramid_slope"
    }
    assert rough.observations.critic.foot_height.params["height_sensor_names"] == (
        "left_foot_height", "right_foot_height"
    )
    assert to_dict(rough.observations.policy) == to_dict(config("Course-Microduck-Walk-v0").observations.policy)
