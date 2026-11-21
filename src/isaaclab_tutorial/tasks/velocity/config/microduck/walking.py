# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walking scene, rewards, and curriculum shared by the MicroDuck tasks."""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import clone, replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import preset

from isaaclab_tutorial.tasks.velocity.mdp import rewards
from isaaclab_tutorial.tasks.velocity.mdp.curriculums import staged_value

from .managers import (
    _backlash_preset,
    _FOOT_BODY_CFG,
    _FOOT_SENSOR_CFG,
    _SERVO_JOINT_CFG,
    _TRUNK_BODY_CFG,
)
from .robot import MICRODUCK_CFG, MICRODUCK_JOINT_NAMES
from .scene import DuckSceneCfg

MICRODUCK_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
MICRODUCK_LEG_JOINT_NAMES = [name for name in MICRODUCK_JOINT_NAMES if name not in MICRODUCK_HEAD_JOINT_NAMES]
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_HEAD_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_HEAD_JOINT_NAMES], preserve_order=True
)
MICRODUCK_SOLE_TO_ANKLE_OFFSET = 0.0225
"""Standing ankle-frame height above the sole [m]."""
MICRODUCK_FOOT_TARGET_HEIGHT = 0.02 + MICRODUCK_SOLE_TO_ANKLE_OFFSET
MICRODUCK_STEPS_PER_ITERATION = 24


def _with_backlash(cfg: ArticulationCfg, usd_name: str) -> ArticulationCfg:
    """Use the USD with a passive gearbox-play hinge in series with each servo."""
    cfg = clone(cfg)
    cfg.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/{usd_name}"
    cfg.actuators["servos"] = replace(cfg.actuators["servos"], backlash_joint_template="passive_{}_backlash")
    cfg.init_state.joint_pos["passive_.*"] = 0.0
    return cfg


@configclass
class WalkingSceneCfg(DuckSceneCfg):
    robot = preset(
        default=MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        backlash=_with_backlash(MICRODUCK_CFG, "microduck_walk_backlash.usd").replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel = RewTerm(
        func=rewards.track_linear_velocity_heading, weight=4.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    track_ang_vel = RewTerm(
        func=rewards.track_angular_velocity_world, weight=4.0, params={"command_name": "base_velocity", "std": 0.35}
    )
    upright = RewTerm(func=rewards.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG})
    pose = RewTerm(
        func=rewards.joint_deviation_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {".*hip_yaw": 0.1, ".*hip_roll": 0.05, ".*hip_pitch": 0.15, ".*knee": 0.15, ".*ankle": 0.1},
            "std_walking": {".*hip_yaw": 0.3, ".*hip_roll": 0.05, ".*hip_pitch": 0.4, ".*knee": 0.4, ".*ankle": 0.25},
            "walking_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True),
            "standing_only": True,
        },
    )
    body_ang_vel = RewTerm(func=rewards.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=rewards.angular_momentum_l2, weight=-0.02)
    # Play hinges rest against their stops, so only servos incur the soft-limit penalty with backlash.
    dof_pos_limits = _backlash_preset(
        RewTerm(func=rewards.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(
        func=rewards.feet_air_time_windowed,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.3,
            "command_threshold": 0.01,
        },
    )
    foot_clearance = RewTerm(
        func=rewards.foot_clearance,
        weight=-2.0,
        params={
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    foot_swing_height = RewTerm(
        func=rewards.foot_swing_height,
        weight=-0.25 * (MICRODUCK_FOOT_TARGET_HEIGHT / 0.02) ** 2,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_BODY_CFG,
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )
    foot_slip = RewTerm(
        func=rewards.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    self_collisions = RewTerm(
        func=rewards.self_collision_cost, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("self_collision")}
    )
    head_pose_tracking = _backlash_preset(
        RewTerm(
            func=rewards.head_pose_tracking,
            weight=2.0,
            params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    head_pose_bias = _backlash_preset(
        RewTerm(
            func=rewards.head_pose_bias_penalty,
            weight=0.0,
            params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )


def _schedule(address: str, *stages: tuple[int, object]) -> CurrTerm:
    """Step ``address`` through ``(upstream_iteration, value)`` stages after its configured value."""
    steps = [(iteration * MICRODUCK_STEPS_PER_ITERATION, value) for iteration, value in stages]
    return CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": address,
            "modify_fn": staged_value,
            "modify_params": {
                "stages": [
                    (f"${{microduck_scale:{step},${{env.ngpus}},${{env.nnodes}}}}", value) for step, value in steps
                ]
            },
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (3000, -0.2), (4000, -0.4), (5000, -0.6), (6000, -0.8), (7000, -1.0)
    )
    head_pose_bias_weight = _schedule("rewards.head_pose_bias.weight", (600, -1.0), (1000, -2.0), (1500, -3.0))
    standing_envs = _schedule(
        "commands.base_velocity.rel_standing_envs", (500, 0.05), (750, 0.1), (1000, 0.15), (1500, 0.2), (2000, 0.25)
    )
    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2000, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (500, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1000, {axis: (-0.01, 0.01) for axis in "xyz"}),
        (1500, {axis: (-0.015, 0.015) for axis in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (500, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1000, {axis: (-0.01, 0.01) for axis in "xyz"}),
    )
