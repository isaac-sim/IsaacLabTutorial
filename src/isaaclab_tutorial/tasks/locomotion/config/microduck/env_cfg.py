# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teaching variants of the maintained Microduck flat locomotion task."""

from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg

from isaaclab_assets.robots.microduck import MICRODUCK_CFG, MICRODUCK_JOINT_NAMES


@configclass
class ResetEventsCfg:
    """Reset physical state and controller targets together."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class NominalEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Flat PPO baseline with fixed head commands and explicit nominal hardware conditions."""

    def __post_init__(self):
        super().__post_init__()
        robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.actuators["servos"] = robot.actuators["servos"].replace(
            min_delay=0, max_delay=0, vin_range=(7.4, 7.4), vin_drop_gain_range=(0.0, 0.0)
        )
        self.scene.robot = robot
        self.events = ResetEventsCfg()
        self.curriculum = None
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.debug_vis = False
        self.observations.policy.enable_corruption = False
        self.observations.policy.imu.params.update(max_angle_deg=0.0, min_lag=0, max_lag=0)
        self.observations.policy.joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)},
        )
        self.compute_final_obs = True


@configclass
class InspectionEnvCfg(NominalEnvCfg):
    """Single supported duck for posing with the actual servo response."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot.spawn.fix_root_link = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
        self.episode_length_s = 3600.0
        self.terminations.fell_over = None


@configclass
class TransferEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Maintained hardware randomization, delays, and optional backlash asset preset."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.curriculum.head_pose_range = None
        self.curriculum.head_pose_bias_weight = None
        self.compute_final_obs = True
