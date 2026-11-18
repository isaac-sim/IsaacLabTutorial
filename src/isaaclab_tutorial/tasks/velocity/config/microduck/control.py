# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Action and command configurations assembled before introducing upstream imports."""

# BEGIN actions
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.velocity.config.microduck.reference import mdp

from isaaclab_tutorial.tasks.velocity.config.microduck.robot import MICRODUCK_JOINT_NAMES


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True, scale=1.0, use_default_offset=True
    )


# END actions


# BEGIN commands
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.15,
        debug_vis=True,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0), ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    )
    # Retained in the deployed observation vector, although body-pose reward is disabled.
    body_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
    )


# END commands
