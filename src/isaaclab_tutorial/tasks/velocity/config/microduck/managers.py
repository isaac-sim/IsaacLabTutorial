# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck observations, reset events, and episode termination conditions."""

import math

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.utils import preset

from isaaclab_tutorial.tasks.velocity.mdp import events, observations, terminations

from .robot import MICRODUCK_JOINT_NAMES

MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=["trunk_base"])
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES], preserve_order=True
)


def _backlash_preset(term, func=None, **params):
    """Use ``term`` as is, or with ``func`` and ``params`` for the backlash robot under ``presets=backlash``."""
    backlash = term.replace(params={**term.params, **params})
    if func is not None:
        backlash.func = func
    return preset(default=term, backlash=backlash)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract."""

        # Angular velocity [rad/s], then projected gravity.
        imu = ObsTerm(
            func=observations.misaligned_imu,
            params={"max_angle_deg": 6.0, "min_lag": 0, "max_lag": 1, "update_period": 64},
            noise=observations.ImuNoiseCfg(ang_vel=0.03, gravity=0.01),
        )
        # With backlash, encoders measure servo plus play angle.
        joint_pos = _backlash_preset(
            ObsTerm(
                func=observations.joint_pos_rel_biased,
                params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
                noise=Unoise(n_min=-0.001, n_max=0.001),
            ),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        joint_vel = _backlash_preset(
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": _SERVO_JOINT_CFG},
                noise=Unoise(n_min=-0.25, n_max=0.25),
                delay_min_lag=1,
                delay_max_lag=1,
            ),
            func=observations.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (76 values)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = _backlash_preset(
            ObsTerm(func=observations.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        joint_vel = _backlash_preset(
            ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG}),
            func=observations.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_height = ObsTerm(func=observations.foot_height_safe, params={"asset_cfg": _FOOT_BODY_CFG})
        foot_air_time = ObsTerm(func=observations.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=observations.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=observations.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    encoder_bias = EventTerm(func=events.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
            ),
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    randomize_joint_friction = EventTerm(
        func=events.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
    )
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    nan_state = DoneTerm(func=terminations.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})
