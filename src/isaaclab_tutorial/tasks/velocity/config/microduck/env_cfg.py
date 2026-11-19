# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck manager-based environments."""

import math

from isaaclab.envs.mdp import modify_term_cfg
from isaaclab.managers import CurriculumTermCfg as Curriculum

from isaaclab_tutorial.tasks.velocity.mdp.curriculums import staged_value

from .walking import CurriculumCfg, RewardsCfg, WalkingSceneCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import action_rate_l2
from isaaclab.managers import RewardTermCfg as Reward
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab_tutorial.tasks.velocity.mdp import rewards

from .managers import EventsCfg, ObservationsCfg, TerminationsCfg

from .control import ActionsCfg, CommandsCfg
from .scene import DuckSceneCfg


@configclass
class NoTermsCfg:
    """No reward or curriculum terms until the corresponding lesson adds them."""


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """The runnable scene, control loop, and reset rules, before adding a reward."""

    ngpus: int | None = None
    nnodes: int | None = None
    sim = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=96,
                nconmax=16,
                iterations=100,
                ls_iterations=50,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2500000),
            default_shape_cfg=NewtonShapeCfg(margin=0.0),
            num_substeps=1,
        )
    )
    scene = DuckSceneCfg(num_envs=4096, env_spacing=2.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards = NoTermsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    curriculum = NoTermsCfg()

    def __post_init__(self):
        self.configure_simulation()
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.rel_standing_envs = 1.0
        cmd.rel_forward_envs = 0.0
        cmd.rel_turn_in_place_envs = 0.0
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.events.push_robot = None

    def configure_simulation(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.self_collision.update_period = self.sim.dt
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))

    def play_mode(self):
        super().play_mode()
        self.events.push_robot = None


@configclass
class InspectEnvCfg(BaseEnvCfg):
    """Hold the base for the first joint actuation exercise, not for learning balance."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.fix_root_link = True
        self.events.encoder_bias = None
        self.events.randomize_com = None
        self.events.randomize_head_com = None
        self.events.randomize_joint_friction = None
        self.events.randomize_armature = None


@configclass
class BalanceRewardsCfg:
    # A spirit level: a vertical trunk earns more than a tilted trunk.
    upright = Reward(
        func=rewards.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=["trunk_base"])},
    )
    # A speedometer near zero: discourage horizontal motion at the current location.
    still_xy = Reward(
        func=rewards.track_linear_velocity_heading, weight=1.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    still_yaw = Reward(
        func=rewards.track_angular_velocity_world, weight=1.0, params={"command_name": "base_velocity", "std": 0.35}
    )
    # A movement budget: discourage abrupt target changes, not all movement.
    action_rate = Reward(func=action_rate_l2, weight=-0.1)
    joint_limits = Reward(func=rewards.joint_pos_limits_finite, weight=-1.0)


@configclass
class BalanceEnvCfg(BaseEnvCfg):
    """Learn balance with a free base; zero commands do not constrain physics."""

    rewards = BalanceRewardsCfg()


@configclass
class BalanceHeadRewardsCfg(BalanceRewardsCfg):
    # Four centering springs: return each neck/head joint to its neutral target.
    head_pose = Reward(
        func=rewards.head_pose_tracking,
        weight=2.0,
        params={
            "command_name": "head_pose",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll"], preserve_order=True
            ),
        },
    )


@configclass
class BalanceHeadEnvCfg(BalanceEnvCfg):
    """Keep the balance objective and add a neutral neck/head pose."""

    rewards = BalanceHeadRewardsCfg()


@configclass
class TrackingRewardsCfg(BalanceHeadRewardsCfg):
    still_xy = None
    still_yaw = None

    # Move the bull's-eye: these terms now track sampled velocity commands.
    track_lin_vel = Reward(
        func=rewards.track_linear_velocity_heading, weight=4.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    track_ang_vel = Reward(
        func=rewards.track_angular_velocity_world, weight=4.0, params={"command_name": "base_velocity", "std": 0.35}
    )
    # Give each moving foot a useful interval in the air.
    air_time = Reward(
        func=rewards.feet_air_time_windowed,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["ankle_left", "ankle_right"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.3,
            "command_threshold": 0.01,
        },
    )
    # A planted shoe: ankle-frame speed during contact is a proxy for sliding.
    foot_slip = Reward(
        func=rewards.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["ankle_left", "ankle_right"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True),
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )


@configclass
class TrackingCurriculumCfg:
    # Increase the price of abrupt target changes as practice progresses.
    action_rate_weight = Curriculum(
        func=modify_term_cfg,
        params={
            "address": "rewards.action_rate.weight",
            "modify_fn": staged_value,
            "modify_params": {
                "stages": [
                    (3000 * 24, -0.2),
                    (4000 * 24, -0.4),
                    (5000 * 24, -0.6),
                    (6000 * 24, -0.8),
                    (7000 * 24, -1.0),
                ]
            },
        },
    )


@configclass
class TrackingEnvCfg(BaseEnvCfg):
    rewards = TrackingRewardsCfg()
    curriculum = TrackingCurriculumCfg()

    def __post_init__(self):
        # Keep sampled commands; BaseEnvCfg.__post_init__ makes them all zero.
        self.configure_simulation()


@configclass
class WalkEnvCfg(TrackingEnvCfg):
    scene = WalkingSceneCfg(num_envs=4096, env_spacing=2.0)
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()
