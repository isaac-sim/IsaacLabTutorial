# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck manager-based environments."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

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
