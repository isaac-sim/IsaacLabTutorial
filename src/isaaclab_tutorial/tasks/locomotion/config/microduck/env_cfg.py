# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teaching variants of the maintained Microduck flat locomotion task."""

from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.microduck import MICRODUCK_CFG, MICRODUCK_JOINT_NAMES

from isaaclab_tutorial.locomotion.assets import verify_scene_robot_asset
from isaaclab_tutorial.locomotion.effects import EffectSettings, apply_effects


@configclass
class ResetEventsCfg:
    """Reset physical state and controller targets together."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    asset_lock = EventTermCfg(func=verify_scene_robot_asset, mode="prestartup")


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


def _training_delays(cfg: NominalEnvCfg) -> None:
    """Randomize only the two declared delay clocks, preserving reset/buffer behavior of the maintained models."""
    cfg.scene.robot.actuators["servos"].max_delay = 6
    cfg.observations.policy.imu.params.update(min_lag=0, max_lag=2)
    cfg.observations.policy.joint_vel.delay_min_lag = 0
    cfg.observations.policy.joint_vel.delay_max_lag = 2


@configclass
class DelayEnvCfg(NominalEnvCfg):
    """Nominal mechanics with randomized command and IMU/joint-velocity delays."""

    def __post_init__(self):
        super().__post_init__()
        _training_delays(self)


@configclass
class BacklashEnvCfg(NominalEnvCfg):
    """Known series-play asset, with no other uncertainty enabled."""

    def __post_init__(self):
        super().__post_init__()
        # Constructors run before Hydra resolves presets. Select the nominal term objects first;
        # apply_effects supplies the explicitly selected backlash selectors and leaves all weights intact.
        for group in (self.observations.policy, self.observations.critic):
            for name in ("joint_pos", "joint_vel"):
                term = getattr(group, name)
                if isinstance(term, PresetCfg):
                    setattr(group, name, term.default)
        for name in ("dof_pos_limits", "head_pose_tracking", "head_pose_bias"):
            term = getattr(self.rewards, name)
            if isinstance(term, PresetCfg):
                setattr(self.rewards, name, term.default)
        apply_effects(self, EffectSettings(backlash=True))


@configclass
class CombinedEnvCfg(BacklashEnvCfg):
    """Backlash plus randomized delays: the fourth cell of the controlled two-factor comparison."""

    def __post_init__(self):
        super().__post_init__()
        _training_delays(self)
