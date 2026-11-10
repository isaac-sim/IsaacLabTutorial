# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit teaching interventions, applied before constructing a fresh Microduck scene."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .deployment import DUCK_JOINTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


@dataclass(frozen=True)
class EffectSettings:
    """Bounded simulation experiments, not measured hardware tolerances or on-robot settings."""

    command_delay_ms: int = 0
    observation_delay_ms: int = 0
    noise_scale: float = 0.0
    voltage: float = 7.4
    encoder_bias_rad: float = 0.0
    backlash: bool = False

    def __post_init__(self) -> None:
        for name, quantum, maximum in (("command_delay_ms", 5, 100), ("observation_delay_ms", 20, 200)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum or value % quantum:
                raise ValueError(f"{name} must be an integer multiple of {quantum}, between zero and {maximum} ms.")
        for name, low, high in (("noise_scale", 0.0, 3.0), ("voltage", 6.0, 8.4), ("encoder_bias_rad", -0.05, 0.05)):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and within [{low}, {high}].")
        if type(self.backlash) is not bool:
            raise ValueError("backlash must be a boolean selecting the real passive-hinge asset.")

    def manifest(self) -> dict:
        """Expose units, cadence, and scene-rebuild semantics alongside the requested numerical settings."""
        return {
            "schema_version": 1,
            "settings": asdict(self),
            "command_delay_physics_steps": self.command_delay_ms // 5,
            "observation_delay_samples": self.observation_delay_ms // 20,
            "noise_bounds": {
                "angular_velocity_rad_s": 0.03 * self.noise_scale,
                "projected_gravity": 0.01 * self.noise_scale,
                "joint_position_rad": 0.001 * self.noise_scale,
                "joint_velocity_rad_s": 0.25 * self.noise_scale,
            },
            "noise_distribution": "independent additive uniform components within plus/minus each bound",
            "noise_cadence": "each actor observation computation; critic remains uncorrupted",
            "encoder_bias_scope": "same signed bias for all 14 encoders, fixed throughout this simulated robot trial",
            "voltage_scope": "fixed BAM supply voltage in volts; battery sag disabled",
            "backlash_scope": "14 passive series hinges, each bounded by asset stops at plus/minus 1 degree",
            "apply_boundary": "fresh simulator process and scene; reset histories before initial observation",
            "delay_startup": (
                "maintained buffers have reset/warmup behavior; requested lag is not uniform sample age at startup"
            ),
        }


def apply_effects(cfg: ManagerBasedRLEnvCfg, settings: EffectSettings) -> None:
    """Overlay known effects on the validated nominal task without enabling unrelated transfer randomization."""
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.utils.noise import UniformNoiseCfg

    from isaaclab_tasks.contrib.microduck import mdp

    from isaaclab_assets.robots.microduck import MICRODUCK_BACKLASH_CFG, MICRODUCK_HEAD_JOINT_NAMES

    if settings.backlash:
        cfg.scene.robot = MICRODUCK_BACKLASH_CFG.replace(prim_path=cfg.scene.robot.prim_path)
        servos = SceneEntityCfg("robot", joint_names=list(DUCK_JOINTS), preserve_order=True)
        play = SceneEntityCfg(
            "robot", joint_names=[f"passive_{name}_backlash" for name in DUCK_JOINTS], preserve_order=True
        )
        for group in (cfg.observations.policy, cfg.observations.critic):
            group.joint_pos.params["backlash_cfg"] = play
            group.joint_vel.func = mdp.joint_vel_rel_backlash
            group.joint_vel.params["backlash_cfg"] = play
        cfg.rewards.dof_pos_limits.params["asset_cfg"] = servos
        head_play = SceneEntityCfg(
            "robot",
            joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_HEAD_JOINT_NAMES],
            preserve_order=True,
        )
        for term in (cfg.rewards.head_pose_tracking, cfg.rewards.head_pose_bias):
            term.params["backlash_cfg"] = head_play
    actuator = cfg.scene.robot.actuators["servos"]
    actuator.min_delay = actuator.max_delay = settings.command_delay_ms // 5
    actuator.vin_range = (settings.voltage, settings.voltage)
    actuator.vin_drop_gain_range = (0.0, 0.0)
    lag = settings.observation_delay_ms // 20
    cfg.observations.policy.imu.params.update(min_lag=lag, max_lag=lag)
    cfg.observations.policy.joint_vel.delay_min_lag = lag
    cfg.observations.policy.joint_vel.delay_max_lag = lag
    if settings.noise_scale:
        bounds = settings.manifest()["noise_bounds"]
        cfg.observations.policy.enable_corruption = True
        cfg.observations.policy.imu.noise = mdp.ImuNoiseCfg(
            ang_vel=bounds["angular_velocity_rad_s"], gravity=bounds["projected_gravity"]
        )
        for term, name in (
            (cfg.observations.policy.joint_pos, "joint_position_rad"),
            (cfg.observations.policy.joint_vel, "joint_velocity_rad_s"),
        ):
            term.noise = UniformNoiseCfg(n_min=-bounds[name], n_max=bounds[name])
    if settings.encoder_bias_rad:
        cfg.events.encoder_bias = EventTermCfg(
            func=mdp.randomize_encoder_bias,
            mode="startup",
            params={"bias_range": (settings.encoder_bias_rad, settings.encoder_bias_rad)},
        )


def mechanical_state(state: dict) -> dict:
    """Map only the known14-servo/28-DoF variants to the same physical output-angle convention.

    Raw passive and servo coordinates remain in the receipt. A series output coordinate is the sum
    of its corresponding servo and passive hinge; this mapping does not remove mechanical play.
    """
    names = state["joint_names"]
    passive = [f"passive_{name}_backlash" for name in DUCK_JOINTS]
    allowed = set(DUCK_JOINTS) | (set(passive) if len(names) == 28 else set())
    if len(set(names)) != len(names) or set(names) != allowed:
        raise ValueError("Mechanical comparison requires the known14-servo or28-joint Microduck names.")
    q = np.asarray(state["joint_pos_rad"], dtype=float)
    dq = np.asarray(state["joint_vel_rad_s"], dtype=float)
    if q.shape != (len(names),) or dq.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError("Mechanical joint position and velocity vectors must be finite and match named coordinates.")
    indices = [names.index(name) for name in DUCK_JOINTS]
    output_q, output_dq = q[indices].copy(), dq[indices].copy()
    if len(names) == 28:
        play_indices = [names.index(name) for name in passive]
        output_q += q[play_indices]
        output_dq += dq[play_indices]
    return {
        **{key: value for key, value in state.items() if key.startswith("root_")},
        "joint_names": list(DUCK_JOINTS),
        "joint_pos_rad": output_q.tolist(),
        "joint_vel_rad_s": output_dq.tolist(),
    }
