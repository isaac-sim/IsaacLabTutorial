# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named, controlled training distributions, separate from fixed evaluation interventions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

PROFILE_TASKS = {
    "nominal": "MicroDuckTutorial-Velocity-Flat",
    "delay": "MicroDuckTutorial-Velocity-Flat-Delay",
    "backlash": "MicroDuckTutorial-Velocity-Flat-Backlash",
    "combined": "MicroDuckTutorial-Velocity-Flat-Combined",
}


def training_profile_metadata(profile: str) -> dict:
    """Describe the declared two-factor ablation; these ranges are teaching choices, not calibrated bounds."""
    if profile not in PROFILE_TASKS:
        raise ValueError(f"Choose a controlled training profile from {tuple(PROFILE_TASKS)}.")
    delayed = profile in ("delay", "combined")
    return {
        "name": profile,
        "task": PROFILE_TASKS[profile],
        "backlash": profile in ("backlash", "combined"),
        "command_delay_physics_steps": [0, 6 if delayed else 0],
        "imu_delay_control_samples": [0, 2 if delayed else 0],
        "joint_velocity_delay_control_samples": [0, 2 if delayed else 0],
        "command_delay_cadence": "uniform integer lag, redrawn each 5 ms physics step" if delayed else "disabled",
        "imu_delay_cadence": (
            "gyro/gravity lags: independent draws every 64 control samples with per-channel phase; reset begins at zero"
            if delayed
            else "disabled"
        ),
        "joint_velocity_delay_cadence": "uniform integer lag held until episode reset" if delayed else "disabled",
        "supply_voltage_v": 7.4,
        "battery_sag": False,
        "additive_sensor_noise": False,
        "encoder_bias_rad": 0.0,
        "other_physics_randomization": False,
        "curricula": False,
        "scope": "Delay × backlash ablation. Full hardware randomization is a separate experiment.",
    }


def validate_profile_checkpoint(checkpoint: Path, profile: str, env_cfg: dict, agent_cfg: dict) -> None:
    """Admit only a declared training task; preserve nominal reward experiments and collection budgets.

    Inference shape checks alone do not establish the training distribution. Full environment fingerprints
    additionally reject undeclared reset, command, terrain, physics or randomization changes. The shared
    fingerprint reader normalizes runtime scene paths, seed, environment count and renderer selection.
    """
    import yaml

    from .evaluation import resolve_checkpoint, validate_training_contract
    from .training import training_fingerprints

    if profile not in PROFILE_TASKS:
        raise ValueError(f"Unknown controlled training profile: {profile}.")
    resolve_checkpoint(checkpoint, task=PROFILE_TASKS[profile])
    validate_training_contract(checkpoint, env_cfg, agent_cfg)
    saved_env = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    saved_agent = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    actual = training_fingerprints(saved_env, saved_agent)
    expected = training_fingerprints(env_cfg, agent_cfg)
    differences = sorted(
        key
        for key in actual.keys() | expected.keys()
        if key.startswith("env.")
        and not (profile == "nominal" and key == "env.rewards")
        and actual.get(key) != expected.get(key)
    )
    if differences:
        raise ValueError(f"Checkpoint's declared training task differs at {', '.join(differences)}.")
