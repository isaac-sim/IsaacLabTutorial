# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent tracking metrics for one attempted episode, including early failures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


class _TrainingConfigLoader(yaml.SafeLoader):
    """Read training's tuple/slice metadata as sequences, without constructing Python objects."""


for _tag in ("tag:yaml.org,2002:python/tuple", "tag:yaml.org,2002:python/object/apply:builtins.slice"):
    _TrainingConfigLoader.add_constructor(_tag, _TrainingConfigLoader.construct_sequence)


def resolve_checkpoint(
    checkpoint: Path,
    *,
    task: str = "MicroDuckTutorial-Velocity-Flat",
    log_root: Path = Path("logs/rsl_rl/microduck_tutorial"),
) -> Path:
    """Use maintained run discovery and reject checkpoints whose training task does not match."""
    if checkpoint == Path("latest"):
        from isaaclab_rl.entrypoints.common import resolve_checkpoint_selector

        checkpoint = Path(
            resolve_checkpoint_selector(
                str(log_root), "latest", library="rsl_rl", task=task, checkpoint_pattern=r"model_.*\.pt"
            )
        )
    if not checkpoint.is_file():
        raise ValueError("A saved checkpoint is required; run the lesson-02 training command first.")
    manifest = checkpoint.parent / "run.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("task") != task:
        raise ValueError("The checkpoint's recorded training task does not match this evaluation task.")
    return checkpoint


def validate_training_contract(checkpoint: Path, env_cfg: dict, agent_cfg: dict) -> None:
    """Require saved inference semantics to match the selected, declared training configuration.

    Training reward changes are allowed: the reward workshop deliberately trains new objectives.
    Shape-compatible observation/action changes require a separate matching evaluator.
    """
    params = checkpoint.parent / "params"
    try:
        saved_env = yaml.load((params / "env.yaml").read_text(), Loader=_TrainingConfigLoader)
        saved_agent = yaml.load((params / "agent.yaml").read_text(), Loader=_TrainingConfigLoader)
    except FileNotFoundError as error:
        raise ValueError("Checkpoint training contract is missing; retain params/env.yaml and agent.yaml.") from error
    # Normalize callable references and tuple tags using the same serializer as training.
    expected_env = yaml.load(yaml.dump(env_cfg, sort_keys=False), Loader=_TrainingConfigLoader)
    expected_agent = yaml.load(yaml.dump(agent_cfg, sort_keys=False), Loader=_TrainingConfigLoader)
    for path in (
        ("actions",),
        ("observations", "policy"),
        ("scene", "robot", "init_state"),
        ("scene", "robot", "spawn", "usd_path"),
        ("scene", "robot", "actuators"),
        ("sim", "dt"),
        ("decimation",),
    ):
        actual, expected = saved_env, expected_env
        for part in path:
            actual, expected = actual[part], expected[part]
        order_changed = path == ("observations", "policy") and list(actual) != list(expected)
        if actual != expected or order_changed:
            raise ValueError(f"Checkpoint training contract differs at env.{'.'.join(path)}; use a matching evaluator.")
    for key in ("actor", "critic", "obs_groups", "clip_actions"):
        if saved_agent.get(key) != expected_agent.get(key):
            raise ValueError(f"Checkpoint training contract differs at agent.{key}; use a matching evaluator.")


def summarize_episode(rows: list[dict], control_dt: float, horizon_s: float) -> dict:
    """Report pre-fall error together with duration, failure, and attempted-horizon coverage."""
    if not rows or control_dt <= 0 or horizon_s <= 0:
        raise ValueError("An attempted episode needs samples and positive time intervals.")
    if any(row.get("termination_reason") == "nan_state" for row in rows):
        raise ValueError("A numerical termination invalidates the trial; it is not an ordinary fall.")
    command = np.asarray([row["command"] for row in rows], dtype=float)
    measured = np.asarray([row["measured"] for row in rows], dtype=float)
    if command.shape != (len(rows), 3) or measured.shape != command.shape:
        raise ValueError("Command and measured velocity must contain vx, vy, and yaw rate for each sample.")
    if not np.isfinite(command).all() or not np.isfinite(measured).all():
        raise ValueError("Non-finite signals invalidate an evaluation rather than counting as good tracking.")
    duration = len(rows) * control_dt
    return {
        "fell": any(
            row.get("termination_reason") == "fell_over"
            or (row.get("termination_reason") is None and row["terminated"])
            for row in rows
        ),
        "alive_time_s": duration,
        "coverage": min(1.0, duration / horizon_s),
        "tracking_rmse": np.sqrt(np.mean((command - measured) ** 2, axis=0)).tolist(),
        "tracking_units": ["m/s", "m/s", "rad/s"],
        "tracking_scope": "available pre-fall samples; interpret together with failure and coverage",
    }
