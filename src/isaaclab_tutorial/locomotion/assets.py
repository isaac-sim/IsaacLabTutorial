# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Byte-locked Microduck robot inputs, independent of mutable CDN URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Each reviewed USD contains its physics and geometry in one non-anonymous layer.
# These are authored MassAPI sums, not runtime aggregates or measured hardware weights.
LOCKED_ROBOTS = {
    "microduck_walk.usd": {
        "sha256": "288c6696ffa0f83df89512497860801f543d72874b835bb716db342df60385d4",
        "bytes": 7479847,
        "authored_mass_sum_kg": 0.7372431745752692,
        "authored_mass_entries": 18,
    },
    "microduck_walk_backlash.usd": {
        "sha256": "0fe30a47867d8f33761f3fc5537aa6983a45cb26cc536eff9c3d2a030066a97b",
        "bytes": 7486953,
        "authored_mass_sum_kg": 0.7372571745752339,
        "authored_mass_entries": 32,
    },
}


def verify_robot_asset_file(path: Path) -> dict:
    """Require the exact reviewed robot bytes; a name or URL alone does not freeze an asset."""
    if path.name not in LOCKED_ROBOTS:
        raise ValueError("Choose a locked Microduck robot variant; an unknown asset needs its own audited contract.")
    expected = LOCKED_ROBOTS[path.name]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected["sha256"]:
        raise ValueError(f"Robot SHA256 changed for {path.name}; preserve the old experiment instead of mixing assets.")
    return {"name": path.name, **expected, "composed_nonanonymous_usd_layers": 1}


def verify_robot_asset(url: str) -> dict:
    """Resolve and check the actual cached USD used by the pinned asset importer."""
    from isaaclab.utils.assets import retrieve_file_path

    path = Path(retrieve_file_path(url))
    return {"url": url, **verify_robot_asset_file(path)}


def verify_scene_robot_asset(env, env_ids) -> None:
    """Check the robot before scene construction without altering physics or consuming learner RNG."""
    env.tutorial_asset_provenance = verify_robot_asset(env.cfg.scene.robot.spawn.usd_path)
