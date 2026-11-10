# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Byte-locked Microduck robot inputs, independent of mutable CDN URLs."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

# Audited with USD composition: each robot has one non-anonymous layer, containing its physics/geometry.
# Masses below are authored USD MassAPI sums, not a runtime aggregate or measured hardware weight.
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


def collect_robot_assets(destination: Path | None = None) -> dict:
    """Check both models, optionally archive their exact bytes, and record the small authored-mass difference."""
    from pxr import Usd

    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_assets.robots.microduck import MICRODUCK_BACKLASH_CFG, MICRODUCK_CFG

    records = []
    for cfg in (MICRODUCK_CFG, MICRODUCK_BACKLASH_CFG):
        url = cfg.spawn.usd_path
        path = Path(retrieve_file_path(url))
        record = {"url": url, **verify_robot_asset_file(path)}
        stage = Usd.Stage.Open(str(path))
        dependencies = {Path(layer.realPath).resolve() for layer in stage.GetUsedLayers() if not layer.anonymous}
        if dependencies != {path.resolve()}:
            raise ValueError(
                "Robot composition has external USD layers; extend the byte lock before publishing results."
            )
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / path.name
            shutil.copyfile(path, target)
            verify_robot_asset_file(target)
        records.append(record)
    return {
        "schema_version": 1,
        "robots": records,
        "authored_mass_difference_kg": records[1]["authored_mass_sum_kg"] - records[0]["authored_mass_sum_kg"],
        "scope": "Self-contained robot USD inputs; simulation configuration overrides are recorded separately.",
        "backlash_note": (
            "The 14 extra passive bodies add 14 mg of authored mass. This is an asset-variant intervention."
        ),
    }
