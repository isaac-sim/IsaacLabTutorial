# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline artifacts for pollen-robotics/microduck's existing feed-forward policy runtime."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Verified against duck-control/src/{model,obs}.rs at this immutable runtime revision.
DUCK_RUNTIME_REVISION = "da34a2212ae12eddfd73d0dfbf924f291aaf1dc9"
DUCK_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
DUCK_HOME_RAD = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530)
OBSERVATION_LAYOUT = {
    "imu": 6,
    "joint_pos": 14,
    "joint_vel": 14,
    "actions": 14,
    "velocity_commands": 3,
    "head_pose_commands": 4,
    "body_pose_commands": 6,
}


@contextmanager
def staged_bundle(destination: Path):
    """Publish a complete checked bundle only on success; preserve any existing bundle."""
    if destination.exists():
        raise FileExistsError(f"Bundle already exists at {destination}; choose a fresh --output directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-check-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "exported"
        stage.mkdir()
        yield stage
        stage.rename(destination)


def inline_onnx_weights(path: Path) -> None:
    """Make the policy independently loadable when the exporter writes an external weight file."""
    import onnx

    model = onnx.load_model(path, load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, path, save_as_external_data=False)


def check_duck_contract(contract: dict) -> None:
    """Reject policies whose physical interpretation differs from the documented Duck bundle."""
    if tuple(contract["joint_names"]) != DUCK_JOINTS:
        raise ValueError("Duck policy joints must follow servo order with the mouth excluded.")
    for field, expected in (("home_rad", DUCK_HOME_RAD), ("action_scale", np.ones(14))):
        actual = np.asarray(contract[field])
        if actual.shape != (14,) or not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError(f"The standard Duck bundle requires the verified {field}.")
    if not np.isclose(contract["control_dt_s"], 0.02, atol=1e-9, rtol=0):
        raise ValueError("The standard Duck runtime consumes this policy at 50 Hz.")
    layout = contract["observation_layout"]
    if list(layout.items()) != list(OBSERVATION_LAYOUT.items()):
        raise ValueError("The ordered observation layout must match the 61-value Duck contract.")
    if contract["normalization"] != "embedded":
        raise ValueError("Observation normalization must be embedded in the exported actor.")


def rehearsal_trace(observations) -> list[dict]:
    """Serialize actual raw policy inputs for Duck's no-actuation policy-rehearsal example."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 61 or not len(observations):
        raise ValueError("Rehearsal needs a nonempty sequence of 61-value observations.")
    if not np.isfinite(observations).all():
        raise ValueError("Rehearsal observations must be finite.")
    return [{"obs": row.tolist(), "reset": i == 0} for i, row in enumerate(observations)]


def build_manifest(receipt: dict, *, onnx_sha256: str) -> dict:
    """Describe a parity-checked feed-forward gait using the standard schema-2 manifest."""
    check_duck_contract(receipt["contract"])
    parity = receipt.get("onnx_parity", {})
    if receipt.get("valid") is not True or not parity.get("checked") or parity.get("samples", 0) <= 0:
        raise ValueError("A valid simulated trial and executed ONNX parity check are required.")
    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
        "name": f"tutorial-flat-{receipt.get('algorithm', 'ppo')}",
        "description": "Microduck policy export; simulation response and inference parity are assessed separately.",
        "kind": "perpetual",
        "slot": "walk",
        "entry_pose": "standing",
        "action_scale": 1.0,
        "command": {
            "encoding": "constant",
            "idle": [0, 0, 0],
            "twist": "forward m/s, left m/s, yaw rad/s",
            "head": "four zero commands; nonzero head commands were not trained",
            "body": "six zero commands; nonzero body commands were not trained",
        },
        "training": {
            "task_id": "MicroDuckTutorial-Velocity-Flat",
            "checkpoint": receipt["checkpoint"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
        },
        "eval": {
            "seed": receipt["seed"],
            "simulation": receipt["metrics"],
            "simulation_qualified": receipt.get("qualification", {}).get("passed"),
            "response_qualification": receipt.get("qualification"),
            "start_condition": receipt.get("start_condition"),
            "conditions": receipt.get("conditions"),
            "effect_settings": receipt.get("effect_settings"),
            "asset_url": receipt.get("asset_url"),
            "onnx_parity": parity,
            "onnx_sha256": onnx_sha256,
            "hardware_validated": False,
        },
        "tutorial_contract": receipt["contract"],
        "runtime_source": f"https://github.com/pollen-robotics/microduck/tree/{DUCK_RUNTIME_REVISION}",
    }
