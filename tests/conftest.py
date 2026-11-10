# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small synthetic report inputs; no captured training or evaluation data is required."""

import pytest


@pytest.fixture
def qualification_reports():
    from isaaclab_tutorial.locomotion.protocols import sample_start_condition

    reports = []
    for delay in (0, 40):
        trials = []
        models = [{"training_seed": seed, "sha256": f"{seed + 1:064x}"} for seed in (0, 42)]
        for model in models:
            for seed in (0, 1):
                trials.append(
                    {
                        "training_seed": model["training_seed"],
                        "checkpoint_sha256": model["sha256"],
                        "seed": seed,
                        "valid": True,
                        "interactive": False,
                        "protocol_id": "flat-velocity-v1",
                        "control_dt_s": 0.02,
                        "horizon_s": 20.0,
                        "conditions": {
                            "start_protocol": "balance-starts-v1",
                            "observation_delay_ms": delay,
                            "command_delay_ms": 0,
                        },
                        "start_condition": sample_start_condition("balance-starts-v1", seed),
                        "initial_state": {
                            "root_pos_w": [0, 0, 0.125],
                            "root_quat_xyzw": [0, 0, 0, 1],
                            "root_linear_velocity_w": [0, 0, 0],
                            "root_angular_velocity_w": [0, 0, 0],
                            "joint_pos_rad": [0] * 14,
                            "joint_vel_rad_s": [0] * 14,
                        },
                        "metrics": {
                            "fell": bool(delay),
                            "coverage": 0.09 if delay else 1.0,
                            "alive_time_s": 1.8 if delay else 20.0,
                            "tracking_rmse": [0.01, 0.02, 0.03],
                        },
                        "qualification": {
                            "passed": not delay,
                            "criteria_sha256": "synthetic-criteria",
                            "segments": {},
                            "reasons": ["fell"] if delay else [],
                        },
                        "receipt_path": "/missing/synthetic-receipt.json",
                        "receipt_sha256": "0" * 64,
                    }
                )
        reports.append(
            {
                "schema_version": 1,
                "complete": True,
                "expected_trials": 4,
                "errors": [],
                "trials": trials,
                "configuration": {
                    "models": models,
                    "seeds": [0, 1],
                    "start_protocol": "balance-starts-v1",
                    "observation_delay_ms": delay,
                    "command_delay_ms": 0,
                },
            }
        )
    return reports


@pytest.fixture
def training_report():
    from isaaclab_tutorial.locomotion.training import PPOBudget

    runs = []
    for count in (128, 256):
        budget = PPOBudget(count, 24, 50, 5, 4).as_dict()
        runs.append(
            {
                "name": f"synthetic-{count}",
                "seed": 42,
                "checkpoint": None,
                "source_sha256": {},
                "config_fingerprints": {},
                "requested_budget": budget,
                "logged_budget": budget,
                "recorded_training_time_s": 10.0,
                "cumulative_training_time_s": [(i + 1) / 5 for i in range(50)],
                "metrics": {
                    "Train/mean_reward": {"label": "Return", "unit": "reward", "points": [[0, 1.0], [49, 2.0]]},
                    "Metrics/base_velocity/error_vel_xy": {
                        "label": "Velocity error",
                        "unit": "m/s",
                        "points": [[0, 0.3], [49, 0.1]],
                    },
                },
            }
        )
    runs[0]["evaluation"] = {
        "valid": False,
        "protocol_id": "flat-velocity-v1",
        "seed": 0,
        "metrics": {"invalid_reason": "nan_state"},
    }
    return {"schema_version": 1, "runs": runs}
