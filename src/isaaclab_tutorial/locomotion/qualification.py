# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate a single-condition qualification matrix without dropping failed attempts."""

from __future__ import annotations

import numpy as np

from .comparison import assert_matched_start


def aggregate_trials(trials: list[dict]) -> dict:
    """Report seed-level rates; error averages are explicitly conditional on complete valid trials."""
    if not trials:
        raise ValueError("Qualification needs attempted trials.")
    groups = {}
    starts = {}
    identities = set()
    model_seeds = {}
    physical_starts = []
    criterion_hashes = set()
    for trial in trials:
        seed = trial["training_seed"]
        identity = (trial["checkpoint_sha256"], trial["seed"])
        if identity in identities:
            raise ValueError("A duplicate checkpoint/evaluation-seed attempt cannot be counted twice.")
        identities.add(identity)
        if trial["conditions"] != trials[0]["conditions"]:
            raise ValueError("A qualification matrix must use one shared hardware condition.")
        if any(
            trial["start_condition"][field] != trials[0]["start_condition"][field]
            for field in ("name", "definition_sha256")
        ):
            raise ValueError("A matrix must use one shared start protocol definition.")
        if trial.get("interactive") or trial["horizon_s"] != 20.0 or trial["protocol_id"] != "flat-velocity-v1":
            raise ValueError("Qualification requires uninterrupted headless 20-second fixed-protocol receipts.")
        previous = model_seeds.setdefault(trial["checkpoint_sha256"], seed)
        if previous != seed:
            raise ValueError("Duplicate policy weights cannot count as independent training seeds.")
        group = groups.setdefault(str(seed), [])
        if group and group[0]["checkpoint_sha256"] != trial["checkpoint_sha256"]:
            raise ValueError("Choose one checkpoint per training seed.")
        group.append(trial)
        if trial["seed"] in starts:
            assert_matched_start(starts[trial["seed"]], trial)
        else:
            starts[trial["seed"]] = trial
        state = trial["initial_state"]
        quaternion = np.asarray(state["root_quat_xyzw"], dtype=float)
        if quaternion[-1] < 0:
            quaternion = -quaternion
        vector = np.concatenate(
            (
                state["root_pos_w"],
                quaternion,
                state["root_linear_velocity_w"],
                state["root_angular_velocity_w"],
                state["joint_pos_rad"],
                state["joint_vel_rad_s"],
            )
        )
        if not np.isfinite(vector).all():
            raise ValueError("Initial physical states must be finite even for later-invalid trials.")
        if not any(np.allclose(vector, existing, atol=1e-5, rtol=0) for existing in physical_starts):
            physical_starts.append(vector)
        if trial["valid"]:
            if trial["qualification"]["passed"] and (
                trial["metrics"]["fell"] or trial["metrics"]["coverage"] < 1.0 - 1e-8
            ):
                raise ValueError("A qualified response cannot contain a fall or incomplete coverage.")
            criterion_hashes.add(trial["qualification"]["criteria_sha256"])
    if len(criterion_hashes) > 1:
        raise ValueError("Qualification criteria changed within this matrix; do not pool different definitions.")
    per_seed = {}
    for seed, records in sorted(groups.items(), key=lambda item: int(item[0])):
        valid = [record for record in records if record["valid"]]
        full = [
            record for record in valid if not record["metrics"]["fell"] and record["metrics"]["coverage"] >= 1.0 - 1e-8
        ]
        per_seed[seed] = {
            "checkpoint_sha256": records[0]["checkpoint_sha256"],
            "attempts": len(records),
            "invalid_attempts": len(records) - len(valid),
            "full_protocol_trials": len(full),
            "survival_rate": len(full) / len(records),
            "qualified_rate": sum(record["valid"] and record["qualification"]["passed"] for record in records)
            / len(records),
            "full_trial_tracking_rmse_mean": np.mean(
                [record["metrics"]["tracking_rmse"] for record in full], axis=0
            ).tolist()
            if full
            else None,
            "tracking_error_scope": (
                "mean of per-trial RMSE over complete valid trials only; inspect failures and counts together"
            ),
        }

    def variation(field):
        values = [record[field] for record in per_seed.values()]
        return {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
            "unit": "one rate per training seed",
        }

    seed_sets = [{record["seed"] for record in group} for group in groups.values()]
    return {
        "attempts": len(trials),
        "invalid_attempts": sum(not record["valid"] for record in trials),
        "training_seeds": len(per_seed),
        "evaluation_seeds": sorted(starts),
        "unique_physical_starts": len(physical_starts),
        "per_training_seed": per_seed,
        "qualified_rate_across_training_seeds": variation("qualified_rate"),
        "survival_rate_across_training_seeds": variation("survival_rate"),
        "shared_evaluation_seeds": sorted(set.intersection(*seed_sets)),
        "paired_initial_states_checked": len(seed_sets) > 1 and all(seeds == seed_sets[0] for seeds in seed_sets),
    }
