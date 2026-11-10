# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit the complete robustness study before extracting standing and requested-target diagnostics."""

import argparse
import json
from pathlib import Path

from summarize_robustness import CASES, audit_trials, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = audit_trials(args.trial)
    attempts = []
    for path in args.trial:
        metadata = json.loads((path / "trial.json").read_text())
        profile, seed = metadata["profile"]["name"], metadata["training_seed"]
        for case in CASES:
            report_path = path / "evaluations" / case / "report.json"
            report = json.loads(report_path.read_text())
            for trial in report["trials"]:
                q = trial.get("qualification") or {}
                standing = q.get("segments", {}).get("stand", {})
                attempts.append(
                    {
                        "profile": profile,
                        "condition": case,
                        "training_seed": seed,
                        "evaluation_seed": trial["seed"],
                        "receipt_sha256": trial["receipt_sha256"],
                        "valid": trial["valid"],
                        "metrics": trial["metrics"],
                        "qualified": q.get("passed"),
                        "standing_coverage": standing.get("coverage", 0.0),
                        "standing_planar_drift_m": q.get("standing_planar_drift_m"),
                        "standing_yaw_drift_rad": q.get("standing_yaw_drift_rad"),
                        "requested_target_rate_rms_rad_s": q.get("requested_target_rate_rms_rad_s"),
                    }
                )
    data = {
        "schema_version": 1,
        "scope": "All 480 attempts from the audited four-profile, three-training-seed study; conditional diagnostics.",
        "source_revision": study["source_revision"],
        "collector_sha256": digest(Path(__file__)),
        "auditor_sha256": digest(Path(__file__).with_name("summarize_robustness.py")),
        "training_seeds": study["training_seeds"],
        "evaluation_seeds": study["evaluation_seeds"],
        "budget": study["budget"],
        "runs": study["runs"],
        "all_raw_motion_metrics_and_qualification_recomputed_by_auditor": True,
        "matched_actual_starts_checked_across_all_profiles_and_conditions": True,
        "standing_scope": (
            "First-to-last pre-step sample before 4 s (0.00 through 3.98 s at 50 Hz). "
            "Plot only complete standing sampling windows; retain incomplete prefixes in data."
        ),
        "target_rate_scope": (
            "sqrt(mean((diff(requested_target_rad) / 0.02 s)^2)) over all 14 joints and adjacent pre-step samples. "
            "Plot only complete valid 20 s non-fallen trials; not measured joint speed, torque or energy."
        ),
        "aggregation": (
            "Each point averages eligible physical starts for one training seed. "
            "Black summaries use available training-seed means and sample SD; missing seeds stay absent. "
            "Counts retain all attempted trials. Starts 0-9 are shared, not independent policies."
        ),
        "attempts": attempts,
        "hardware_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Audited and retained {len(attempts)} attempts; wrote {args.output}.")


if __name__ == "__main__":
    main()
