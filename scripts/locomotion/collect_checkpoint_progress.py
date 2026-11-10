# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Associate actual checkpoint rollouts with verified fresh-PPO collection/update progress."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.evaluation import summarize_episode
from isaaclab_tutorial.locomotion.protocols import analyze_velocity_protocol
from isaaclab_tutorial.locomotion.training import checkpoint_progress, read_training_run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = read_training_run(args.run)
    budget = run["requested_budget"]
    clips, receipts = [], []
    for path in args.receipt:
        receipt = json.loads(path.read_text())
        checkpoint = args.run / Path(receipt["checkpoint"]).name
        if (
            receipt["valid"] is not True
            or receipt["algorithm"] != "ppo"
            or receipt["training_profile"] != "nominal"
            or sha256(checkpoint) != receipt["checkpoint_sha256"]
        ):
            parser.error("Choose valid nominal PPO recordings of this run's actual saved checkpoints.")
        progress = checkpoint_progress(checkpoint)
        n = progress["completed_iterations"]
        if n > run["logged_budget"]["iterations"] or progress["optimizer_step_counters"] != {
            "optimizer_state_dict": [n * budget["epochs"] * budget["minibatches"]]
        }:
            parser.error("The selected checkpoint's realized collection/optimizer progress disagrees with its logs.")
        if (
            summarize_episode(receipt["trajectory"], receipt["control_dt_s"], receipt["horizon_s"])
            != receipt["metrics"]
            or analyze_velocity_protocol(receipt["trajectory"], receipt["control_dt_s"]) != receipt["qualification"]
        ):
            parser.error("Recorded motion does not reproduce the claimed evaluation metrics or response result.")
        if receipts:
            RecordedComparison(receipts[0], receipt)
            if any(
                receipts[0][key] != receipt[key]
                for key in ("conditions", "effect_settings", "asset_provenance", "body_names", "reward_weights")
            ):
                parser.error("Checkpoint progress needs the same physical conditions, robot and evaluation objective.")
        receipts.append(receipt)
        clips.append(
            {
                "receipt_path": os.path.relpath(path.resolve(), args.output.parent.resolve()),
                "receipt_sha256": sha256(path),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "completed_iterations": n,
                "collected_transitions": n * budget["rollout_transitions"],
                "optimizer_steps": n * budget["epochs"] * budget["minibatches"],
                "recorded_collection_and_update_time_s": run["cumulative_training_time_s"][n - 1],
                "metrics": receipt["metrics"],
                "qualified": receipt["qualification"]["passed"],
            }
        )
    clips.sort(key=lambda clip: clip["completed_iterations"])
    if len({clip["completed_iterations"] for clip in clips}) != len(clips):
        parser.error("Choose distinct checkpoints rather than duplicating a saved policy.")
    root = Path(__file__).resolve().parents[2]
    data = {
        "schema_version": 1,
        "scope": "Snapshots of one fresh PPO training run and one evaluation start; not independent training seeds.",
        "run": str(args.run.resolve()),
        "training_seed": run["seed"],
        "task": run["task"],
        "training_budget": budget,
        "logged_budget": run["logged_budget"],
        "run_inputs_sha256": run["source_sha256"],
        "config_fingerprints": run["config_fingerprints"],
        "evaluation_seed": receipts[0]["seed"],
        "start_condition": receipts[0]["start_condition"],
        "actual_starts_commands_conditions_matched": True,
        "asset_provenance": receipts[0]["asset_provenance"],
        "checkpoint_numbering": "model_0 is saved AFTER one completed rollout/update cycle; it is not untrained.",
        "recorded_training_time_scope": run["recorded_time_scope"],
        "clips": clips,
        "receipt_path_base": "Study directory; the renderer's --receipt-root can relocate the archived recordings.",
        "source_sha256": {
            str(p.relative_to(root)): sha256(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("evaluate.py"),
                root / "src/isaaclab_tutorial/locomotion/training.py",
                root / "src/isaaclab_tutorial/locomotion/protocols.py",
            )
        },
        "hardware_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(clips)} checkpoint recordings against actual models, optimizer counters and event logs.")


if __name__ == "__main__":
    main()
