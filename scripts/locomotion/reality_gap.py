# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record isolated baseline/effect trials, then inspect their non-physical reference ghost."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.playground.ghost import show_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, help="Compare an actually trained second policy.")
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-only", action="store_true", help="Produce matched receipts without opening playback.")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/reality-gap"))
    args = parser.parse_args()
    if not args.checkpoint.is_file() or (args.candidate_checkpoint and not args.candidate_checkpoint.is_file()):
        parser.error("Comparison policies must be saved checkpoints, not a moving 'latest' selector.")
    evaluate = Path(__file__).with_name("evaluate.py")
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["--seed", str(args.seed), "--seconds", str(args.seconds)]
    for name, checkpoint, effects in (
        ("reference", args.checkpoint, []),
        (
            "candidate",
            args.candidate_checkpoint or args.checkpoint,
            [
                "--command-delay-ms",
                str(args.command_delay_ms),
                "--observation-delay-ms",
                str(args.observation_delay_ms),
            ],
        ),
    ):
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(evaluate),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--output",
                    str(args.output / name),
                    *common,
                    *effects,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    reference = json.loads((args.output / "reference/receipt.json").read_text())
    candidate = json.loads((args.output / "candidate/receipt.json").read_text())
    comparison = RecordedComparison(reference, candidate)
    result = {
        "kind": "policy" if args.candidate_checkpoint else "hardware_effect",
        "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "same_policy": reference["checkpoint_sha256"] == candidate["checkpoint_sha256"],
        "reference_conditions": reference["conditions"],
        "candidate_conditions": candidate["conditions"],
        "reference_metrics": reference["metrics"],
        "candidate_metrics": candidate["metrics"],
        "matched_initial_state_and_commands": True,
    }
    if not args.record_only:
        result["playback"] = show_comparison(
            comparison,
            headless=args.headless,
            side_by_side=args.side_by_side,
            capture=args.output / "comparison.png",
        )
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
