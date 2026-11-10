# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay a non-physical baseline ghost on an independently recorded candidate trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.playground.ghost import show_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--hide-reference", action="store_true")
    parser.add_argument("--opacity", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    comparison = RecordedComparison(reference, candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    result = show_comparison(
        comparison,
        headless=args.headless,
        capture=args.output / "comparison.png",
        side_by_side=args.side_by_side,
        opacity=args.opacity,
        show_reference=not args.hide_reference,
    )
    result.update(
        reference_receipt_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        candidate_receipt_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        reference_metrics=reference["metrics"],
        candidate_metrics=candidate["metrics"],
    )
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
