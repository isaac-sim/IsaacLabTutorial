# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Choose physical effects in Newton, restart a real trial, and inspect its unchanged baseline ghost."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.effects import EffectSettings
from isaaclab_tutorial.locomotion.evaluation import resolve_checkpoint
from isaaclab_tutorial.locomotion.playground.effects import EffectControls
from isaaclab_tutorial.locomotion.playground.ghost import show_comparison
from isaaclab_tutorial.locomotion.protocols import START_PROTOCOLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact saved nominal PPO checkpoint.")
    parser.add_argument("--effects", type=Path, help="EffectSettings JSON for the initial candidate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-protocol", choices=START_PROTOCOLS, default="balance-starts-v1")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--headless", action="store_true", help="Record and render one candidate without desktop widgets."
    )
    parser.add_argument("--record-only", action="store_true", help="Record one candidate without rendering.")
    parser.add_argument(
        "--capture-ui", action="store_true", help="Render native widgets in an offscreen headless capture."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/effects-workshop"))
    args = parser.parse_args()
    try:
        if args.checkpoint == Path("latest"):
            raise ValueError("Choose an exact saved checkpoint so edits cannot select a different policy.")
        checkpoint = resolve_checkpoint(args.checkpoint).resolve()
        settings = EffectSettings(**json.loads(args.effects.read_text())) if args.effects else EffectSettings()
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Choose a fresh --output directory; previous experiment recordings are preserved.")
    args.output.mkdir(parents=True)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluate = Path(__file__).with_name("evaluate.py")
    source_files = (
        evaluate,
        Path(__file__).resolve().parents[2] / "src/isaaclab_tutorial/locomotion/effects.py",
        Path(__file__).resolve().parents[2] / "src/isaaclab_tutorial/locomotion/protocols.py",
        Path(__file__).resolve().parents[2] / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/env_cfg.py",
    )
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    def record(directory: Path, selected: EffectSettings) -> dict:
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash or any(
            hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path)] for path in source_files
        ):
            raise ValueError("Policy or evaluator source changed; preserve this workshop and start a fresh one.")
        directory.mkdir()
        snapshot = directory / "effects.json"
        snapshot.write_text(json.dumps(asdict(selected), indent=2) + "\n")
        command = [
            sys.executable,
            str(evaluate),
            "--checkpoint",
            str(checkpoint),
            "--start-protocol",
            args.start_protocol,
            "--seed",
            str(args.seed),
            "--seconds",
            str(args.seconds),
            "--effects",
            str(snapshot),
            "--output",
            str(directory),
        ]
        print(f"Recording {directory.name}: {asdict(selected)}", flush=True)
        with (directory / "evaluation.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        (directory / "execution.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        if result.returncode:
            raise RuntimeError(f"Trial failed; inspect {directory / 'evaluation.log'} and its diagnostic receipt.")
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["checkpoint_sha256"] != checkpoint_hash or receipt["effect_settings"] != selected.manifest():
            raise ValueError("The recording does not match the frozen policy/effect snapshot.")
        return receipt

    reference_path = args.output / "reference"
    reference = record(reference_path, EffectSettings())
    reference_hash = hashlib.sha256((reference_path / "receipt.json").read_bytes()).hexdigest()
    experiments = []
    while True:
        if hashlib.sha256((reference_path / "receipt.json").read_bytes()).hexdigest() != reference_hash:
            raise ValueError("The baseline recording changed; preserve this workshop and start a fresh one.")
        directory = args.output / f"trial-{len(experiments):02d}"
        candidate = record(directory, settings)
        comparison = RecordedComparison(reference, candidate)
        controls = EffectControls(settings)
        result = {
            "settings": settings.manifest(),
            "reference_receipt": str(reference_path / "receipt.json"),
            "candidate_receipt": str(directory / "receipt.json"),
            "reference_receipt_sha256": reference_hash,
            "candidate_receipt_sha256": hashlib.sha256((directory / "receipt.json").read_bytes()).hexdigest(),
            "reference_metrics": reference["metrics"],
            "candidate_metrics": candidate["metrics"],
            "matched_mechanical_start_and_commands": True,
        }
        if not args.record_only:
            result["playback"] = show_comparison(
                comparison,
                headless=args.headless,
                capture=directory / "comparison.png",
                effect_controls=controls,
                render_ui=args.capture_ui,
            )
        experiments.append(result)
        (args.output / "workshop.json").write_text(
            json.dumps(
                {"checkpoint_sha256": checkpoint_hash, "producer_sha256": source_hashes, "experiments": experiments},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if args.headless or args.record_only or controls.pending is None:
            break
        settings = controls.pending


if __name__ == "__main__":
    main()
