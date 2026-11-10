# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify published production-inventory links and exact output/evidence hashes without a simulator."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("docs/locomotion/production.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        inventory = json.loads(args.manifest.read_text())
        assets = inventory["assets"]
        experiences = inventory["experiences"]
        if len(assets) != 45 or {item["id"] for item in assets} != {f"V{i:02d}" for i in range(1, 46)}:
            raise ValueError("Inventory must identify all 45 requested visuals exactly once.")
        if len(experiences) != 8 or {item["id"] for item in experiences} != {f"I{i:02d}" for i in range(1, 9)}:
            raise ValueError("Inventory must identify all eight requested experiences exactly once.")
        records = [*inventory["source_lock"], *inventory["policy_and_evaluation_evidence"]]
        for asset in assets:
            if asset["status"] not in {"produced", "partial", "missing", "needs_hardware", "delegated_followup"}:
                raise ValueError(f"Unknown visual status for {asset['id']}.")
            if asset["status"] == "produced" and not asset["outputs"]:
                raise ValueError(f"Produced visual {asset['id']} has no output.")
            records.extend(asset["outputs"])
        for experience in experiences:
            if experience["status"] not in {"implemented", "partial", "delegated_followup"}:
                raise ValueError(f"Unknown experience status for {experience['id']}.")
            records.extend(experience["evidence"])
            for key in ("entry", "lesson"):
                if not (root / experience[key]).is_file():
                    raise ValueError(f"Missing {key} for {experience['id']}.")
        for record in records:
            path = root / record["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Changed or missing production input/output: {record['path']}")
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f"Visuals: {dict(Counter(item['status'] for item in assets))}")
    print(f"Experiences: {dict(Counter(item['status'] for item in experiences))}")
    print("Recorded files and hashes match. This check does not close visual-review or hardware gates.")


if __name__ == "__main__":
    main()
