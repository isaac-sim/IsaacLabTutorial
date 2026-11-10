# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the byte-locked robot inputs before reproducing the tutorial experiments."""

import argparse
import json
from pathlib import Path

from isaaclab_tutorial.locomotion.assets import collect_robot_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/asset-verification.json"))
    parser.add_argument("--copy-to", type=Path, help="Archive the verified USD bytes for offline provenance.")
    args = parser.parse_args()
    report = collect_robot_assets(args.copy_to)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Both robot assets match the locked SHA256 values. Recorded {args.output}.")


if __name__ == "__main__":
    main()
