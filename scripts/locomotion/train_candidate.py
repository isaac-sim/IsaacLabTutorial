# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute a saved one-factor PPO candidate only if its baseline still matches current source."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from isaaclab_tutorial.locomotion.training import candidate_command, current_nominal_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify the contract and print arguments without training."
    )
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        command = candidate_command(plan, *current_nominal_parameters())
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        result = subprocess.run(command, check=False)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
