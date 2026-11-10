# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package only tutorial source and locked environments for an independent OSMO trial."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/osmo-input"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(set(names)):
            if name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE") or name.startswith(
                ("src/", "environments/workflows/", "scripts/locomotion/")
            ):
                path = root / name
                if path.is_file() and not path.is_symlink():
                    bundle.add(path, arcname=name, recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / "source.sha256").write_text(f"{digest}  source.tar.gz\n")
    print(f"{archive.resolve()}\nsha256: {digest}")


if __name__ == "__main__":
    main()
