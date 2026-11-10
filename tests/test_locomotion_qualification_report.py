# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The portable report's displayed condition must agree with the actual recorded intervention."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_matrix_cannot_relabel_nominal_evidence_as_delayed(tmp_path, qualification_reports):
    root = Path(__file__).resolve().parents[1]
    report = qualification_reports[0]
    report["configuration"]["observation_delay_ms"] = 40
    source = tmp_path / "mislabeled.json"
    source.write_text(json.dumps(report))
    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/locomotion/qualification_workshop.py"),
            "--report",
            str(source),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2 and "displayed condition" in result.stderr
    assert not output.exists()


def test_condition_explorer_rejects_policy_changes_hidden_behind_equal_seed_labels(tmp_path, qualification_reports):
    root = Path(__file__).resolve().parents[1]
    baseline = qualification_reports[0]
    baseline_source = tmp_path / "baseline.json"
    baseline_source.write_text(json.dumps(baseline))
    candidate = qualification_reports[1]
    replacements = {
        model["sha256"]: hashlib.sha256(("different-policy-" + model["sha256"]).encode()).hexdigest()
        for model in candidate["configuration"]["models"]
    }
    for model in candidate["configuration"]["models"]:
        model["sha256"] = replacements[model["sha256"]]
    for trial in candidate["trials"]:
        trial["checkpoint_sha256"] = replacements[trial["checkpoint_sha256"]]
    source = tmp_path / "different-policies.json"
    source.write_text(json.dumps(candidate))
    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/locomotion/qualification_workshop.py"),
            "--report",
            str(baseline_source),
            "--report",
            str(source),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2 and "policy weights fixed" in result.stderr
    assert not output.exists()

    # Disjoint attempted evaluation seeds cannot hide changed configured policies either.
    baseline.update(complete=False, trials=[baseline["trials"][0]])
    candidate.update(complete=False, trials=[candidate["trials"][1]])
    baseline_source = tmp_path / "partial-baseline.json"
    baseline_source.write_text(json.dumps(baseline))
    source.write_text(json.dumps(candidate))
    command = list(result.args)
    command[command.index("--report") + 1] = str(baseline_source)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "policy weights fixed" in result.stderr
    assert not output.exists()
