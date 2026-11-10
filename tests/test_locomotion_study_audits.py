# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collected-study readers distinguish numerical diagnostics from unexplained process failures."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def readers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/locomotion"))
    return importlib.import_module("summarize_robustness"), importlib.import_module("summarize_effect_sweeps")


def test_diagnostic_receipts_survive_expected_exit_but_other_process_failures_do_not(tmp_path, readers):
    reader, _ = readers
    receipt = {
        "valid": False,
        "metrics": {"invalid_reason": "nan_state"},
        "qualification": {"passed": False, "invalid_reason": "nan_state"},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    trial = {"receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "process_returncode": 1}
    assert reader.load_trial_receipt(path, trial) == receipt
    trial["process_returncode"] = -9
    with pytest.raises(ValueError, match="process"):
        reader.load_trial_receipt(path, trial)
    trial["process_returncode"] = 1
    receipt["valid"] = True
    path.write_text(json.dumps(receipt))
    trial["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="process"):
        reader.load_trial_receipt(path, trial)


def test_invalid_diagnostics_need_consistent_classification_and_unchanged_bytes(tmp_path, readers):
    reader, _ = readers
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "valid": False,
                "metrics": {"invalid_reason": "decoder_crashed"},
                "qualification": {"passed": False, "invalid_reason": "decoder_crashed"},
            }
        )
    )
    trial = {"receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "process_returncode": 1}
    with pytest.raises(ValueError, match="diagnostic"):
        reader.load_trial_receipt(path, trial)
    trial["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes"):
        reader.load_trial_receipt(path, trial)


def test_policy_identity_uses_bytes_and_rejects_duplicate_matches(tmp_path, readers):
    _, reader = readers
    policy = tmp_path / "model_49.pt"
    policy.write_bytes(b"Frozen policy bytes; basename does not define realized progress.")
    (tmp_path / "run.json").write_text("{}")
    expected = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert reader.find_policy(tmp_path, expected) == policy
    (tmp_path / "copy.pt").write_bytes(policy.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        reader.find_policy(tmp_path, expected)
