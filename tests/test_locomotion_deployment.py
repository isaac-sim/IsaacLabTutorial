# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard Duck deployment must preserve physical conventions, not just tensor sizes."""

import copy

import numpy as np
import pytest

from isaaclab_tutorial.locomotion.deployment import (
    DUCK_HOME_RAD,
    DUCK_JOINTS,
    OBSERVATION_LAYOUT,
    build_manifest,
    check_duck_contract,
    inline_onnx_weights,
    rehearsal_trace,
    staged_bundle,
)


def contract():
    return {
        "joint_names": list(DUCK_JOINTS),
        "home_rad": list(DUCK_HOME_RAD),
        "action_scale": [1.0] * 14,
        "control_dt_s": 0.02,
        "observation_layout": dict(OBSERVATION_LAYOUT),
        "normalization": "embedded",
    }


def test_duck_contract_rejects_shape_compatible_but_physically_different_policies():
    check_duck_contract(contract())
    for field, value in (
        ("joint_names", list(reversed(DUCK_JOINTS))),
        ("home_rad", [0.0] * 14),
        ("action_scale", [0.9] * 14),
        ("control_dt_s", 0.01),
        ("observation_layout", {"opaque": 61}),
    ):
        changed = contract()
        changed[field] = value
        with pytest.raises(ValueError):
            check_duck_contract(changed)


def test_export_rehearsal_preserves_real_observations_and_episode_reset():
    observations = np.arange(122, dtype=np.float32).reshape(2, 61)
    trace = rehearsal_trace(observations)
    assert trace[0]["reset"] is True and trace[1]["reset"] is False
    np.testing.assert_array_equal(trace[1]["obs"], observations[1])
    observations[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rehearsal_trace(observations)


def test_manifest_requires_executed_parity_and_records_standard_runtime_contract():
    receipt = {
        "valid": True,
        "checkpoint": "saved/model_9.pt",
        "checkpoint_sha256": "a" * 64,
        "contract": contract(),
        "seed": 42,
        "metrics": {"fell": False},
        "qualification": {"passed": False, "criteria": {"id": "flat-response-v1"}},
        "conditions": {"observation_delay_ms": 40},
        "effect_settings": {"settings": {"backlash": True, "voltage": 6.5}},
        "asset_url": "assets/microduck_walk_backlash.usd",
        "onnx_parity": {"checked": True, "samples": 2, "max_abs_error": 1e-6},
    }
    manifest = build_manifest(receipt, onnx_sha256="b" * 64)
    assert manifest["schema_version"] == 2
    assert (manifest["obs_len"], manifest["action_len"], manifest["robot"]["control_hz"]) == (61, 14, 50)
    assert manifest["kind"] == "perpetual" and manifest["slot"] == "walk"
    assert manifest["training"]["checkpoint_sha256"] == "a" * 64
    assert manifest["eval"]["simulation_qualified"] is False
    assert manifest["eval"]["response_qualification"] == receipt["qualification"]
    assert manifest["eval"]["conditions"] == receipt["conditions"]
    assert manifest["eval"]["effect_settings"] == receipt["effect_settings"]
    assert manifest["eval"]["asset_url"] == receipt["asset_url"]
    changed = copy.deepcopy(receipt)
    changed["onnx_parity"]["checked"] = False
    with pytest.raises(ValueError, match="parity"):
        build_manifest(changed, onnx_sha256="b" * 64)


def test_failed_export_cannot_replace_a_valid_bundle_or_publish_partial_files(tmp_path):
    destination = tmp_path / "exported"
    with pytest.raises(RuntimeError), staged_bundle(destination) as stage:
        (stage / "policy.onnx").write_bytes(b"unchecked")
        raise RuntimeError("parity failed")
    assert not destination.exists()
    with staged_bundle(destination) as stage:
        (stage / "policy.onnx").write_bytes(b"checked")
        (stage / "manifest.json").write_text('{"checkpoint":"A"}')
    with pytest.raises(FileExistsError), staged_bundle(destination):
        pytest.fail("An existing bundle must be rejected before any file is replaced")
    assert (destination / "policy.onnx").read_bytes() == b"checked"
    assert (destination / "manifest.json").read_text() == '{"checkpoint":"A"}'


def test_policy_onnx_carries_weights_when_the_exporter_creates_an_external_sidecar(tmp_path):
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    weights = np.array([[2, 0], [0, 3]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "actor",
        [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 2])],
        initializer=[onnx.numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "policy.onnx"
    sidecar = tmp_path / "policy.onnx.data"
    onnx.save_model(
        model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=sidecar.name, size_threshold=0
    )
    assert sidecar.exists()
    inline_onnx_weights(path)
    sidecar.unlink()
    restored = onnx.load(path)
    onnx.checker.check_model(restored)
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(restored.graph.initializer[0]), weights)
