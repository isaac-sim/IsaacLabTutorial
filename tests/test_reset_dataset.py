"""Portable reset-artifact schema tests."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tutorial.assets import RESET_DATASET
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_INITIAL_JOINT_POSITION
from isaaclab_tutorial.tasks.place_vial.mdp.events import _ids, _phase_balanced_row_weights
from isaaclab_tutorial.tasks.place_vial.reset import dataset as reset_dataset
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START
from isaaclab_tutorial.tasks.place_vial.reset.dataset import PHASE_NAMES, load_reset_dataset, save_reset_dataset


def _states(rows: int = 8) -> dict[str, torch.Tensor]:
    vial_pose = torch.zeros((rows, 7))
    vial_pose[:, 6] = 1.0
    return {
        "joint_position": torch.zeros((rows, 6)),
        "joint_target": torch.zeros((rows, 6)),
        "vial_pose": vial_pose,
        "phase": torch.arange(rows, dtype=torch.long) % len(PHASE_NAMES),
        "difficulty": torch.linspace(0.0, 1.0, rows),
        "grasped": torch.arange(rows) >= 2,
        "lifted": torch.arange(rows) >= 3,
    }


def test_reset_dataset_round_trip(tmp_path):
    path = tmp_path / "resets.pt"
    written = save_reset_dataset(path, _states(), generator={"seed": 42}, validation={"physics": "newton"})
    loaded = load_reset_dataset(path)

    assert written["content_sha256"] == loaded["content_sha256"]
    assert loaded["row_count"] == 8
    assert loaded["phase_names"] == PHASE_NAMES
    assert torch.equal(loaded["states"]["phase"], torch.arange(8))


def test_reset_dataset_rejects_non_unit_quaternion(tmp_path):
    states = _states()
    states["vial_pose"][0, 6] = 0.5
    with pytest.raises(ValueError, match="normalized"):
        save_reset_dataset(tmp_path / "bad.pt", states, generator={}, validation={})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("phase", torch.zeros(8), "integer dtype"),
        ("grasped", torch.zeros(8, dtype=torch.long), "boolean dtype"),
        ("joint_target", torch.zeros((8, 6), dtype=torch.long), "floating-point dtype"),
        ("vial_pose", None, "must be a tensor"),
    ),
)
def test_reset_dataset_rejects_invalid_field_types(tmp_path, field, value, message):
    states = _states()
    states[field] = value

    with pytest.raises(ValueError, match=message):
        save_reset_dataset(tmp_path / "bad.pt", states, generator={}, validation={})


def test_reset_dataset_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "resets.pt"
    path.write_bytes(b"previous artifact")

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(reset_dataset.torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="save failed"):
        save_reset_dataset(path, _states(), generator={}, validation={})

    assert path.read_bytes() == b"previous artifact"
    assert list(tmp_path.glob(".resets.pt.*")) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("states", None, "must be a mapping"),
        ("generator", [], "metadata must be a dictionary"),
        ("row_count", True, "row_count"),
    ),
)
def test_reset_dataset_rejects_malformed_artifacts(tmp_path, field, value, message):
    path = tmp_path / "resets.pt"
    artifact = save_reset_dataset(path, _states(), generator={}, validation={})
    artifact[field] = value
    torch.save(artifact, path)

    with pytest.raises(ValueError, match=message):
        load_reset_dataset(path)


def test_phase_weights_are_spread_uniformly_over_each_phase_rows():
    phase = torch.tensor([0, 0, 0, 0, 1, 1])

    row_weights = _phase_balanced_row_weights(phase, phase_weights=(1.0, 3.0))

    assert row_weights[phase == 0].tolist() == pytest.approx([0.25] * 4)
    assert row_weights[phase == 1].tolist() == pytest.approx([1.5] * 2)


def test_phase_weights_reject_a_requested_phase_with_no_rows():
    with pytest.raises(ValueError, match="no eligible rows for phases \\[1\\]"):
        _phase_balanced_row_weights(torch.tensor([0, 0, 2]), phase_weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="exactly 3 values"):
        _phase_balanced_row_weights(torch.tensor([0, 1, 2]), phase_weights=(1.0, 1.0))


@pytest.mark.parametrize(
    ("phase", "message"),
    (
        (torch.tensor([], dtype=torch.long), "nonempty"),
        (torch.tensor([0.0]), "nonnegative integers"),
        (torch.tensor([-1]), "nonnegative integers"),
    ),
)
def test_phase_weights_reject_malformed_phase_columns(phase, message):
    with pytest.raises(ValueError, match=message):
        _phase_balanced_row_weights(phase, phase_weights=(1.0,))


def test_reset_ids_are_validated():
    env = SimpleNamespace(num_envs=3, device="cpu")

    assert _ids(env, None).tolist() == [0, 1, 2]
    assert _ids(env, 1).tolist() == [1]
    with pytest.raises(ValueError, match="lie in"):
        _ids(env, [-1])
    with pytest.raises(ValueError, match="duplicates"):
        _ids(env, [1, 1])
    with pytest.raises(ValueError, match="integers"):
        _ids(env, [1.0])


def test_bundled_artifact_supports_both_reset_distributions():
    states = load_reset_dataset(RESET_DATASET)["states"]

    for phase_weights in (CANONICAL_START, ALL_PHASES):
        row_weights = _phase_balanced_row_weights(states["phase"], phase_weights)
        assert torch.isfinite(row_weights).all()
        assert row_weights.sum() == pytest.approx(sum(phase_weights))


def test_bundled_canonical_rows_are_exact_unstarted_home_resets():
    artifact = load_reset_dataset(RESET_DATASET)
    states = artifact["states"]
    canonical = states["phase"] == 0
    home = torch.tensor(WORKSHOP_INITIAL_JOINT_POSITION, dtype=states["joint_target"].dtype)

    assert canonical.sum().item() == 128
    assert torch.equal(states["joint_target"][canonical], home.expand(canonical.sum(), -1))
    assert torch.equal(states["difficulty"][canonical], torch.zeros(canonical.sum()))
    assert not states["grasped"][canonical].any()
    assert not states["lifted"][canonical].any()

    # These are distinct full-task starts, not repeated copies of one vial
    # pose. Bounds are measured after gravity settles the horizontal vial.
    tabletop_xy = states["vial_pose"][canonical, :2]
    span = tabletop_xy.amax(dim=0) - tabletop_xy.amin(dim=0)
    assert span[0] >= 0.050
    assert span[1] >= 0.075
    assert artifact["generator"]["vial_position_half_range"] == pytest.approx((0.030, 0.040))

    x, y, z, w = states["vial_pose"][canonical, 3:7].unbind(dim=-1)
    vial_axis_x = 2.0 * (x * z + w * y)
    vial_axis_y = 2.0 * (y * z - w * x)
    heading = torch.atan2(vial_axis_y, vial_axis_x)
    assert heading.amin() <= -0.34
    assert heading.amax() >= 0.34
