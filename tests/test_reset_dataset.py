"""Portable reset-artifact schema tests."""

import pytest
import torch

from so101_vial_place.assets import RESET_DATASET
from so101_vial_place.tasks.place_vial.config.so101.control import WORKSHOP_INITIAL_JOINT_POSITION
from so101_vial_place.tasks.place_vial.mdp.events import _phase_balanced_row_weights
from so101_vial_place.tasks.place_vial.reset.curriculum import (
    RESET_CURRICULA,
    RESET_MAXIMUM_DIFFICULTY,
    RESET_MINIMUM_DIFFICULTY,
)
from so101_vial_place.tasks.place_vial.reset.dataset import PHASE_NAMES, load_reset_dataset, save_reset_dataset


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


def test_curriculum_probability_is_independent_of_eligible_row_count():
    phase = torch.tensor([0, 0, 0, 0, 1, 1])
    difficulty = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    row_weights = _phase_balanced_row_weights(
        phase,
        difficulty,
        phase_weights=(1.0, 3.0),
        minimum_difficulty=((0, 0.4),),
    )

    assert row_weights[:3].tolist() == [0.0, 0.0, 0.0]
    assert row_weights[phase == 0].sum() == pytest.approx(1.0)
    assert row_weights[phase == 1].sum() == pytest.approx(3.0)


def test_curriculum_rejects_a_requested_phase_with_no_eligible_rows():
    phase = torch.tensor([0, 1])
    difficulty = torch.tensor([0.1, 0.9])

    with pytest.raises(ValueError, match="no eligible rows for phases \\[0\\]"):
        _phase_balanced_row_weights(
            phase,
            difficulty,
            phase_weights=(1.0, 1.0),
            minimum_difficulty=((0, 0.5),),
        )


def test_curriculum_can_bound_a_local_bridge_on_both_sides():
    phase = torch.tensor([0, 0, 0, 1, 1, 1])
    difficulty = torch.tensor([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])

    row_weights = _phase_balanced_row_weights(
        phase,
        difficulty,
        phase_weights=(1.0, 1.0),
        minimum_difficulty=((0, 0.2),),
        maximum_difficulty=((1, 0.7),),
    )

    assert row_weights.tolist() == pytest.approx([0.0, 0.5, 0.5, 0.5, 0.5, 0.0])


def test_every_named_curriculum_has_eligible_rows_in_bundled_artifact():
    states = load_reset_dataset(RESET_DATASET)["states"]

    for name, phase_weights in RESET_CURRICULA.items():
        row_weights = _phase_balanced_row_weights(
            states["phase"],
            states["difficulty"],
            phase_weights,
            RESET_MINIMUM_DIFFICULTY.get(name),
            RESET_MAXIMUM_DIFFICULTY.get(name),
        )
        assert torch.isfinite(row_weights).all()
        assert row_weights.sum() > 0.0


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
