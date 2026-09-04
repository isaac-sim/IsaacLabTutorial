import pytest
import torch

from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    GRASP_GRIPPER_POSITION,
    PREGRASP_GRIPPER_POSITION,
    RELEASE_GRIPPER_POSITION,
    TABLETOP_VIAL_HEADING_RANGE,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from isaaclab_tutorial.tasks.place_vial.mdp.terms import RACK_CLEARANCE_HEIGHT, VIAL_REST_HEIGHT
from isaaclab_tutorial.tasks.place_vial.reset.generator import (
    PREGRASP_PROOF_DIFFICULTY,
    TABLETOP_VIAL_POSITION_HALF_RANGE,
    WORKSHOP_PREGRASP_JOINT_POSITION,
    WORKSHOP_TASK_WAYPOINTS,
    _represented_lift,
    measured_lift_progress,
)


def test_pregrasp_resets_reach_the_grasp_transition():
    assert 0.23 < PREGRASP_PROOF_DIFFICULTY < 0.24


def test_tabletop_reset_region_is_modest_but_requires_object_relative_reaching():
    x_half_range, y_half_range = TABLETOP_VIAL_POSITION_HALF_RANGE

    assert x_half_range == pytest.approx(0.030)
    assert y_half_range == pytest.approx(0.040)
    assert 0.03 <= 2.0 * x_half_range <= 0.08
    assert 0.03 <= 2.0 * y_half_range <= 0.08
    assert pytest.approx((-0.35, 0.35)) == TABLETOP_VIAL_HEADING_RANGE


def test_gripper_commands_follow_real_robot_mapping():
    assert GRASP_GRIPPER_POSITION < PREGRASP_GRIPPER_POSITION < RELEASE_GRIPPER_POSITION
    assert len(WORKSHOP_PREGRASP_JOINT_POSITION) == 6
    assert (
        pytest.approx(
            (-0.1221070742, -0.9066845838, 0.1900876486, 1.4797928525, -0.8044013083, PREGRASP_GRIPPER_POSITION)
        )
        == WORKSHOP_INITIAL_JOINT_POSITION
    )


def test_real_robot_waypoints_cover_every_loaded_phase():
    assert tuple(WORKSHOP_TASK_WAYPOINTS) == (3, 4, 5, 6, 7)
    assert tuple(len(WORKSHOP_TASK_WAYPOINTS[phase]) for phase in (3, 4)) == (3, 3)
    assert all(len(WORKSHOP_TASK_WAYPOINTS[phase]) == 3 for phase in (5, 6, 7))
    assert all(len(waypoint) == 6 for segment in WORKSHOP_TASK_WAYPOINTS.values() for waypoint in segment)
    assert all(
        waypoint[-1] == pytest.approx(GRASP_GRIPPER_POSITION)
        for phase in (3, 4, 5, 6)
        for waypoint in WORKSHOP_TASK_WAYPOINTS[phase]
    )


def test_generator_lift_history_matches_physical_clearance():
    # Horizontal vials need their root one radius above the required lowest
    # point. The rack frame starts 40 mm above world zero.
    root_threshold = 0.040 + RACK_CLEARANCE_HEIGHT + 0.017
    vial_pose = torch.tensor(
        [
            [0.0, 0.0, root_threshold - 1.0e-4, 0.0, 2**-0.5, 0.0, 2**-0.5],
            [0.0, 0.0, root_threshold + 1.0e-4, 0.0, 2**-0.5, 0.0, 2**-0.5],
        ]
    )
    grasped = torch.ones(2, dtype=torch.bool)
    rack_z = torch.full((2,), 0.040)

    assert _represented_lift(3, vial_pose, rack_z, grasped).tolist() == [False, True]
    assert _represented_lift(4, vial_pose, rack_z, grasped).tolist() == [True, True]


def test_lift_difficulty_uses_measured_motion_not_command_fraction():
    initial = torch.tensor([0.052, 0.052, 0.052])
    height = torch.tensor([0.052, 0.1045, 0.157])

    assert torch.allclose(measured_lift_progress(height, initial), torch.tensor([0.0, 0.5, 1.0]))


def test_insertion_and_release_resets_preserve_historical_lift():
    vial_pose = torch.tensor([[0.0, 0.0, VIAL_REST_HEIGHT, 0.0, 0.0, 0.0, 1.0]])
    rack_z = torch.tensor([0.040])

    assert _represented_lift(6, vial_pose, rack_z, torch.ones(1, dtype=torch.bool)).item()
    assert _represented_lift(7, vial_pose, rack_z, torch.ones(1, dtype=torch.bool)).item()
    assert not _represented_lift(7, vial_pose, rack_z, torch.zeros(1, dtype=torch.bool)).item()
