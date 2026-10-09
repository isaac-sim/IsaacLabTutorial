"""Physical transfer uncertainty must be real, bounded, and reset-safe."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from isaaclab.envs.mdp.actions import RelativeJointPositionAction
from isaaclab.managers import EventTermCfg

from isaaclab_tutorial.assets import RESET_DATASET
from isaaclab_tutorial.tasks.place_vial.config.so101.robust_env_cfg import (
    SO101VialCameraTransferEnvCfg,
    SO101VialTransferEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.mdp.actions import (
    RobustRelativeJointPositionAction,
    SoftLimitRelativeGripperAction,
)
from isaaclab_tutorial.tasks.place_vial.mdp.events import ResetFromDataset
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import tabletop_vial_overlaps_rack
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import CANONICAL_START


def test_transfer_teacher_student_share_physics_and_play_retains_it():
    state, camera = SO101VialTransferEnvCfg(), SO101VialCameraTransferEnvCfg()
    for name, event in vars(state.events).items():
        if event is not None:
            assert event.to_dict() == getattr(camera.events, name).to_dict()
    assert state.actions.to_dict() == camera.actions.to_dict()
    before = state.events.to_dict(), camera.events.to_dict()
    for events in before:
        events["reset_from_dataset"]["params"]["sequential"] = True
    state.play_mode()
    camera.play_mode()
    assert (state.events.to_dict(), camera.events.to_dict()) == before
    assert state.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START


def test_transfer_reset_partial_ids_clearance_height_and_curriculum():
    class Scene(dict):
        pass

    count = 1024
    rack = MagicMock()
    rack.data.default_root_pose.torch = torch.tensor([0.18, 0.08, 0.04, 0, 0, 0, 1.0]).repeat(count, 1)
    support = MagicMock()
    support.data.default_root_pose.torch = torch.tensor([0.22, 0.02, 0.0325, 0, 0, 0, 1.0]).repeat(count, 1)
    scene = Scene(robot=MagicMock(), vial=MagicMock(), rack=rack, support=support)
    scene.env_origins = torch.zeros(count, 3)
    scene.env_origins[:, 0] = torch.arange(count).remainder(32) * 0.9
    env = SimpleNamespace(
        num_envs=count, device="cpu", cfg=SimpleNamespace(seed=144), scene=scene, action_manager=MagicMock()
    )
    term = ResetFromDataset(
        EventTermCfg(func=ResetFromDataset, mode="reset", params={"dataset_path": str(RESET_DATASET)}), env
    )
    for _ in range(3):
        ids = torch.arange(0, count, 2)
        term(
            env,
            ids,
            str(RESET_DATASET),
            sequential=True,
            home_position_noise=0.02,
            home_rack_clearance=0.001,
            home_rack_position_noise=0.005,
            home_rack_yaw_noise=0.0873,
            home_heading_noise=0.262,
            support_height_range=(0.0, 0.004),
        )
        rows = env._so101_reset_row[ids]
        home = term.states["phase"][rows] == 0
        vial = env._so101_reset_vial_pose[ids]
        actual_rack = rack.write_root_pose_to_sim_index.call_args.kwargs["root_pose"].clone()
        actual_rack[:, :3] -= scene.env_origins[ids]
        height = env._so101_support_height[ids]
        assert (height >= 0).all() and (height <= 0.004).all()
        assert (height[~home] == 0).all()
        assert not tabletop_vial_overlaps_rack(vial[home], env._so101_reset_rack_pose[ids][home], 0.001).any()
        assert not tabletop_vial_overlaps_rack(vial[home], actual_rack[home], 0.00099).any()
        torch.testing.assert_close(vial[~home], term.states["vial_pose"][rows[~home]])
        torch.testing.assert_close(vial[:, 2], term.states["vial_pose"][rows, 2] + height)
        assert (actual_rack[:, :2] - rack.data.default_root_pose.torch[ids, :2]).abs().max() <= 0.005001
        assert (env._so101_support_height[1::2] == 0).all()


def test_command_delay_uses_previous_command_and_resets_only_selected_rows():
    action = RobustRelativeJointPositionAction.__new__(RobustRelativeJointPositionAction)
    action._env = SimpleNamespace(_so101_command_delay=torch.tensor([False, True]))
    action._previous_command = torch.tensor([[0.1], [0.2]])
    action._command_gain = torch.ones(2, 1)
    action._joint_names = ["gripper"]
    action.cfg = SimpleNamespace(gain_range=(1.0, 1.0), delay_probability=0.25)
    with patch.object(RelativeJointPositionAction, "process_actions") as process:
        action.process_actions(torch.tensor([[0.3], [0.4]]))
        torch.testing.assert_close(process.call_args.args[1], torch.tensor([[0.3], [0.2]]))
        action._env._so101_command_delay[:] = False
        action._command_gain[:] = 1.1
        action.process_actions(torch.tensor([[1.0], [-2.0]]))
        torch.testing.assert_close(process.call_args.args[1], torch.tensor([[1.1], [-1.1]]))
    with patch.object(SoftLimitRelativeGripperAction, "reset"):
        action.reset(torch.tensor([1]))
    torch.testing.assert_close(action._previous_command, torch.tensor([[1.0], [0.0]]))


def test_rolling_resistance_only_enables_support_contacts():
    from isaaclab_tutorial.tasks.place_vial.mdp.events import configure_support_rolling_contacts

    builder = SimpleNamespace(
        shape_label=[
            "/env_0/Support/geometry",
            "/env_0/Desk/collision",
            "/env_0/Vial/body_collider",
            "/env_0/Robot/gripper/collision",
            "/env_0/Rack/collision",
            "/env_1/Support/geometry",
        ],
        shape_world=[0, 0, 0, 0, 0, 1],
        shape_material_mu_rolling=[0.05] * 6,
        shape_material_mu_torsional=[0.005] * 6,
        custom_attributes={"mujoco:condim": SimpleNamespace(values={i: 3 for i in range(6)})},
    )
    configure_support_rolling_contacts(builder, (0.0002, 0.002), (0.001, 0.005))
    assert list(builder.custom_attributes["mujoco:condim"].values.values()) == [6, 6, 3, 3, 3, 6]
    assert builder.shape_material_mu_rolling[0] == builder.shape_material_mu_rolling[1]
    assert 0.0002 <= builder.shape_material_mu_rolling[0] <= 0.002
    assert builder.shape_material_mu_rolling[2:5] == [0.0, 0.0, 0.0]
    assert builder.shape_material_mu_torsional[2:5] == [0.0, 0.0, 0.0]
