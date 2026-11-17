"""Camera removal must preserve grasp geometry and support portable continuation."""

import pytest


def test_camera_removal_disables_sensor_visuals_and_contacts_only():
    from pxr import Usd, UsdPhysics

    from isaaclab_tutorial.active_gaze.robot_geometry import WRIST_CAMERA_PARTS, remove_wrist_camera

    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim("/Robot", "Xform")
    gripper = stage.DefinePrim("/Robot/gripper", "Xform")
    mass = UsdPhysics.MassAPI.Apply(gripper)
    mass.CreateMassAttr(0.087)
    jaw = stage.DefinePrim("/Robot/gripper/collisions/fixed_jaw", "Cube")
    UsdPhysics.CollisionAPI.Apply(jaw)
    moving = stage.DefinePrim("/Robot/moving_jaw_so101_v1", "Xform")
    for part in WRIST_CAMERA_PARTS:
        stage.DefinePrim(robot.GetPath().AppendPath(part), "Xform")
    housing = stage.DefinePrim("/Robot/gripper/collisions/camera_mount/collider", "Cube")
    UsdPhysics.CollisionAPI.Apply(housing)
    remove_wrist_camera(robot)
    assert all(not stage.GetPrimAtPath(robot.GetPath().AppendPath(p)).IsActive() for p in WRIST_CAMERA_PARTS)
    assert jaw.IsActive() and jaw.HasAPI(UsdPhysics.CollisionAPI)
    assert moving.IsActive()
    assert mass.GetMassAttr().Get() == pytest.approx(0.087)
    active_colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert active_colliders == [jaw]


def test_camera_variant_keeps_policy_spaces_and_other_task_settings():
    from isaaclab_tutorial.active_gaze.configuration import compose

    stock = compose(["presets=numbered,stock_wrist_camera"])[0].to_dict()
    removed = compose(["presets=numbered"])[0].to_dict()
    assert removed["scene"]["robot"]["spawn"].pop("func") != stock["scene"]["robot"]["spawn"].pop("func")
    assert removed == stock
