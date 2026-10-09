"""Minimal collision topology must not alter the robot's other physics properties."""

from isaaclab_tasks.utils.hydra import resolve_presets
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    MINIMAL_SO101_CFG,
    SO101_CFG,
    SO101VialEnvCfg,
    _remove_non_gripper_colliders,
)


def test_minimal_only_removes_proximal_collisions():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.1)
        mesh = UsdGeom.Mesh.Define(stage, f"/Robot/{link}/collisions/mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
    camera = UsdGeom.Cube.Define(stage, "/Robot/gripper/collisions/camera_mount/Cube").GetPrim()
    UsdPhysics.CollisionAPI.Apply(camera)
    _remove_non_gripper_colliders(root)
    kept = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    assert kept == {
        "/Robot/gripper/collisions/mesh",
        "/Robot/moving_jaw_so101_v1/collisions/mesh",
        "/Robot/gripper/collisions/camera_mount/Cube",
    }
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert prim.GetAttribute("physics:mass").Get() > 0
    proximal = stage.GetPrimAtPath("/Robot/upper_arm/collisions/mesh")
    assert proximal.IsActive()
    assert not proximal.HasAPI(UsdPhysics.MeshCollisionAPI)


def test_minimal_config_changes_only_spawn_function_and_is_default():
    full, minimal = SO101_CFG.to_dict(), MINIMAL_SO101_CFG.to_dict()
    assert full["spawn"].pop("func") != minimal["spawn"].pop("func")
    assert full == minimal
    cfg = resolve_presets(SO101VialEnvCfg(), ["newton_mjwarp"])
    assert cfg.scene.robot.spawn.func == MINIMAL_SO101_CFG.spawn.func
    cfg.from_dict(
        {
            "scene": {
                "robot": {
                    "spawn": {
                        "func": (
                            "isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg:"
                            "_spawn_so101_with_camera_overrides"
                        )
                    }
                }
            }
        }
    )
    assert callable(cfg.scene.robot.spawn.func)
    assert cfg.scene.robot.spawn.func._resolve() == SO101_CFG.spawn.func


def test_all_camera_tasks_default_to_minimal_with_housing_override():
    from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import (
        SO101VialCameraDistillationEnvCfg,
        SO101VialCameraEnvCfg,
        _spawn_minimal_so101_for_wrist_camera,
    )
    from isaaclab_tutorial.tasks.place_vial.config.so101.robust_env_cfg import SO101VialCameraAppearanceEnvCfg
    from isaaclab_tutorial.tasks.place_vial.config.so101.sim2real_env_cfg import (
        SO101VialCameraSim2RealEnvCfg,
        SO101VialSim2RealEnvCfg,
    )

    for config in (
        SO101VialCameraEnvCfg,
        SO101VialCameraDistillationEnvCfg,
        SO101VialCameraSim2RealEnvCfg,
        SO101VialCameraAppearanceEnvCfg,
    ):
        cfg = resolve_presets(config(), ["newton_mjwarp", "newton_renderer"])
        assert cfg.scene.robot.spawn.func == _spawn_minimal_so101_for_wrist_camera
    cfg = resolve_presets(SO101VialSim2RealEnvCfg(), ["newton_mjwarp"])
    assert cfg.scene.robot.spawn.func == MINIMAL_SO101_CFG.spawn.func


def test_both_robot_configs_disable_self_collision_on_both_backends():
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    for cfg in (SO101_CFG, MINIMAL_SO101_CFG):
        for properties in cfg.spawn.articulation_props:
            if isinstance(properties, NewtonArticulationCfg):
                assert properties.self_collision_enabled is False
            else:
                assert properties.enabled_self_collisions is False
        assert cfg.spawn.activate_contact_sensors is True
