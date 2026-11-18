"""Remove the unused wrist camera without altering the stock arm or jaw dynamics."""

from isaaclab.sim.utils import clone

from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import _spawn_so101 as spawn_so101

WRIST_CAMERA_PARTS = (
    "gripper/wowrobo_2MP_camera",
    "gripper/visuals/camera_mount",
    "gripper/collisions/camera_mount",
)


def remove_wrist_camera(robot_prim):
    """Deactivate the sensor, housing, and its colliders in this scene's USD layer."""
    stage = robot_prim.GetStage()
    parts = [stage.GetPrimAtPath(robot_prim.GetPath().AppendPath(path)) for path in WRIST_CAMERA_PARTS]
    if not all(parts):
        raise ValueError("SO-101 asset no longer contains the expected wrist camera components")
    for prim in parts:
        prim.SetActive(False)


@clone
def spawn_without_wrist_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_so101(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    remove_wrist_camera(prim)
    return prim
