"""Recording variant of the SO-101 vial task: the state task plus a fixed third-person camera.

The policy observations are unchanged, so any state checkpoint plays here. The camera is only read by
:mod:`isaaclab_tutorial.utils.sim2sim_video` to render side-by-side rollouts across physics backends.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101SceneCfg, SO101VialEnvCfg

RECORD_CAMERA_EYE = (0.62, -0.42, 0.42)
"""Camera position in the environment frame [m]: front-right of the robot, above the table."""

RECORD_CAMERA_TARGET = (0.20, 0.03, 0.08)
"""Point the camera looks at [m]: between the vial pick-up spot and the rack."""

RECORD_CAMERA_SIZE = (320, 240)
"""Recorded frame size (width, height) in pixels."""


def look_at_quaternion(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return the ``(x, y, z, w)`` rotation of a +X-forward, +Z-up camera at ``eye`` looking at ``target``.

    The rotation is a yaw about the world +Z axis toward the target followed by a pitch about the camera's
    +Y axis, so the image stays level (no roll). The component order matches :class:`CameraCfg.OffsetCfg`.
    """
    dx, dy, dz = (t - e for t, e in zip(target, eye, strict=True))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))  # positive pitch tilts the +X forward axis toward -Z
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    # q = q_yaw(Z) * q_pitch(Y)
    w = cy * cp
    x = -sy * sp
    y = cy * sp
    z = sy * cp
    return (x, y, z, w)


@configclass
class SO101RecordSceneCfg(SO101SceneCfg):
    """The SO-101 scene with a fixed third-person camera per environment."""

    record_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(
            pos=RECORD_CAMERA_EYE, rot=look_at_quaternion(RECORD_CAMERA_EYE, RECORD_CAMERA_TARGET), convention="world"
        ),
        data_types=["rgb"],
        width=RECORD_CAMERA_SIZE[0],
        height=RECORD_CAMERA_SIZE[1],
        update_period=1.0 / 30.0,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class SO101VialRecordEnvCfg(SO101VialEnvCfg):
    """State task with the recording camera attached (observations unchanged)."""

    scene: SO101RecordSceneCfg = SO101RecordSceneCfg(num_envs=8, env_spacing=0.9, replicate_physics=True)

    def play_mode(self):
        from isaaclab_tutorial.utils import evaluation

        super().play_mode()
        if not evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(self.scene.num_envs, 8)
