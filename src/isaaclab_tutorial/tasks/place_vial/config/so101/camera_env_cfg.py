"""Wrist-camera variant of the SO-101 vial placement task, distilled from the state teacher.

It adds the wrist camera to :class:`~.env_cfg.SO101VialEnvCfg`. With ``presets=sim2real`` it uses the state task's
physical randomization plus appearance and camera randomization.
"""

import math

from isaaclab.assets import VisualMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.assets import VIAL_LABELED_USD
from isaaclab_tutorial.tasks.place_vial import mdp

from .env_cfg import (
    CriticStateGroupCfg,
    EventsCfg,
    PolicyStateGroupCfg,
    Sim2RealEventsCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
)


def _camera_renderer_cfg() -> MultiBackendRendererCfg:
    """Enable the supported shadow settings on both standalone renderers."""
    cfg = MultiBackendRendererCfg()
    cfg.newton_renderer.enable_shadows = True
    cfg.default.enable_shadows = True
    cfg.ovrtx.enable_shadows = True
    return cfg


# The camera renders an 80 x 60 overscan of the 64 x 48 policy view, so varied intrinsics still see real pixels.
POLICY_IMAGE_SIZE = (48, 64)
_FOCAL_LENGTH = 13.6
_HORIZONTAL_APERTURE = 20.955 * 80 / 64
WRIST_FOCAL_LENGTH_PIXELS = 80 * _FOCAL_LENGTH / _HORIZONTAL_APERTURE


# BEGIN lesson-06-SO101CameraSceneCfg
@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    pass
# END lesson-06-SO101CameraSceneCfg


# BEGIN lesson-06-WristImageCfg
@configclass
class WristImageCfg(ObsGroup):
    pass
# END lesson-06-WristImageCfg


# BEGIN lesson-06-ProprioceptionCfg
@configclass
class ProprioceptionCfg(ObsGroup):
    pass
# END lesson-06-ProprioceptionCfg


# BEGIN lesson-06-CameraObservationsCfg
@configclass
class CameraObservationsCfg:
    pass
# END lesson-06-CameraObservationsCfg


def _color_event(materials: str, color_range) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={"materials": SceneEntityCfg(materials), "channels": {"color": color_range}},
    )


@configclass
class CameraSim2RealEventsCfg(Sim2RealEventsCfg):
    """Sim2real physics plus appearance and wrist-camera mount variation."""

    # The printed-part material also binds the gripper; motors and pads keep their own materials.
    robot_color = _color_event("robot_visual", ((0.08, 0.04, 0.008), (1.0, 0.85, 0.85)))
    desk_color = _color_event("desk_visual", ((0.025, 0.025, 0.025), (0.60, 0.60, 0.60)))
    rack_color = _color_event("rack_visual", ((0.06, 0.06, 0.02), (1.0, 0.95, 0.95)))
    vial_body_color = _color_event("vial_body_visual", ((0.12, 0.12, 0.12), (0.95, 0.95, 0.95)))
    vial_cap_color = _color_event("vial_cap_visual", ((0.02, 0.02, 0.02), (0.95, 0.95, 0.95)))
    vial_label_color = _color_event("vial_label_visual", ((0.65, 0.65, 0.60), (1.0, 1.0, 1.0)))
    camera_mount = EventTerm(
        func=mdp.randomize_wrist_camera_mount,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "position_range": 0.003,
            "rotation_range": math.radians(3),
        },
    )


# BEGIN lesson-06-SO101VialCameraEnvCfg
@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    pass
# END lesson-06-SO101VialCameraEnvCfg
