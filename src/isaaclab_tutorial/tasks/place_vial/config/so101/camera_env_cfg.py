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


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    """SO-101 scene with the rectified pinhole wrist camera and handles to the randomized visual materials."""

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera/pinhole",
        spawn=PinholeCameraCfg(focal_length=_FOCAL_LENGTH, horizontal_aperture=_HORIZONTAL_APERTURE),
        offset=CameraCfg.OffsetCfg(convention="opengl"),
        data_types=["rgb"],
        width=80,
        height=60,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=_camera_renderer_cfg(),
        background_color=(0.3, 0.3, 0.3),
    )

    # The Newton renderer only reads material color.
    robot_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Robot/Looks/material_a_d_printed", spawn=None)
    desk_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Desk/Looks/Wood", spawn=None)
    rack_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Rack/WorkshopVisual/Looks/OmniPBR", spawn=None)
    vial_body_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_plastic", spawn=None)
    vial_cap_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_cap", spawn=None)
    vial_label_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Label", spawn=None)

    def __post_init__(self):
        self.robot.spawn.hide_camera_housing = True
        self.vial.spawn.usd_path = str(VIAL_LABELED_USD)


@configclass
class WristImageCfg(ObsGroup):
    """Two-frame wrist RGB history with episode-consistent photometric (and, for sim2real, intrinsics) variation."""

    image = ObsTerm(
        func=mdp.wrist_rgb,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
            "gamma_range": preset(default=(1.0, 1.0), sim2real=(0.85, 1.15)),
            "blur_range": preset(default=(0.0, 0.0), sim2real=(0.0, 0.5)),
            "history_length": 2,
            "projection_size": POLICY_IMAGE_SIZE,
            "focal_length_pixels": WRIST_FOCAL_LENGTH_PIXELS,
            "focal_scale_range": preset(default=(1.0, 1.0), sim2real=(0.95, 1.05)),
            "principal_point_pixels": preset(default=0.0, sim2real=1.5),
            "radial_distortion_range": preset(default=(0.0, 0.0), sim2real=(-0.04, 0.04)),
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ProprioceptionCfg(ObsGroup):
    """Proprioceptive observations available on the real robot, in the encoders' (possibly biased) coordinates."""

    joint_pos = ObsTerm(
        func=mdp.encoder_joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    joint_target = ObsTerm(
        func=mdp.encoder_joint_target,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005),
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    """Deployable student inputs, the state teacher's inputs (distillation labels), and the privileged critic."""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


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


@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    """Vial placement from wrist RGB and proprioception, distilled from the state teacher."""

    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()
    events: EventsCfg = preset(default=EventsCfg(), sim2real=CameraSim2RealEventsCfg())

    def play_mode(self):
        from isaaclab_tutorial.utils import evaluation

        super().play_mode()
        if not evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(self.scene.num_envs, 8)
