"""Wrist-camera and proprioception variant of the vial task."""

import os

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from . import mdp
from .env_cfg import CriticStateGroupCfg, PolicyStateGroupCfg, SO101SceneCfg, SO101VialEnvCfg


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    """Base scene plus the physical SO-101 wrist camera."""

    wrist_camera = CameraCfg(
        # The Sensor=sensors variant supplies the measured projection below,
        # but its central lens pose is unusable for this cap grasp: the cap is
        # only about 28 mm in front of, and coaxial with, that lens. It hides
        # the rack completely after pickup. Model the real camera on a rigid
        # 55 mm side bracket instead. This is a fixed, buildable wrist mount,
        # never a moving or world-space camera target.
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.6,
            horizontal_aperture=25.657,
            vertical_aperture=19.266,
            clipping_range=(0.01, 10.0),
            distortion=sim_utils.OpenCvPinholeDistortionCfg(
                fx=33.926593,
                fy=33.882010,
                cx=32.355810,
                cy=25.027360,
                image_size=(64, 48),
                k1=0.07702322,
                k2=-0.13605453,
                k3=0.05163219,
                p1=-0.00024938,
                p2=-0.00175006,
            ),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.055, 0.052, -0.035),
            # XYZW OpenGL frame. The fixed -Z boresight intersects the task
            # corridor at (0.010, -0.140, -0.090) m in the gripper frame.
            rot=(-0.09871531, 0.59943614, 0.78375556, -0.12906908),
            convention="opengl",
        ),
        data_types=["rgb"],
        # Retain the calibrated sensor's 4:3 aspect ratio while keeping the
        # on-policy rollout buffer tractable for the camera batch.
        width=64,
        height=48,
        update_period=1.0 / 30.0,
        # Articulation link poses are updated by the physics view, not by
        # rewriting their authored USD transforms. Synchronizing the camera
        # pose here keeps the renderer attached to the moving gripper.
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class WristImageCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.DomainRandomizedCameraImage,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ProprioceptionCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    joint_target = ObsTerm(func=mdp.joint_target, noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005))
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    """Wrist RGB and proprioception actor with a privileged state critic."""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    """Vision actor limited to wrist RGB, joint state, and previous action."""

    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=128, env_spacing=0.9, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def play_mode(self):
        super().play_mode()
        if not os.environ.get("SO101_EVAL_EPISODES"):
            self.scene.num_envs = min(self.scene.num_envs, 8)
