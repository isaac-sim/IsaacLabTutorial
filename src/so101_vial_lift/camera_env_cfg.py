"""Dual-camera asymmetric variant of the SO-101 vial task."""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from . import mdp
from .env_cfg import SO101SceneCfg, SO101VialEnvCfg, StateGroupCfg


def _camera(prim_path: str, *, spawn, offset: CameraCfg.OffsetCfg) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=spawn,
        offset=offset,
        data_types=["rgb"],
        width=64,
        height=64,
        update_period=1.0 / 30.0,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    ego_camera = _camera(
        "{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,
        offset=CameraCfg.OffsetCfg(),
    )
    external_camera = _camera(
        "{ENV_REGEX_NS}/ExternalCamera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.147562,
            focus_distance=0.4,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1592609, -0.4546363, 0.3167156),
            # OpenGL camera frame aimed at the center of the pickup-to-rack
            # workspace (the camera looks along local -Z with local +Y up).
            rot=(0.494294, -0.074370, -0.128862, 0.856468),
            convention="opengl",
        ),
    )


@configclass
class EgoImageCfg(ObsGroup):
    image = ObsTerm(func=mdp.NormalizedCameraImage, params={"sensor_cfg": SceneEntityCfg("ego_camera")})


@configclass
class ExternalImageCfg(ObsGroup):
    image = ObsTerm(func=mdp.NormalizedCameraImage, params={"sensor_cfg": SceneEntityCfg("external_camera")})


@configclass
class ProprioceptionCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    # Separate top-level groups let RSL-RL create one encoder for each image
    # while concatenating the 1-D proprioception directly into the MLP head.
    ego_rgb: EgoImageCfg = EgoImageCfg()
    external_rgb: ExternalImageCfg = ExternalImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    critic: StateGroupCfg = StateGroupCfg()


@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    """Camera actor with a privileged state critic."""

    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
