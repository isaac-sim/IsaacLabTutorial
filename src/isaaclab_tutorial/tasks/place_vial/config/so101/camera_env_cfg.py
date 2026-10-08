"""Wrist-camera observation variant of the SO-101 vial task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from pxr import UsdGeom

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    CriticStateGroupCfg,
    PolicyStateGroupCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
    _spawn_so101_with_camera_overrides,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.visuals import (
    CAMERA_BACKGROUND_COLOR,
    workshop_camera_renderer_cfg,
)


@clone
def _spawn_so101_for_wrist_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = _spawn_so101_with_camera_overrides(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    # Mount randomization moves the optical frame, not the housing mesh. The camera cannot
    # see its own housing in reality; exclude only its visual assembly, retaining collisions
    # and both gripper jaws. Newton ray tracing does not apply the camera's near clipping plane.
    housing = prim.GetStage().GetPrimAtPath(f"{prim_path}/gripper/visuals/camera_mount")
    UsdGeom.Imageable(housing).MakeInvisible()
    return prim


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    """SO-101 scene with the same rectified pinhole wrist camera on every renderer."""

    wrist_camera = CameraCfg(
        # Inherit the asset's camera pose, but not its RTX-only calibrated lens model.
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera/pinhole",
        spawn=PinholeCameraCfg(focal_length=13.6, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(convention="opengl"),
        data_types=["rgb"],
        width=64,
        height=48,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=workshop_camera_renderer_cfg(),
        background_color=CAMERA_BACKGROUND_COLOR,
    )

    def __post_init__(self):
        self.robot.spawn.func = _spawn_so101_for_wrist_camera


@configclass
class WristImageCfg(ObsGroup):
    """Domain-randomized wrist RGB observations."""

    image = ObsTerm(
        func=mdp.DomainRandomizedCameraImage,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
            "gamma_range": (1.0, 1.0),
            "normalize_intensity": False,
            "data_type": "rgb",
            "encode_srgb": False,
            "shift_pixels": 0,
            "blur_range": (0.0, 0.0),
            "history_length": 1,
            "dropout_probability": 0.0,
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ProprioceptionCfg(ObsGroup):
    """Proprioceptive observations available on the real robot."""

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
    """Deployable actor inputs plus the privileged asymmetric-critic state."""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class DistillationObservationsCfg(CameraObservationsCfg):
    """Camera observations plus the state teacher's inputs, used only to label the student's data."""

    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()


@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    """Vial placement from wrist RGB and proprioception."""

    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def play_mode(self):
        from isaaclab_tutorial.utils import evaluation

        super().play_mode()
        if not evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(self.scene.num_envs, 8)


@configclass
class SO101VialCameraDistillationEnvCfg(SO101VialCameraEnvCfg):
    """The camera task with the teacher's state observations attached for distillation."""

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
