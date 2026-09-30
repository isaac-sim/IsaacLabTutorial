"""Wrist-camera observation variant of the SO-101 vial task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    CriticStateGroupCfg,
    PolicyStateGroupCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
)


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    """SO-101 scene reading the wrist camera authored in the robot asset."""

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,
        data_types=["rgb"],
        width=64,
        height=48,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


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
