"""SO-101 vial placement seen through a foveated scene camera whose gaze the policy moves.

The task is the tutorial's state task with the actor's state replaced by the camera's retina, the gaze and
proprioception. The critic keeps the full state. Select a variant with ``presets=``; the same name selects its agent
in :mod:`.agents`:

* ``nearest``: place the vial in any opening, with a CNN policy.
* ``numbered``: place each vial in the opening with its number, with a CNN policy (default).
* ``dino_language``: numbered, with frozen foveated DINO tokens and a language instruction.

A matched non-foveated model is any variant with a large CMF constant, for example ``env.scene.eye.cmf_a_deg=1000``.
"""

from __future__ import annotations

import math

import torch
from isaaclab_fovi.sensors import FoveatedCameraCfg

from isaaclab.assets import AssetBaseCfg, VisualMaterialCfg
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import DomeLightCfg, PinholeCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_tutorial.assets.numbers import LABEL_ASSETS
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import CameraSim2RealEventsCfg, ProprioceptionCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    ActionsCfg,
    CriticStateGroupCfg,
    EventsCfg,
    RewardsCfg,
    Sim2RealEventsCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
    TerminationsCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES

from .gaze import GazeActionCfg, oracle_labels
from .goals import (
    NumberedResetFromDataset,
    TargetPlacementSuccess,
    balanced_shuffled_clones,
    critic_goal_state,
    held_target_goal,
    numbered_spawn,
    numbered_vials,
    target_placement_features,
)
from .language import instruction_index
from .learning.dino import DEFAULT_CHECKPOINT, DEFAULT_MD5
from .learning.observations import DinoTokens
from .render_options import CameraOutputCfg, CameraOutputNameCfg, CameraRenderSettingsCfg, FoviRendererCfg
from .robot_geometry import spawn_without_wrist_camera
from .sampling import ResetSampling
from .sim2real import PHOTOMETRIC_RANGES, ExtrinsicsObservations, PhotometricVariation, randomize_camera_mount

##
# Scene camera
##


def camera_rotation(position, target) -> tuple[float, float, float, float]:
    """OpenGL look-at orientation (x, y, z, w) with world +Z up, computed without a simulator."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    back = np.asarray(position) - np.asarray(target)
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return tuple(float(v) for v in Rotation.from_matrix(np.stack([right, up, back], axis=1)).as_quat())


def scene_camera(position, target, field_of_view_deg: float, resolution: int = 64) -> FoveatedCameraCfg:
    """A fixed foveated camera at ``position`` looking at ``target``.

    The physical camera is an ideal square pinhole with the given field of view, and the retina spans it.
    """
    aperture = 48.0 * math.tan(math.radians(field_of_view_deg / 2))
    return FoveatedCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneEye",
        data_types=CameraOutputCfg(),
        renderer_cfg=FoviRendererCfg(),
        render_settings=CameraRenderSettingsCfg(),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=aperture,
            vertical_aperture=aperture,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=camera_rotation(position, target), convention="opengl"),
        foveated_res=resolution,
        mask_to_sensor_fov=True,
        update_period=0.0,
    )


# A front view along the robot's working axis, centered on the arm and the rack.
NEAREST_CAMERA = scene_camera((0.52, 0.0, 0.36), (0.12, 0.0, 0.12), 80.0)
# Closer, so the number labels cover more samples.
NUMBERED_CAMERA = scene_camera((0.46, 0.015, 0.34), (0.17, 0.025, 0.12), 78.0)


@configclass
class RobotSpawnCfg(PresetCfg):
    """The tutorial's SO-101; its wrist camera is removed unless ``stock_wrist_camera`` is selected."""

    stock_wrist_camera = SO101SceneCfg().robot.spawn
    default = stock_wrist_camera.replace(func=spawn_without_wrist_camera)


@configclass
class SceneCfg(SO101SceneCfg):
    robot = SO101SceneCfg().robot.replace(spawn=RobotSpawnCfg())
    eye = NEAREST_CAMERA
    light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=2000))

    # Materials that the sim2real colour events vary; the Newton renderer only reads material colour.
    robot_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Robot/Looks/material_a_d_printed", spawn=None)
    desk_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Desk/Looks/Wood", spawn=None)
    rack_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Rack/WorkshopVisual/Looks/OmniPBR", spawn=None)
    vial_body_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_plastic", spawn=None)
    vial_cap_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_cap", spawn=None)


@configclass
class NumberedSceneCfg(SceneCfg):
    """Vials numbered 1 to 4 in equal shares, and a rack with a number beside each opening."""

    eye = NUMBERED_CAMERA
    vial = SceneCfg().vial.replace(spawn=numbered_vials(SceneCfg().vial.spawn, str(LABEL_ASSETS)))
    rack = SceneCfg().rack.replace(spawn=numbered_spawn(SceneCfg().rack.spawn, "rack", str(LABEL_ASSETS)))

    def __post_init__(self):
        super().__post_init__()
        self.clone_cfg.clone_strategy = balanced_shuffled_clones


##
# MDP
##


class SceneRGB(ManagerTermBase):
    """The scene camera's retina as RGB in [0, 1], shape (num_envs, 3, H, W).

    The image is read again only when the policy decides, so an action chunk's native steps reuse it. With
    ``photometric`` ranges the retina's appearance varies per episode.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ranges = cfg.params.get("photometric")
        self.photometric = PhotometricVariation(env.num_envs, env.device, ranges) if ranges is not None else None
        self.cached = None

    def reset(self, env_ids=None):
        if self.photometric is not None:
            self.photometric.reset(env_ids)

    def __call__(self, env, photometric=None):
        if not getattr(env, "refresh_policy_vision", True) and self.cached is not None:
            return self.cached
        camera = env.scene["eye"]
        image = camera.data.output[camera.cfg.data_types[0]].torch[..., :3]
        if self.photometric is not None:
            image = self.photometric(image)
        self.cached = image.float().permute(0, 3, 1, 2).contiguous() / 255.0
        return self.cached


def gaze_state(env) -> torch.Tensor:
    return env.action_manager.get_term("gaze").proprioception()


def motor_action_rate(env) -> torch.Tensor:
    """The stock action-rate penalty on the six motor actions only; gaze moves freely."""
    actions = env.action_manager
    return (actions.action[:, :6] - actions.prev_action[:, :6]).square().sum(-1)


@configclass
class ActiveGazeActionsCfg(ActionsCfg):
    gaze = GazeActionCfg()


PHOTOMETRIC = preset(default=None, sim2real=PHOTOMETRIC_RANGES)
"""Episode-wise photometric variation of the retina, on with ``presets=sim2real``."""


@configclass
class ImageCfg(ObservationGroupCfg):
    image = ObservationTermCfg(func=SceneRGB, params=dict(photometric=PHOTOMETRIC))
    concatenate_dim = 0
    enable_corruption = False


@configclass
class GazeCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=gaze_state)
    enable_corruption = False


@configclass
class OracleCfg(ObservationGroupCfg):
    """The teacher's gaze action, used to supervise the gaze during training."""

    labels = ObservationTermCfg(func=oracle_labels)
    enable_corruption = False


@configclass
class OracleObservationCfg(PresetCfg):
    """Gaze supervision labels; never part of the actor's observations."""

    grasp_gated = OracleCfg()
    no_gaze_supervision = None
    default = grasp_gated


@configclass
class NumberedCriticCfg(CriticStateGroupCfg):
    """The stock critic state, measured against the target opening, plus the target's position and number."""

    placement = CriticStateGroupCfg().placement.replace(func=target_placement_features)
    numbered_goal = ObservationTermCfg(func=critic_goal_state)


@configclass
class ObservationsCfg:
    """Retina, proprioception and gaze for the CNN actor; the stock state for the critic."""

    scene_rgb = ImageCfg()
    proprioception = ProprioceptionCfg()
    gaze = GazeCfg()
    critic = CriticStateGroupCfg()
    oracle = OracleObservationCfg()


@configclass
class NumberedObservationsCfg(ObservationsCfg):
    critic = NumberedCriticCfg()


@configclass
class LanguageProprioceptionCfg(ProprioceptionCfg):
    gaze = ObservationTermCfg(func=gaze_state)


@configclass
class VisionCfg(ObservationGroupCfg):
    tokens = ObservationTermCfg(
        func=DinoTokens,
        params=dict(
            reference=DEFAULT_CHECKPOINT,
            checksum=DEFAULT_MD5,
            batch_size=128,
            # Transformer layers to run before exiting; None runs all of them.
            num_layers=None,
            data_type=CameraOutputNameCfg(),
            photometric=PHOTOMETRIC,
        ),
    )
    concatenate_dim = 0
    enable_corruption = False


@configclass
class InstructionCfg(ObservationGroupCfg):
    index = ObservationTermCfg(func=instruction_index)
    enable_corruption = False


@configclass
class LanguageObservationsCfg:
    """Frozen DINO tokens of the retina, an instruction, and proprioception with the gaze for the actor."""

    proprio = LanguageProprioceptionCfg()
    vision = VisionCfg()
    instruction = InstructionCfg()
    critic = NumberedCriticCfg()
    oracle = OracleObservationCfg()
    extrinsics = preset(default=None, sim2real=ExtrinsicsObservations())


@configclass
class ResetModeCfg(PresetCfg):
    default = "sgs"
    uniform = "uniform"
    sgs = "sgs"


@configclass
class CurriculumCfg:
    """Success-guided sampling of the reset dataset's phases."""

    reset_sampling = CurriculumTermCfg(
        func=ResetSampling,
        params=dict(mode=ResetModeCfg(), history=100, target=0.5, concentration=1.0, temperature=2.0, epsilon=1e-4),
    )


@configclass
class ActiveGazeRewardsCfg(RewardsCfg):
    action_rate = RewardsCfg().action_rate.replace(func=motor_action_rate)


@configclass
class NumberedRewardsCfg(ActiveGazeRewardsCfg):
    held_goal = RewardsCfg().held_goal.replace(func=held_target_goal)


@configclass
class NumberedEventsCfg(EventsCfg):
    reset_from_dataset = EventsCfg().reset_from_dataset.replace(func=NumberedResetFromDataset)


@configclass
class ActiveGazeSim2RealEventsCfg(Sim2RealEventsCfg):
    """The tutorial's physical sim2real randomization, the scene's colours and the camera's mount.

    The colours are those of the tutorial's wrist-camera task. Unlike the tutorial's sim2real resets, every reset
    phase stays in use, so the reset curriculum still applies.
    """

    robot_color = CameraSim2RealEventsCfg().robot_color
    desk_color = CameraSim2RealEventsCfg().desk_color
    rack_color = CameraSim2RealEventsCfg().rack_color
    vial_body_color = CameraSim2RealEventsCfg().vial_body_color
    vial_cap_color = CameraSim2RealEventsCfg().vial_cap_color
    camera_mount = EventTermCfg(func=randomize_camera_mount, mode="reset", params=dict(camera_name="eye"))

    def __post_init__(self):
        super().__post_init__()
        self.reset_from_dataset.params["phase_weights"] = ALL_PHASES


@configclass
class NumberedSim2RealEventsCfg(ActiveGazeSim2RealEventsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.reset_from_dataset.func = NumberedResetFromDataset


@configclass
class NumberedTerminationsCfg(TerminationsCfg):
    success = TerminationsCfg().success.replace(func=TargetPlacementSuccess)


@configclass
class FinalObservationCfg(PresetCfg):
    """Capture the critic's pre-reset state, which timeouts inside an action chunk need."""

    default = False
    chunking = True


##
# Environments
##


@configclass
class NearestEnvCfg(SO101VialEnvCfg):
    """Place the vial in any opening, seen through the foveated scene camera."""

    scene: SceneCfg = SceneCfg(num_envs=128, env_spacing=0.9)
    actions: ActiveGazeActionsCfg = ActiveGazeActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events = preset(default=EventsCfg(), sim2real=ActiveGazeSim2RealEventsCfg())
    rewards: ActiveGazeRewardsCfg = ActiveGazeRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    compute_final_obs = FinalObservationCfg()
    num_rerenders_on_reset = 1

    def play_mode(self):
        super().play_mode()
        self.curriculum.reset_sampling.params["mode"] = "uniform"


@configclass
class NumberedEnvCfg(NearestEnvCfg):
    """Place each vial in the opening with its number."""

    scene: NumberedSceneCfg = NumberedSceneCfg(num_envs=128, env_spacing=0.9)
    observations: NumberedObservationsCfg = NumberedObservationsCfg()
    events = preset(default=NumberedEventsCfg(), sim2real=NumberedSim2RealEventsCfg())
    rewards: NumberedRewardsCfg = NumberedRewardsCfg()
    terminations: NumberedTerminationsCfg = NumberedTerminationsCfg()


@configclass
class LanguageEnvCfg(NumberedEnvCfg):
    """Numbered placement from frozen foveated DINO tokens and a language instruction."""

    scene: NumberedSceneCfg = NumberedSceneCfg(
        num_envs=128, env_spacing=0.9, eye=NUMBERED_CAMERA.replace(foveated_res=128)
    )
    observations: LanguageObservationsCfg = LanguageObservationsCfg()


@configclass
class ActiveGazeEnvCfg(PresetCfg):
    """Task variants; each name also selects its agent."""

    nearest = NearestEnvCfg()
    numbered = NumberedEnvCfg()
    dino_language = LanguageEnvCfg()
    default = numbered
