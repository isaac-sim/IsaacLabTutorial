"""A fixed, calibrated scene camera on the stock SO-101 task."""

import math
from typing import ClassVar

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import DomeLightCfg, PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg, resolve_presets

from isaaclab_tutorial.assets.numbers import LABEL_ASSETS
from isaaclab_tutorial.active_gaze.learning.dino import DEFAULT_CHECKPOINT, DEFAULT_MD5
from isaaclab_tutorial.active_gaze.learning.observations import DinoTokens
from isaaclab_fovi.sensors import FoveatedTiledCameraCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    ActionsCfg,
    CriticStateGroupCfg,
    EventsCfg,
    SO101VialEnvCfg,
)

from .env_cfg import ProprioceptionCfg
from .gaze import CachedGazeActionCfg, oracle_labels
from .language import instruction_index
from .render_options import CameraOutputCfg, CameraOutputNameCfg, CameraRenderSettingsCfg, FoviRendererCfg
from .robot_geometry import spawn_without_wrist_camera
from .sampling import ResetSampling, validate_settings
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101SceneCfg
from .transfer_options import CameraRandomizationCfg

# Front view along the robot's working axis, centered on the arm and vial board.
CAMERA_POSITION = (0.52, 0.0, 0.36)
CAMERA_TARGET = (0.12, 0.0, 0.12)
SOURCE_SIZE = 640
HORIZONTAL_FOV = 80.0
FOCAL_PIXELS = SOURCE_SIZE / (2 * math.tan(math.radians(HORIZONTAL_FOV / 2)))
CALIBRATION = dict(
    model="pinhole",
    image_size=(SOURCE_SIZE, SOURCE_SIZE),
    intrinsics=(FOCAL_PIXELS, FOCAL_PIXELS, 319.5, 319.5),
)


def camera_rotation(position=CAMERA_POSITION, target=CAMERA_TARGET):
    """OpenGL look-at orientation, XYZW, computed without a simulator."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    back = np.asarray(position) - np.asarray(target)
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return tuple(float(v) for v in Rotation.from_matrix(np.stack([right, up, back], axis=1)).as_quat())


class SceneRGB(ManagerTermBase):
    """Read the camera at policy boundaries while native control steps continue."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cached = None

    def __call__(self, env):
        if not getattr(env, "refresh_policy_vision", True) and self.cached is not None:
            return self.cached
        image = env.scene["eye"].data.output[env.scene["eye"].cfg.data_types[0]]
        image = image.torch if hasattr(image, "torch") else image
        self.cached = image[..., :3].float().permute(0, 3, 1, 2).contiguous() / 255.0
        return self.cached


def gaze_state(env):
    return env.action_manager.get_term("gaze").proprioception()


def motor_action_rate(env):
    # Preserve the stock penalty on the six motor increments only.
    return (env.action_manager.action[:, :6] - env.action_manager.prev_action[:, :6]).square().sum(-1)


@configclass
class FoveatedActionsCfg(ActionsCfg):
    """Preserve the pretrained motor contract across camera/randomization variants."""

    arm_action = resolve_presets(ActionsCfg()).arm_action
    gaze = CachedGazeActionCfg(angle_limit=math.radians(HORIZONTAL_FOV / 2 - 1), max_delta=0.12)


@configclass
class ImageCfg(ObservationGroupCfg):
    image = ObservationTermCfg(func=SceneRGB)
    concatenate_dim = 0
    enable_corruption = False


@configclass
class GazeCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=gaze_state)
    enable_corruption = False


@configclass
class OracleCfg(ObservationGroupCfg):
    labels = ObservationTermCfg(func=oracle_labels)
    enable_corruption = False


@configclass
class LanguageProprioceptionCfg(ProprioceptionCfg):
    gaze = ObservationTermCfg(func=gaze_state)


@configclass
class VisionCfg(ObservationGroupCfg):
    tokens = ObservationTermCfg(
        func=DinoTokens,
        params=dict(
            reference=DEFAULT_CHECKPOINT, checksum=DEFAULT_MD5, batch_size=128, data_type=CameraOutputNameCfg()
        ),
    )
    concatenate_dim = 0
    enable_corruption = False


@configclass
class InstructionCfg(ObservationGroupCfg):
    index = ObservationTermCfg(func=instruction_index)
    enable_corruption = False


@configclass
class OracleObservationCfg(PresetCfg):
    """Training-only gaze labels; never part of the actor observation group."""

    grasp_gated = OracleCfg()
    no_gaze_supervision = None
    default = grasp_gated


@configclass
class AuxiliaryObservationCfg(PresetCfg):
    from .sim2real import ExtrinsicsObservations as _ExtrinsicsObservations

    default = None
    extrinsics = _ExtrinsicsObservations()


@configclass
class ObservationsCfg:
    scene_rgb = ImageCfg()
    proprioception = ProprioceptionCfg()
    gaze = GazeCfg()
    # Stock state and contacts, with last_action naturally including gaze.
    critic = CriticStateGroupCfg()
    oracle = OracleObservationCfg()
    proprio: LanguageProprioceptionCfg | None = None
    vision: VisionCfg | None = None
    instruction: InstructionCfg | None = None
    extrinsics = AuxiliaryObservationCfg()


@configclass
class ResetModeCfg(PresetCfg):
    default = "sgs"
    uniform = "uniform"
    sgs = "sgs"


@configclass
class CurriculumCfg:
    reset_sampling = CurriculumTermCfg(
        func=ResetSampling,
        params=dict(mode=ResetModeCfg(), history=100, target=0.5, concentration=1.0, temperature=2.0, epsilon=1e-4),
    )


@configclass
class RobotSpawnCfg(PresetCfg):
    _stock = SO101SceneCfg().robot.spawn.replace(shoulder_response=None, elbow_response=None)
    stock_wrist_camera = _stock
    default = _stock.replace(func=spawn_without_wrist_camera)


def scene_camera(position=CAMERA_POSITION, target=CAMERA_TARGET, hfov=HORIZONTAL_FOV):
    """Construct one calibrated camera; scalar overrides address its native fields."""
    focal = SOURCE_SIZE / (2 * math.tan(math.radians(hfov / 2)))
    return FoveatedTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneEye",
        data_types=CameraOutputCfg(),
        renderer_cfg=FoviRendererCfg(),
        render_settings=CameraRenderSettingsCfg(),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=48.0 * math.tan(math.radians(hfov / 2)),
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=camera_rotation(position, target), convention="opengl"),
        foveated_res=64,
        foveated_style="warped_cartesian_as_grid",
        ndc_layout="native_grid",
        field_geometry="spherical",
        camera_calibration=dict(
            model="pinhole", image_size=(SOURCE_SIZE, SOURCE_SIZE), intrinsics=(focal, focal, 319.5, 319.5)
        ),
        cmf_a=0.036021,
        fixed_software_fov=True,
        software_fov_image_height=SOURCE_SIZE,
        software_fov_image_width=SOURCE_SIZE,
        update_period=0.0,
    )


@configclass
class SceneCfg(SO101SceneCfg):
    robot = SO101SceneCfg().robot.replace(
        spawn=RobotSpawnCfg(),
        init_state=resolve_presets(SO101SceneCfg().robot.init_state),
    )
    eye = scene_camera()
    light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=2000))
    diagnostic_camera: CameraCfg | None = None


@configclass
class NumberedSceneCfg(SceneCfg):
    eye = scene_camera((0.46, 0.015, 0.34), (0.17, 0.025, 0.12), 78.0).replace(
        cmf_a=0.015,
    )


@configclass
class LanguageObservationsCfg(ObservationsCfg):
    scene_rgb = None
    proprioception = None
    gaze = None
    proprio = LanguageProprioceptionCfg()
    vision = VisionCfg()
    instruction = InstructionCfg()


@configclass
class FinalObservationCfg(PresetCfg):
    """Capture pre-reset critic state for timeouts inside an action chunk."""

    default = False
    chunking = True


@configclass
class SO101FoveatedEnvCfg(SO101VialEnvCfg):
    """Nearest-slot CNN task with a calibrated foveated scene camera."""

    compute_final_obs = FinalObservationCfg()

    scene = SceneCfg(num_envs=128, env_spacing=0.9)
    observations = ObservationsCfg()
    actions = FoveatedActionsCfg()
    curriculum = CurriculumCfg()
    events = EventsCfg()
    mixed_gaze_resets: bool = True
    reset_sampling_evaluation: bool = False
    sim2real: CameraRandomizationCfg | None = None
    goal_mode: ClassVar[str] = "nearest"
    policy: ClassVar[str] = "cnn"
    number_label_assets: str = str(LABEL_ASSETS)

    def __post_init__(self):
        super().__post_init__()
        # The imported actor and mid-task reset dataset use the nominal SO-101
        # geometry, action scales, rewards, and horizon in every camera variant.
        self.episode_length_s = 20.0
        self.rewards = resolve_presets(self.rewards)
        self.rewards.action_rate.func = motor_action_rate
        self.num_rerenders_on_reset = 1

    def validate_config(self):
        """Validate the final config after presets and scalar overrides are applied."""
        if self.resolution() not in (64, 128):
            raise ValueError("SO-101 foveated resolution must be 64 or 128")
        if not math.isfinite(self.cmf_a()) or self.cmf_a() <= 0:
            raise ValueError("cmf_a must be positive and finite")
        if self.scene.eye.data_types != ["rgb"] and self.renderer_backend() != "ovrtx":
            raise ValueError("Minimal render modes require renderer=ovrtx")
        if self.policy == "dino_language":
            if self.resolution() != 128:
                raise ValueError("The pretrained foveated DINO policy requires native 128px input")
            tokens = self.observations.vision.tokens.params
            if tokens["data_type"] not in self.scene.eye.data_types:
                raise ValueError("The DINO observation must use an output produced by the scene camera")
            if tokens["batch_size"] <= 0:
                raise ValueError("DINO batch_size must be positive")
        if self.extrinsics_head() and self.policy != "dino_language":
            raise ValueError("Extrinsics supervision requires the DINO/language preset")
        validate_settings(**self.curriculum.reset_sampling.params)

    def resolution(self):
        return self.scene.eye.foveated_res

    def cmf_a(self):
        return self.scene.eye.cmf_a

    def camera_position(self):
        return self.scene.eye.offset.pos

    def camera_hfov(self):
        camera = self.scene.eye.spawn
        return math.degrees(2 * math.atan(camera.horizontal_aperture / (2 * camera.focal_length)))

    def gaze_limit(self):
        return self.actions.gaze.angle_limit

    def gaze_max_delta(self):
        return self.actions.gaze.max_delta

    def renderer_backend(self):
        renderer = self.scene.eye.renderer_cfg
        renderer = renderer.default if isinstance(renderer, PresetCfg) else renderer
        return "ovrtx" if renderer.renderer_type == "ovrtx" else "warp"

    def render_mode(self):
        from .render_options import RENDER_MODES

        if self.renderer_backend() == "warp":
            return "rgb"
        outputs = self.scene.eye.data_types
        output = (outputs.default if isinstance(outputs, PresetCfg) else outputs)[0]
        renderer = self.scene.eye.renderer_cfg
        renderer = renderer.default if isinstance(renderer, PresetCfg) else renderer
        shadows = getattr(renderer, "enable_shadows", False)
        shadows = shadows.default if isinstance(shadows, PresetCfg) else shadows
        return next(
            name for name, value in RENDER_MODES.items() if value == (output, shadows if output != "rgb" else False)
        )

    def wrist_camera(self):
        spawn = self.scene.robot.spawn
        spawn = spawn.default if isinstance(spawn, PresetCfg) else spawn
        return "removed" if spawn.func == spawn_without_wrist_camera else "stock"

    def retinal_geometry(self):
        return self.scene.eye.field_geometry

    def extrinsics_head(self):
        return self.observations.extrinsics is not None

    def play_mode(self):
        super().play_mode()
        self.reset_sampling_evaluation = True
        self.curriculum.reset_sampling.params["mode"] = "uniform"
        if self.sim2real is not None:
            self.sim2real.ramp_steps = 0

    def enable_diagnostics(self):
        camera = self.scene.eye
        self.scene.diagnostic_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/DiagnosticCamera",
            data_types=list(camera.data_types),
            width=640,
            height=640,
            renderer_cfg=camera.renderer_cfg.copy(),
            render_settings=camera.render_settings.copy(),
            spawn=camera.spawn.copy(),
            offset=camera.offset.copy(),
        )


@configclass
class SO101NumberedEnvCfg(SO101FoveatedEnvCfg):
    """Match a vial's number to its rack slot using textured camera observations."""

    goal_mode: ClassVar[str] = "numbered"
    scene = NumberedSceneCfg(num_envs=128, env_spacing=0.9)
    actions = FoveatedActionsCfg(gaze=CachedGazeActionCfg(angle_limit=math.radians(38.0), max_delta=0.12))

    def __post_init__(self):
        super().__post_init__()
        from .goals import configure_numbered_scene

        configure_numbered_scene(self)


@configclass
class SO101LanguageEnvCfg(SO101NumberedEnvCfg):
    """Numbered placement using frozen DINO and language features."""

    policy: ClassVar[str] = "dino_language"
    observations = LanguageObservationsCfg()
    scene = NumberedSceneCfg(num_envs=128, env_spacing=0.9, eye=NumberedSceneCfg().eye.replace(foveated_res=128))


@configclass
class SO101PlanarEnvCfg(SO101LanguageEnvCfg):
    """Language policy with the same planar software retina used on the robot."""

    def __post_init__(self):
        super().__post_init__()
        from .sim2real import configure_planar_camera

        configure_planar_camera(self)


@configclass
class SO101Sim2RealEnvCfg(SO101PlanarEnvCfg):
    """Planar camera, physical randomization, and auxiliary camera-pose labels."""

    sim2real = CameraRandomizationCfg()

    def __post_init__(self):
        super().__post_init__()
        from .sim2real import ExtrinsicsObservations, configure

        configure(self)
        self.observations.extrinsics = ExtrinsicsObservations()


@configclass
class SO101EnvCfg(PresetCfg):
    """Registered task variants; shared names also select the matching agent."""

    nearest = SO101FoveatedEnvCfg()
    numbered = SO101NumberedEnvCfg()
    dino_language = SO101LanguageEnvCfg()
    planar = SO101PlanarEnvCfg()
    sim2real = SO101Sim2RealEnvCfg()
    default = numbered
