"""Opt-in physics and appearance randomizations and uncalibrated planar software gaze."""

import math
from copy import deepcopy

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
)

from isaaclab_fovi.deployment.so101.planar import PlanarRetina, gaze_basis, project_pinhole_rays
from isaaclab_tutorial.active_gaze.learning.observations import DinoTokens
from isaaclab_fovi.sensors.calibrated_geometry import compose_rotations, rotate_camera_rays
from isaaclab_fovi.sensors.foveated_rtx_tiled_camera import FoveatedRTXTiledCamera

from .camera_geometry import lens_visibility, pose_targets
from .transfer_options import settings


class PlanarSceneCamera(FoveatedRTXTiledCamera):
    """Direct RTX foveation with one shared planar LUT and per-scene poses."""

    def _initialize_impl(self):
        super()._initialize_impl()
        position, orientation = self._view.get_world_poses()
        self.physical_position = position.torch.clone()
        self.physical_rotation = orientation.torch.clone()
        self.gaze = torch.zeros(len(position.torch), 2, device=self.device)
        self.retina = PlanarRetina(self._camera_hfov_deg, self.cfg.cmf_a, self.cfg.foveated_res).to(self.device)
        self._physical_rays = self._fixed_fov_ray_directions * self.gaze.new_tensor((1, -1, -1))

    def set_calibrated_angles(self, angles):
        # Keep the shared action API. These are software angles using a fixed
        # nominal pinhole convention, not measured physical camera intrinsics.
        self.gaze = angles.clone()
        self._write_gaze_pose()

    def set_physical_poses(self, position, rotation, env_ids):
        self.physical_position[env_ids] = position
        self.physical_rotation[env_ids] = rotation
        self._write_gaze_pose(env_ids)

    def _write_gaze_pose(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        basis = self.gaze.new_tensor((1, -1, -1))
        relative = gaze_basis(self.gaze[ids]) * basis[None, :, None] * basis[None, None, :]
        rotation = quat_from_matrix(compose_rotations(matrix_from_quat(self.physical_rotation[ids]), relative))
        Camera.set_world_poses(self, self.physical_position[ids], rotation, env_ids=env_ids, convention="opengl")

    def get_world_positions(self):
        return self.physical_position

    def _apply_fixed_fov_mask(self):
        rotated = rotate_camera_rays(self._physical_rays, gaze_basis(self.gaze))
        _, valid = project_pinhole_rays(
            rotated,
            self.retina.focal_pixels,
            self.retina.center_pixels,
            self.retina.width,
        )
        for name, output in self._foveated_output.items():
            if name == "rgb" and "rgba" in self._foveated_output:
                continue
            output.masked_fill_(~valid[:, :, None, None], 0)
        self._fixed_fov_valid_fraction = valid.float().mean(1)
        valid = valid & self._ray_coords.valid_mask
        self.public_valid = valid[:, self._grid_indices].reshape(-1, self.cfg.foveated_res, self.cfg.foveated_res, 1)


class RandomizedDinoTokens(DinoTokens):
    def __call__(self, env, reference, checksum, batch_size=128, data_type="rgb"):
        if not getattr(env, "refresh_policy_vision", True) and self.cached is not None:
            return self.cached
        image = env.scene["eye"].data.output[data_type]
        image = image.torch if hasattr(image, "torch") else image
        image = env.so101_camera_randomization.image(image)
        self.cached = self.encoder(image)
        return self.cached


class CameraRandomization(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene["eye"]
        self.camera.gaze = torch.zeros(self.num_envs, 2, device=self.device)
        self.retina = PlanarRetina(env.cfg.camera_hfov(), env.cfg.cmf_a(), env.cfg.resolution()).to(self.device)
        self.options = settings(env.cfg.sim2real if env.cfg.sim2real is not None else {"nominal_fraction": 1.0})
        self.position = torch.tensor(self.camera.cfg.offset.pos, device=self.device).expand(self.num_envs, -1).clone()
        self.rotation = torch.tensor(self.camera.cfg.offset.rot, device=self.device).expand(self.num_envs, -1).clone()
        self.nominal_position, self.nominal_rotation = self.position.clone(), self.rotation.clone()
        self.intrinsics = (
            torch.tensor(self.camera.cfg.camera_calibration["intrinsics"], device=self.device)
            .expand(self.num_envs, -1)
            .clone()
        )
        self.nominal_intrinsics = self.intrinsics.clone()
        self.radial = torch.zeros(self.num_envs, device=self.device)
        self.targets = torch.zeros(self.num_envs, 9, device=self.device)
        self.nominal = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.photo = torch.ones(self.num_envs, 7, device=self.device)
        self.photo[:, 5:] = 0
        env.so101_camera_randomization = self

    def __call__(self, env, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        n = len(ids)
        if n == 0:
            return
        cfg = self.options
        strength = self.strength
        env.extras.setdefault("log", {})["Randomization/camera_strength"] = strength
        nominal = torch.rand(n, device=self.device) < cfg["nominal_fraction"]
        self.nominal[ids] = nominal
        active = (~nominal).float()[:, None]

        def uniform(width, bound):
            return torch.empty(n, width, device=self.device).uniform_(-bound, bound) * active * strength

        delta = uniform(3, cfg["position_range"])
        perturb = quat_from_euler_xyz(*uniform(3, math.radians(cfg["rotation_deg"])).T)
        self.position[ids] = self.nominal_position[ids] + delta
        self.rotation[ids] = quat_mul(self.nominal_rotation[ids], perturb)
        robot_rotation = env.scene["robot"].data.root_quat_w.torch[ids]
        # Translation in the robot-base frame; rotation relative to its nominal
        # camera mount. Neither label includes environment-grid coordinates.
        self.targets[ids] = pose_targets(quat_apply_inverse(robot_rotation, delta), matrix_from_quat(perturb))
        self.camera.set_physical_poses(self.position[ids] + env.scene.env_origins[ids], self.rotation[ids], env_ids=ids)
        self.camera.reset(env_ids=ids)
        if "diagnostic_camera" in env.scene.sensors:
            diagnostic = env.scene["diagnostic_camera"]
            diagnostic.set_world_poses(
                self.position[ids] + env.scene.env_origins[ids],
                self.rotation[ids],
                env_ids=ids,
                convention="opengl",
            )
            diagnostic.reset(env_ids=ids)
        # Tutorial photometric families, held fixed for the whole episode.
        # exposure, contrast, gamma, white-balance R/B, brightness, blur mix.
        for index, (lo, hi) in enumerate(
            (
                (0.75, 1.25),
                (0.85, 1.15),
                (0.85, 1.15),
                (0.9, 1.1),
                (0.9, 1.1),
                (-0.05, 0.05),
                (0, 0.5),
            )
        ):
            base = 0.0 if index >= 5 else 1.0
            self.photo[ids, index] = (
                base + (torch.empty(n, device=self.device).uniform_(lo, hi) - base) * active[:, 0] * strength
            )

    def visibility(self, directions):
        return lens_visibility(directions[:, None], self.intrinsics, self.radial, (640, 640)).squeeze(-1)

    def gaze_direction(self, directions, ids=None):
        """True scene ray -> physical pixel -> uncalibrated action coordinates."""
        selection = slice(None) if ids is None else ids
        pixels = (
            directions[:, :2] / directions[:, 2:].clamp_min(1e-7) * self.intrinsics[selection, :2]
            + self.intrinsics[selection, 2:]
        )
        return self.retina.angles(pixels)

    @property
    def strength(self):
        ramp = self.options["ramp_steps"]
        return min(1.0, max(0.1, self._env.common_step_counter / ramp)) if ramp else 1.0

    def image(self, images):
        from torch.nn import functional as F

        valid = self.camera.public_valid
        rgb = images[..., :3].permute(0, 3, 1, 2).float() / 255
        values = self.photo[:, :, None, None]
        blurred = F.avg_pool2d(rgb, 3, stride=1, padding=1, count_include_pad=False)
        rgb = rgb.lerp(blurred, values[:, 6:7]).pow(values[:, 2:3])
        rgb = (rgb - 0.5) * values[:, 1:2] + 0.5
        white = torch.cat((values[:, 3:4], torch.ones_like(values[:, :1]), values[:, 4:5]), 1)
        rgb = (rgb * values[:, :1] * white + values[:, 5:6]).clamp(0, 1)
        # Preserve the physical camera/retinal footprint after color transforms.
        return (rgb.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).masked_fill(~valid, 0)


def extrinsics_labels(env):
    state = getattr(env, "so101_camera_randomization", None)
    if state is None:
        return torch.zeros(env.num_envs, 9, device=env.device)
    return state.targets.clone()


@configclass
class ExtrinsicsObservations(ObservationGroupCfg):
    pose = ObservationTermCfg(func=extrinsics_labels)
    enable_corruption = False


def configure(cfg):
    """Apply physical and appearance randomization while retaining numbered SGS resets."""
    from isaaclab_tutorial.active_gaze.randomization_cfg import (
        AppearanceRandomizationEventsCfg,
        RandomizedSceneAssetsCfg,
    )

    # Keep the calibrated geometry, controller scales, and mid-task reset
    # distribution used by the pretrained actor.
    reset = deepcopy(cfg.events.reset_from_dataset)
    events = AppearanceRandomizationEventsCfg()
    reset.params.update(
        {
            key: value
            for key, value in events.reset_from_dataset.params.items()
            if key not in ("dataset_path", "phase_weights", "sequential")
        }
    )
    events.reset_from_dataset = reset
    cfg.events = events
    appearance = RandomizedSceneAssetsCfg()
    cfg.scene.support = appearance.support
    cfg.scene.fixed_jaw_support = appearance.fixed_jaw_support
    cfg.scene.moving_jaw_support = appearance.moving_jaw_support
    for name in (
        "robot_visual",
        "desk_visual",
        "rack_visual",
        "vial_body_visual",
        "vial_cap_visual",
    ):
        setattr(cfg.scene, name, deepcopy(getattr(appearance, name)))
    for spawn in cfg.scene.vial.spawn.assets_cfg:
        spawn.randomize_diameters = True
    configure_planar_camera(cfg)


def configure_planar_camera(cfg):
    cfg.events.scene_camera = EventTermCfg(func=CameraRandomization, mode="reset")
    cfg.scene.eye.class_type = PlanarSceneCamera
    cfg.scene.eye.field_geometry = "planar"
    cfg.scene.eye.update_latest_camera_pose = True
    cfg.observations.vision.tokens.func = RandomizedDinoTokens
