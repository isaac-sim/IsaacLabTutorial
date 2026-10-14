"""Observation terms: task-specific state features and the randomized wrist camera image."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.images import CameraFrameStack

from .progress import placement_progress
from .task import jaw_contacts, nearest_hole_distance, vial_in_rack, vial_upright

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint-position targets currently sent to the drives [rad]."""
    return env.scene[asset_cfg.name].data.joint_pos_target.torch[:, asset_cfg.joint_ids]


def encoder_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions as the robot's encoders report them, including the sampled calibration bias [rad]."""
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids] + getattr(
        env, "_so101_encoder_bias", 0.0
    )


def encoder_joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint targets in the same (biased) encoder coordinates as :func:`encoder_joint_pos` [rad]."""
    return joint_target(env, asset_cfg) + getattr(env, "_so101_encoder_bias", 0.0)


def vial_in_rack_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vial root position in the rack frame [m]."""
    return vial_in_rack(env)[0]


def placement_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance to the nearest hole, lowest vial point above the rack, uprightness, and vial speed."""
    position, _, lowest = vial_in_rack(env)
    speed = torch.linalg.vector_norm(env.scene["vial"].data.root_lin_vel_w.torch, dim=-1)
    return torch.stack((nearest_hole_distance(position), lowest, vial_upright(env), speed), dim=-1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Latched grasp, lift and insertion milestones, which make the once-per-episode rewards Markov."""
    progress = placement_progress(env)
    return torch.stack((progress.grasped, progress.lifted, progress.inserted), dim=-1).float()


def jaw_contact_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged fixed-jaw, moving-jaw and bilateral vial contact."""
    fixed, moving = jaw_contacts(env)
    return torch.stack((fixed, moving, fixed & moving), dim=-1).float()


def camera_sampling_grid(
    focal_scale: torch.Tensor,
    principal_offset: torch.Tensor,
    radial_distortion: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    focal_length_pixels: float,
) -> torch.Tensor:
    """``grid_sample`` grid that renders a varied pinhole/Brown-k1 camera from a centred, overscanned pinhole image.

    Offsets are in output pixels; focal scale is independent for x/y. The small k1 distortion is inverted
    iteratively. Overscan supplies real rendered pixels outside the nominal view.
    """
    height, width = output_size
    source_height, source_width = input_size
    y, x = torch.meshgrid(
        torch.arange(height, device=focal_scale.device, dtype=focal_scale.dtype) + 0.5 - height / 2,
        torch.arange(width, device=focal_scale.device, dtype=focal_scale.dtype) + 0.5 - width / 2,
        indexing="ij",
    )
    pixels = torch.stack((x, y), dim=-1)[None]
    distorted = (pixels - principal_offset[:, None, None]) / (focal_length_pixels * focal_scale[:, None, None])
    undistorted = distorted
    for _ in range(5):
        radius_squared = undistorted.square().sum(-1, keepdim=True)
        undistorted = distorted / (1 + radial_distortion[:, None, None, None] * radius_squared)
    source_pixels = undistorted * focal_length_pixels
    return source_pixels * source_pixels.new_tensor((2 / source_width, 2 / source_height))


class wrist_rgb(ManagerTermBase):
    """Wrist RGB in [0, 1], channel-first, with per-episode camera intrinsics and photometric variation.

    The overscanned render is resampled to ``projection_size`` through a camera whose focal scale, principal point
    and radial distortion are drawn per episode. While the group's corruption is enabled (training), gamma,
    contrast, exposure, white balance, brightness and blur are applied with per-episode values. The last
    ``history_length`` frames are stacked oldest first; after a reset the history repeats the first frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        camera = env.scene.sensors[params["sensor_cfg"].name]
        self.input_size = (camera.cfg.height, camera.cfg.width)
        n = self.num_envs
        self.focal_scale = torch.ones(n, 2, device=self.device)
        self.principal_offset = torch.zeros(n, 2, device=self.device)
        self.radial_distortion = torch.zeros(n, device=self.device)
        self.gamma, self.contrast, self.exposure, self.brightness, self.blur = (
            torch.zeros(n, 1, 1, 1, device=self.device) for _ in range(5)
        )
        self.white_balance = torch.zeros(n, 3, 1, 1, device=self.device)
        self.grid = torch.zeros(n, *params["projection_size"], 2, device=self.device)
        self.frames = CameraFrameStack(n, self.device, params.get("history_length", 1), channel_first=True)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        params = self.cfg.params
        principal = params.get("principal_point_pixels", 0.0)
        for tensor, value_range in (
            (self.focal_scale, params.get("focal_scale_range", (1.0, 1.0))),
            (self.principal_offset, (-principal, principal)),
            (self.radial_distortion, params.get("radial_distortion_range", (0.0, 0.0))),
            (self.gamma, params.get("gamma_range", (1.0, 1.0))),
            (self.contrast, params["contrast_range"]),
            (self.exposure, params["exposure_range"]),
            (self.white_balance, params["white_balance_range"]),
            (self.brightness, params["brightness_range"]),
            (self.blur, params.get("blur_range", (0.0, 0.0))),
        ):
            tensor[ids] = torch.empty_like(tensor[ids]).uniform_(*value_range)
        self.grid[ids] = camera_sampling_grid(
            self.focal_scale[ids],
            self.principal_offset[ids],
            self.radial_distortion[ids],
            input_size=self.input_size,
            output_size=params["projection_size"],
            focal_length_pixels=params["focal_length_pixels"],
        )
        self.frames.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        projection_size: tuple[int, int],
        focal_length_pixels: float,
        exposure_range: tuple[float, float],
        contrast_range: tuple[float, float],
        white_balance_range: tuple[float, float],
        brightness_range: tuple[float, float],
        gamma_range: tuple[float, float] = (1.0, 1.0),
        blur_range: tuple[float, float] = (0.0, 0.0),
        focal_scale_range: tuple[float, float] = (1.0, 1.0),
        principal_point_pixels: float = 0.0,
        radial_distortion_range: tuple[float, float] = (0.0, 0.0),
        history_length: int = 1,
    ) -> torch.Tensor:
        image = env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]
        image = image.permute(0, 3, 1, 2).float() / 255.0
        image = torch.nn.functional.grid_sample(image, self.grid, padding_mode="border", align_corners=False)
        if env.cfg.observations.wrist_rgb.enable_corruption:
            blurred = torch.nn.functional.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
            image = image.lerp(blurred, self.blur).pow(self.gamma)
            image = (image - 0.5) * self.contrast + 0.5
            image = (image * self.exposure * self.white_balance + self.brightness).clamp(0.0, 1.0)
        return self.frames(image.permute(0, 2, 3, 1))
