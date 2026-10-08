"""Small camera-calibration variations around the known WowRobo wrist camera."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

from .events import _ids


class RandomizeWristCameraMount(ManagerTermBase):
    """Sample an episode-fixed camera pose relative to its carrier, without accumulating offsets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._nominal_position = None
        self._nominal_orientation = None
        for name in ("position_range", "rotation_range"):
            value = cfg.params[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def __call__(self, env, env_ids, sensor_cfg, asset_cfg, position_range: float, rotation_range: float):
        """Perturb translation [m] and XYZ Euler angles [rad] in the nominal optical frame."""
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        camera = env.scene.sensors[sensor_cfg.name]
        robot = env.scene[asset_cfg.name]
        # Dataset resets just wrote joint positions. Refresh kinematics before reading the camera/carrier.
        env.sim.forward()
        carrier_position = robot.data.body_pos_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, 3)
        carrier_orientation = robot.data.body_quat_w.torch[:, asset_cfg.body_ids].reshape(env.num_envs, 4)
        if self._nominal_position is None:
            self._nominal_position, self._nominal_orientation = subtract_frame_transforms(
                carrier_position, carrier_orientation, camera.data.pos_w.torch, camera.data.quat_w_ros.torch
            )
        translation = torch.empty((len(ids), 3), device=env.device).uniform_(-position_range, position_range)
        angles = torch.empty_like(translation).uniform_(-rotation_range, rotation_range)
        rotation = quat_from_euler_xyz(*angles.unbind(-1))
        local_position, local_orientation = combine_frame_transforms(
            self._nominal_position[ids], self._nominal_orientation[ids], translation, rotation
        )
        position, orientation = combine_frame_transforms(
            carrier_position[ids], carrier_orientation[ids], local_position, local_orientation
        )
        camera.set_world_poses(position, orientation, env_ids=ids, convention="ros")
        # Reading camera.data above may have rendered the old pose already. Invalidate those
        # images so the first observation of the new episode uses the sampled mount as well.
        camera.reset(env_ids=ids)


def camera_sampling_grid(
    focal_scale: torch.Tensor,
    principal_offset: torch.Tensor,
    radial_distortion: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    focal_length_pixels: float,
) -> torch.Tensor:
    """Map a varied pinhole/Brown-k1 projection into a centered, overscanned pinhole image.

    Offsets are in output pixels; focal scale is independent for x/y. Pixel centers use
    ``align_corners=False``. The small k1 distortion is inverted iteratively, including its
    effect on both axes. Overscan supplies real rendered pixels outside the nominal view.
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
