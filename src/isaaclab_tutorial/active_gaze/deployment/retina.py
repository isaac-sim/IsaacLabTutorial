"""Sample real RGB images using the simulator's calibrated, upright ray grid."""

import math

import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from torch import nn
from torch.nn import functional as F

from isaaclab_fovi.sensors.native_grid import native_grid_directions


def nominal_calibration(hfov: float, size: int = 640) -> dict:
    focal = size / (2 * math.tan(math.radians(hfov / 2)))
    return dict(
        model="pinhole",
        image_size=[size, size],
        intrinsics=[focal, focal, (size - 1) / 2, (size - 1) / 2],
    )


class SoftwareRetina(nn.Module):
    """Bilinear dense-image sampling with the exact native-grid permutation.

    Calibration describes the actual capture raster, without implicit resizing.
    Angular coverage and CMF describe the trained retina independently of it.
    Inputs are uint8 BHWC RGB and (tilt, pan) radians. Output is uint8 BHWC.
    """

    def __init__(self, calibration: dict, hfov: float, cmf_a: float, resolution: int = 128):
        super().__init__()
        self.camera = CameraModel(**calibration)
        if self.camera.model != "pinhole":
            raise ValueError("SO-101 deployment requires a calibrated pinhole camera")
        self.height, self.width = self.camera.image_size
        self.resolution = resolution
        coords = SamplingCoords(
            fov=hfov,
            cmf_a=cmf_a * hfov,
            res=resolution,
            style="warped_cartesian_as_grid",
            fov_type="circular",
            radius_norm=2.0,
            field_geometry="spherical",
            device="cpu",
        )
        rays = native_grid_directions(coords, tangent_scale=1.0)
        # Renderer OpenGL rays -> upright OpenCV rays, then public image ordering.
        rays = coords.as_grid(rays * rays.new_tensor((1, -1, -1)), sample_dim=0)
        self.register_buffer("rays", rays.reshape(-1, 3))
        self.register_buffer("valid", coords.as_grid(coords.valid_mask).reshape(1, resolution, resolution, 1))

    def grid(self, gaze: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tilt, pan = gaze.float().unbind(-1)
        direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
        rotation = gaze_rotation(direction, "pan_tilt")
        # Explicit reductions avoid TF32 errors in camera geometry.
        rays = (self.rays[None, :, None, :] * rotation[:, None, :, :]).sum(-1)
        pixels, visible = self.camera.project(rays)
        extent = pixels.new_tensor((self.width, self.height))
        grid = (2 * (pixels + 0.5) / extent - 1).reshape(-1, self.resolution, self.resolution, 2)
        visible = visible.reshape(-1, self.resolution, self.resolution, 1) & self.valid
        return grid, visible

    def forward(self, rgb: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        grid, valid = self.grid(gaze)
        sampled = F.grid_sample(
            rgb.permute(0, 3, 1, 2).float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).masked_fill(~valid, 0).round().clamp(0, 255).to(torch.uint8)
