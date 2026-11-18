"""Sample a real camera's images the way the simulator's foveated camera samples the scene."""

import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from isaaclab_fovi.sensors.native_grid import GRID_STYLE, native_grid_directions
from torch import nn
from torch.nn import functional as F


class SoftwareRetina(nn.Module):
    """Bilinear sampling of dense images on the simulator's retinal ray grid, in its upright image order.

    The retina is the trained one: its field of view and CMF constant [deg] and its resolution. ``camera_model``
    describes the real camera's capture raster as :class:`fovi.sensing.projection.CameraModel` fields, without any
    implicit resizing. Inputs are uint8 (B, H, W, 3) RGB images and gaze (tilt, pan) [deg]; the output is uint8
    (B, R, R, 3), zero where the real camera does not see.
    """

    def __init__(self, camera_model: dict, fov_deg: float, cmf_a_deg: float, resolution: int):
        super().__init__()
        self.camera = CameraModel(**camera_model)
        self.height, self.width = self.camera.image_size
        self.resolution = resolution
        coords = SamplingCoords(
            fov=fov_deg, cmf_a=cmf_a_deg, res=resolution, style=GRID_STYLE, field_geometry="spherical", device="cpu"
        )
        rays = native_grid_directions(coords)
        # Renderer OpenGL rays -> upright OpenCV rays, then public image ordering.
        rays = coords.as_grid(rays * rays.new_tensor((1, -1, -1)), sample_dim=0)
        self.register_buffer("rays", rays.reshape(-1, 3))
        self.register_buffer("valid", coords.as_grid(coords.valid_mask).reshape(1, resolution, resolution, 1))

    def grid(self, gaze_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tilt, pan = gaze_deg.float().deg2rad().unbind(-1)
        direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
        rotation = gaze_rotation(direction, "pan_tilt")
        # Explicit reductions avoid TF32 errors in camera geometry.
        rays = (self.rays[None, :, None, :] * rotation[:, None, :, :]).sum(-1)
        pixels, visible = self.camera.project(rays)
        extent = pixels.new_tensor((self.width, self.height))
        grid = (2 * (pixels + 0.5) / extent - 1).reshape(-1, self.resolution, self.resolution, 2)
        visible = visible.reshape(-1, self.resolution, self.resolution, 1) & self.valid
        return grid, visible

    def forward(self, rgb: torch.Tensor, gaze_deg: torch.Tensor) -> torch.Tensor:
        grid, valid = self.grid(gaze_deg)
        sampled = F.grid_sample(
            rgb.permute(0, 3, 1, 2).float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).masked_fill(~valid, 0).round().clamp(0, 255).to(torch.uint8)
