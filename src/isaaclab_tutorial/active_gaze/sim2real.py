"""Robot-transfer randomization of the active-gaze task: the scene camera's mount and the retina's appearance.

``presets=sim2real`` adds these to the tutorial's physical sim2real randomization. The camera's mount moves every
episode, and the ``extrinsics`` observation group labels that move for the actor's auxiliary head. Retinal images get
an episode-wise photometric variation.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .camera_geometry import pose_targets

PHOTOMETRIC_RANGES = dict(
    exposure=(0.75, 1.25),
    contrast=(0.85, 1.15),
    gamma=(0.85, 1.15),
    white_balance=(0.9, 1.1),
    brightness=(-0.05, 0.05),
    blur=(0.0, 0.5),
)
"""Ranges of the episode-wise photometric variation, as in the tutorial's wrist-camera task."""


class PhotometricVariation:
    """Episode-wise exposure, contrast, gamma, white balance, brightness and blur of uint8 retinal images.

    Samples the physical camera does not see are zero and stay zero.
    """

    def __init__(self, num_envs: int, device, ranges: dict[str, tuple[float, float]]):
        self.ranges = ranges
        self.values = {
            name: torch.zeros(num_envs, 3 if name == "white_balance" else 1, device=device) for name in ranges
        }
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name, (low, high) in self.ranges.items():
            self.values[name][ids] = torch.empty_like(self.values[name][ids]).uniform_(low, high)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Vary (num_envs, H, W, 3) uint8 images."""
        unseen = (images[..., :3] == 0).all(-1, keepdim=True)
        rgb = images[..., :3].permute(0, 3, 1, 2).float() / 255

        def value(name):
            return self.values[name][:, :, None, None]

        blurred = F.avg_pool2d(rgb, 3, stride=1, padding=1, count_include_pad=False)
        rgb = rgb.lerp(blurred, value("blur")).pow(value("gamma"))
        rgb = (rgb - 0.5) * value("contrast") + 0.5
        rgb = (rgb * value("exposure") * value("white_balance") + value("brightness")).clamp(0, 1)
        return (rgb.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).masked_fill(unseen, 0)


class randomize_camera_mount(ManagerTermBase):
    """Move the scene camera's mount by up to ``position_range`` [m] and ``rotation_range_deg`` [deg] each episode.

    A fraction ``nominal_fraction`` of episodes keep the nominal mount. The ranges ramp up from 10 % over
    ``ramp_steps`` control steps. This term owns the camera's mount pose and the move's extrinsics labels. The gaze
    stays relative to the moved mount, and a gaze action with ``mount="tracked"`` aims its teacher from it.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene[cfg.params["camera_name"]]
        offset = self.camera.cfg.offset
        self.nominal_position = torch.tensor(offset.pos, device=self.device).expand(self.num_envs, -1).clone()
        """Nominal mount position in the environment frame [m]."""
        self.nominal_rotation = torch.tensor(offset.rot, device=self.device).expand(self.num_envs, -1).clone()
        """Nominal mount orientation (x, y, z, w), OpenGL convention."""
        self.position = self.nominal_position.clone()
        self.rotation = self.nominal_rotation.clone()
        self.labels = torch.zeros(self.num_envs, 9, device=self.device)
        """The move of each mount in the robot base frame, scaled for the extrinsics head, shape (num_envs, 9)."""

    def strength(self, ramp_steps: int) -> float:
        return min(1.0, max(0.1, self._env.common_step_counter / ramp_steps)) if ramp_steps else 1.0

    def __call__(
        self,
        env,
        env_ids,
        camera_name: str = "eye",
        position_range: float = 0.03,
        rotation_range_deg: float = 7.0,
        nominal_fraction: float = 0.2,
        ramp_steps: int = 30000,
    ):
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        if not len(ids):
            return
        strength = self.strength(ramp_steps)
        env.extras.setdefault("log", {})["Randomization/camera_strength"] = strength
        moved = (torch.rand(len(ids), device=self.device) >= nominal_fraction).float()[:, None] * strength

        def uniform(bound):
            return torch.empty(len(ids), 3, device=self.device).uniform_(-bound, bound) * moved

        delta = uniform(position_range)
        perturb = quat_from_euler_xyz(*uniform(math.radians(rotation_range_deg)).T)
        self.position[ids] = self.nominal_position[ids] + delta
        self.rotation[ids] = quat_mul(self.nominal_rotation[ids], perturb)
        robot_rotation = env.scene["robot"].data.root_quat_w.torch[ids]
        # Translation in the robot base frame, rotation relative to the nominal mount; no environment-grid offset.
        self.labels[ids] = pose_targets(quat_apply_inverse(robot_rotation, delta), matrix_from_quat(perturb))
        world = self.position[ids] + env.scene.env_origins[ids]
        self.camera.set_world_poses(world, self.rotation[ids], env_ids=ids, convention="opengl")


def mount_randomization(env) -> randomize_camera_mount | None:
    """The episode's camera-mount randomization, or None when the camera stays on its nominal mount."""
    for name in env.event_manager.active_terms.get("reset", []):
        term = env.event_manager.get_term_cfg(name).func
        if isinstance(term, randomize_camera_mount):
            return term
    return None


def extrinsics_labels(env) -> torch.Tensor:
    """Labels of the camera mount's move for the actor's extrinsics head, shape (num_envs, 9)."""
    term = mount_randomization(env)
    return term.labels.clone() if term is not None else torch.zeros(env.num_envs, 9, device=env.device)


@configclass
class ExtrinsicsObservations(ObservationGroupCfg):
    """Training labels of the camera mount's move; never part of the actor's observations."""

    pose = ObservationTermCfg(func=extrinsics_labels)
    enable_corruption = False
