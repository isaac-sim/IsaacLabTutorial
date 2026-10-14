"""Reward terms: dense approach and transport shaping, one-time milestones, and success."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .geometry import hole_relative_positions, symmetric_axial_keypoint_error
from .progress import placement_progress
from .task import VIAL_AXIS_MAX, VIAL_AXIS_MIN, contact_force, grasp_center_w, vial_grasp_point_w, vial_in_rack

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

HELD_INSERTION_TARGET = (0.0, 0.0, 0.060)
"""Rack-local hole-relative root position at the end of a held insertion; gravity finishes the travel."""


class approach_progress(ManagerTermBase):
    """Pay the decrease in pad-centre-to-cap distance, in units of ``scale``, until the vial is grasped.

    A difference of potentials has no incentive to hover, and gives the same gradient 25 cm from the vial as 2 cm
    away, which is what lets a pixels-only policy discover the approach from the home pose.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_distance[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01, height_offset: float = 0.0) -> torch.Tensor:
        # Raising the target above the cap favours a shallower pinch; grasp milestones still need physical proof.
        target = vial_grasp_point_w(env)
        target = target + target.new_tensor((0.0, 0.0, height_offset))
        distance = torch.linalg.vector_norm(grasp_center_w(env) - target, dim=-1)
        progress = ((self.previous_distance - distance) / scale).clamp(-1.0, 1.0)
        self.previous_distance = distance
        # The first step after a reset has no previous distance (NaN) and pays nothing.
        return torch.where(placement_progress(env).grasped, 0.0, progress.nan_to_num(0.0))


def held_goal(env: ManagerBasedRLEnv, std: float = 0.10) -> torch.Tensor:
    """Bring the held vial to the insertion pose above the closest hole, until it is inserted.

    There is deliberately no equivalent bump around the vial before the grasp: a policy whose grasp is still
    unreliable learns to hover in it instead of attempting the grasp.
    """
    position, axis, _ = vial_in_rack(env)
    error = symmetric_axial_keypoint_error(
        hole_relative_positions(position),
        axis.unsqueeze(1),
        position.new_tensor(HELD_INSERTION_TARGET),
        position.new_tensor((0.0, 0.0, 1.0)),
        VIAL_AXIS_MIN,
        VIAL_AXIS_MAX,
    ).amin(dim=-1)
    progress = placement_progress(env)
    return torch.where(progress.grasped & ~progress.inserted, torch.exp(-error / std), 0.0)


class milestones(ManagerTermBase):
    """Pay grasp, lift and insertion once per episode, in the ratio 1:2:4.

    Milestones present on the first step after a reset (restored from the reset dataset) are not paid.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        progress = placement_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted, progress.inserted), dim=-1)
        reward = ((reached & ~self.paid).float() * reached.new_tensor((1.0, 2.0, 4.0), dtype=torch.float32)).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step confirmed placement success."""
    return placement_progress(env).success.float()


def jaw_support_force(env: ManagerBasedRLEnv, free_force: float = 1.0, force_span: float = 5.0) -> torch.Tensor:
    """Penalize either jaw pushing on the desk or mat beyond ``free_force`` [N], saturating after ``force_span``."""
    force = torch.maximum(contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support"))
    return ((force - free_force) / force_span).clamp(0.0, 1.0)
