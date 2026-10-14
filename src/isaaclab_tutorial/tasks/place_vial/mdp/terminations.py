"""Termination terms: confirmed placement success and a dropped vial.

Out-of-workspace vials and unstable robots use the lift task's ``out_of_bound`` and ``abnormal_robot_state``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .geometry import hole_relative_positions, inside_bounds
from .progress import placement_progress
from .task import (
    HARD_RACK_IMPACT_FORCE,
    INSERTION_RADIUS,
    RACK_CLEARANCE_HEIGHT,
    RACK_RIM_HEIGHT,
    SEATED_LOWER,
    SEATED_UPPER,
    UPRIGHT_ALIGNMENT,
    VIAL_REST_HEIGHT,
    contact_force,
    gripper_openness,
    jaw_contacts,
    nearest_hole_distance,
    vial_height,
    vial_in_rack,
    vial_upright,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class placement_success(ManagerTermBase):
    """Update the task milestones every step and terminate once the vial is seated in any hole.

    Milestones a dataset reset starts with (for example a held vial) are restored on reset. The term also records the
    terminal statistics read by :mod:`isaaclab_tutorial.utils.evaluation`.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = placement_progress(env)
        self.max_rack_force = torch.zeros(self.num_envs, device=self.device)
        self.max_support_force = torch.zeros_like(self.max_rack_force)
        self.support_contact_time = torch.zeros_like(self.max_rack_force)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.progress.reset(ids)
        self.max_rack_force[ids] = 0.0
        self.max_support_force[ids] = 0.0
        self.support_contact_time[ids] = 0.0
        if hasattr(self._env, "_so101_reset_grasped"):
            grasped = self._env._so101_reset_grasped[ids]
            self.progress.grasped[ids] = grasped
            self.progress.grasp_count[ids] = torch.where(grasped, self.progress.grasp_steps, 0)
            self.progress.lifted[ids] = self._env._so101_reset_lifted[ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        position, _, lowest = vial_in_rack(env)
        upright = vial_upright(env) > UPRIGHT_ALIGNMENT
        fixed, moving = jaw_contacts(env)
        support_height = getattr(env, "_so101_support_height", 0.0)
        holding = fixed & moving & (vial_height(env) > VIAL_REST_HEIGHT + 0.006 + support_height)
        cleared = lowest >= RACK_CLEARANCE_HEIGHT
        inserted = (nearest_hole_distance(position) < INSERTION_RADIUS) & (lowest < RACK_RIM_HEIGHT) & upright
        vial = env.scene["vial"]
        still = (torch.linalg.vector_norm(vial.data.root_lin_vel_w.torch, dim=-1) < 0.06) & (
            torch.linalg.vector_norm(vial.data.root_ang_vel_w.torch, dim=-1) < 0.8
        )
        released = ~(fixed | moving) & (gripper_openness(env) > 0.2)
        placed = inside_bounds(hole_relative_positions(position), SEATED_LOWER, SEATED_UPPER).any(dim=-1)
        rack_force = contact_force(env, "vial_rack_contact")
        self.max_rack_force = torch.maximum(self.max_rack_force, rack_force)
        self.progress.unsafe_rack_contact |= rack_force > HARD_RACK_IMPACT_FORCE
        success = self.progress.update(
            holding, cleared, inserted, placed & upright & released & still, env.episode_length_buf
        )

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/insertion_rate"] = self.progress.inserted.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Safety/unsafe_rack_contact_rate"] = self.progress.unsafe_rack_contact.float().mean()
        if "fixed_jaw_support" in env.scene.sensors:
            support_force = torch.maximum(
                contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support")
            )
            self.max_support_force = torch.maximum(self.max_support_force, support_force)
            self.support_contact_time += (support_force > 1.0).float() * env.step_dt
            log["Safety/peak_jaw_support_force_n"] = self.max_support_force.mean()
            env._so101_terminal_max_support_force = self.max_support_force.clone()
            env._so101_terminal_support_contact_time = self.support_contact_time.clone()

        # Terminal statistics survive the reset that follows, so evaluation can read them.
        env._so101_terminal_progress = torch.stack(
            (self.progress.grasped, self.progress.lifted, self.progress.inserted, self.progress.unsafe_rack_contact),
            dim=-1,
        )
        env._so101_terminal_max_rack_force = self.max_rack_force.clone()
        env._so101_terminal_time_to_success_s = self.progress.time_to_success.float() * env.step_dt
        nearest_hole = hole_relative_positions(position)[..., :2].square().sum(dim=-1).argmin(dim=-1)
        env._so101_terminal_hole = torch.where(success, nearest_hole, -1)
        return success


def vial_dropped(env: ManagerBasedRLEnv, height_above_robot: float = 0.004) -> torch.Tensor:
    """A vial that was lifted has fallen back to the desk."""
    robot_height = env.scene["robot"].data.root_link_pos_w.torch[:, 2]
    fallen = env.scene["vial"].data.root_pos_w.torch[:, 2] - robot_height < height_above_robot
    return placement_progress(env).lifted & fallen
