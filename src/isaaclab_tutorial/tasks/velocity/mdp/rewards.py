# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance and head-pose rewards for MicroDuck."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import body_projected_gravity_b
from isaaclab.envs.mdp import joint_pos_limits as _joint_pos_limits
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .observations import joint_pos_rel_biased, zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def track_linear_velocity_heading(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track horizontal root-link velocity in the heading frame with tolerance ``std`` [m/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.torch), asset.data.root_link_lin_vel_w.torch
    )
    error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return zero_non_finite(torch.exp(-error / std**2))


def track_angular_velocity_world(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track world-frame yaw rate with tolerance ``std`` [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w.torch[:, 2])
    return zero_non_finite(torch.exp(-error / std**2))


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward keeping one body's frame aligned with gravity, using a Gaussian kernel."""
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    xy_squared = torch.nan_to_num(xy_squared, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy_squared / std**2)


def head_pose_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    backlash_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Track head encoder offsets with tolerance ``std`` [rad], including play when configured."""
    command = env.command_manager.get_command(command_name)
    error = joint_pos_rel_biased(env, asset_cfg, biased=False, backlash_cfg=backlash_cfg) - command
    return zero_non_finite(torch.exp(-((error / std) ** 2)).mean(dim=-1))


def joint_pos_limits_finite(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """:func:`~isaaclab.envs.mdp.joint_pos_limits`, zero for environments whose state is not finite."""
    return zero_non_finite(_joint_pos_limits(env, asset_cfg))
