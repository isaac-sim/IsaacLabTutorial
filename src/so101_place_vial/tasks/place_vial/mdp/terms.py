"""Tutorial observation helpers needed before the full task is introduced."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid terminal-state values with finite neutral values."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint positions for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    position = _finite(_tensor(robot.data.joint_pos)[:, asset_cfg.joint_ids])
    limits = _tensor(robot.data.soft_joint_pos_limits)[:, asset_cfg.joint_ids]
    return position.clamp(limits[..., 0], limits[..., 1])


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint velocities for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    return _finite(_tensor(robot.data.joint_vel)[:, asset_cfg.joint_ids]).clamp(-12.0, 12.0)
