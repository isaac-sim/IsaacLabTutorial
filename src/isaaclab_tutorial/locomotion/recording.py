# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record tracking error at reward computation time, before environment autoreset."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_tasks.contrib.microduck.mdp.rewards import track_linear_velocity_heading


class RecordedLinearTracking(ManagerTermBase):
    """Preserve the maintained reward and expose its matching-time error for rescoring."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.last_error_sq = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env, std: float, command_name: str, asset_cfg: SceneEntityCfg | None = None):
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot")
        robot = env.scene[asset_cfg.name]
        velocity = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch)
        command = env.command_manager.get_command(command_name)
        self.last_error_sq = torch.sum((command[:, :2] - velocity[:, :2]) ** 2, dim=1)
        return track_linear_velocity_heading(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
