# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read physical playground signals before autoreset so a failed trial remains visible."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def physical_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return pre-reset pose, raw IMU/encoders, and measured heading velocity for display."""
    robot = env.scene["robot"]
    data = robot.data
    velocity = quat_apply_inverse(yaw_quat(data.root_link_quat_w.torch), data.root_link_lin_vel_w.torch)
    return torch.cat(
        (
            robot.data.root_link_pos_w.torch,
            robot.data.root_link_quat_w.torch,
            data.joint_pos.torch,
            data.root_link_ang_vel_b.torch,
            data.projected_gravity_b.torch,
            data.joint_vel.torch,
            velocity[:, :2],
            data.root_link_ang_vel_w.torch[:, 2:3],
        ),
        dim=-1,
    )


@configclass
class PlaygroundTelemetryCfg(ObservationGroupCfg):
    """Extra display data excluded from the actor and critic observation groups."""

    state = ObservationTermCfg(func=physical_state)
    enable_corruption = False
