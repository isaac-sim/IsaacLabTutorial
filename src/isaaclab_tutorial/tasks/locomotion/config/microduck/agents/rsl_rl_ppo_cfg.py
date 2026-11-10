# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO teaching preset; learning success is evaluated separately from runner success."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class PPORunnerCfg(MicroDuckPPORunnerCfg):
    experiment_name = "microduck_tutorial"
    max_iterations = 50
    save_interval = 25
