# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add training-only world-model information to the same nominal BAM Duck task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import ObservationsCfg

from isaaclab_tutorial.locomotion.supermimic import DuckEventLabels, DuckPhysicalState, duck_context, duck_goal

from .env_cfg import NominalEnvCfg


@configclass
class StateCfg(ObservationGroupCfg):
    physical = ObservationTermCfg(func=DuckPhysicalState)
    enable_corruption = False


@configclass
class GoalCfg(ObservationGroupCfg):
    command = ObservationTermCfg(func=duck_goal)
    enable_corruption = False


@configclass
class ContextCfg(ObservationGroupCfg):
    masses = ObservationTermCfg(func=duck_context)
    enable_corruption = False


@configclass
class LabelsCfg(ObservationGroupCfg):
    events = ObservationTermCfg(func=DuckEventLabels)
    enable_corruption = False


@configclass
class PreviousActionsCfg(ObservationGroupCfg):
    action = ObservationTermCfg(func=mdp.last_action)
    enable_corruption = False


@configclass
class SuperMimicObservationsCfg(ObservationsCfg):
    state = StateCfg()
    goal = GoalCfg()
    context = ContextCfg()
    labels = LabelsCfg()
    previous_actions = PreviousActionsCfg()


@configclass
class SuperMimicEnvCfg(NominalEnvCfg):
    """Preserve nominal rewards/actuators and capture every physical pre-reset successor."""

    observations: SuperMimicObservationsCfg = SuperMimicObservationsCfg()
    final_obs_every_step = True
