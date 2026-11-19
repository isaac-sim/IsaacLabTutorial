# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck PPO configuration."""

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from isaaclab_tasks.utils import preset

from isaaclab_tutorial.tasks.velocity.mdp.symmetry import compute_symmetric_states

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class BalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MicroDuck velocity-tracking tasks."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 250
    experiment_name = "course_microduck_balance"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BalanceHeadPPORunnerCfg(BalancePPORunnerCfg):
    experiment_name = "course_microduck_balance_head"
    max_iterations = 1000


@configclass
class WalkPPORunnerCfg(BalancePPORunnerCfg):
    experiment_name = preset(default="course_microduck_walk", backlash="course_microduck_walk_backlash")
    max_iterations = "${microduck_scale:50000,${env.ngpus},${env.nnodes}}"
    save_interval = "${microduck_scale:250,${env.ngpus},${env.nnodes}}"


@configclass
class SymmetricWalkPPORunnerCfg(WalkPPORunnerCfg):
    experiment_name = preset(default="course_microduck_walk_symmetry", backlash="course_microduck_walk_backlash_symmetry")

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=compute_symmetric_states,
        )
