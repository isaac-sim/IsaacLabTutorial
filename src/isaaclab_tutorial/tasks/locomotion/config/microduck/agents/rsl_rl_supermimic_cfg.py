# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Microduck model-based RL with the incoming native RSL-RL SuperMimic workflow."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)


@configclass
class MicroDuckSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """Fit a Duck world model and optimize its history actor through short imagined rollouts."""

    num_steps_per_env = 1
    max_iterations = 10_000
    save_interval = 1000
    experiment_name = "microduck_tutorial_supermimic"
    clip_actions = None
    obs_groups = {
        "state": ["state"],
        "goal": ["goal"],
        "context": ["context"],
        "labels": ["labels"],
        "prev_action": ["previous_actions"],
    }
    actor = RslRlSuperMimicActorCfg(
        history_length=5,
        latent_dim=32,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(512, 256, 128),
    )
    world_model = RslRlWorldModelCfg(hidden_dims=(512, 512, 512), label_hidden_dims=(256, 128))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlSuperMimicAlgorithmCfg(
        task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tutorial.locomotion.supermimic:MicroDuckTask"),
        replay_buffer_size=1_000_000,
        max_replay_memory_fraction=0.3,
        updates_per_step=0.4,
        num_world_model_updates=4,
        world_model_window=8,
        world_model_batch_size=1024,
        horizon=16,
        policy_batch_size=1024,
        warmup_steps=64,
        world_model_warmup_iterations=500,
        # dt × (tracking4+4, upright2, pose1, air-time3 per foot, head2).
        max_step_reward=0.38,
    )
