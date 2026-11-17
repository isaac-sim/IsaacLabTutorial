# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Native RSL-RL training with periodic rollout-only evaluation blocks.

The learning loop is adapted from RSL-RL 5.5.1 (BSD-3-Clause) to insert
evaluation between updates and explicitly replace the post-reset observation.
"""

import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from isaaclab_tutorial.active_gaze.training.evaluation import PeriodicEvaluation, coordinated_errors


def initialize_distributed():
    """Bind NCCL explicitly and validate communication before expensive RTX startup."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        probe = torch.ones(1, device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(probe)
        if probe.item() != world_size:
            raise RuntimeError("Distributed startup collective failed")
        print(
            f"ISAACLAB_FOVI_DISTRIBUTED rank={rank} local_rank={local_rank} world_size={world_size} ready",
            flush=True,
        )
    if torch.distributed.get_rank() != rank or torch.distributed.get_world_size() != world_size:
        raise ValueError("Existing process group differs from the requested ranks")


class DistributedOnPolicyRunner(OnPolicyRunner):
    """Native runner with explicit, reusable NCCL initialization."""

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            return super()._configure_multi_gpu()
        self.gpu_global_rank = int(os.environ["RANK"])
        self.gpu_local_rank = int(os.environ["LOCAL_RANK"])
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError("Runner device does not match the distributed local rank")
        initialize_distributed()
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }


class EvaluationOnPolicyRunner(DistributedOnPolicyRunner):
    """Native PPO and gradient reduction, with evaluation between updates."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.current_learning_iteration = -1
        self.evaluation = PeriodicEvaluation(env, self.cfg.get("evaluation", {}))

    def _log_action_diagnostics(self, iteration):
        """Separate motor/gaze saturation; aggregate only executed chunk prefixes."""
        writer = self.logger.writer
        if writer is None and not self.is_distributed:
            return
        storage = self.alg.storage
        mean, std = storage.distribution_params
        shape = (*storage.actions.shape[:2], self.env.chunk_size, -1)
        actions, mean, std = (value.reshape(shape) for value in (storage.actions, mean, std))
        valid = storage.observations.get("chunk_valid")
        if valid is None:
            valid = torch.ones(actions.shape[:-1], device=actions.device, dtype=torch.bool)
        motor = self.env.unwrapped.action_manager.get_term("motor")
        groups = {
            "arm": slice(0, motor.arm_action_dim),
            "hand": slice(motor.arm_action_dim, motor.action_dim),
            "gaze": slice(motor.action_dim, motor.action_dim + 2),
        }
        names, sums, counts = [], [], []
        for name, selection in groups.items():
            count = valid.sum() * actions[..., selection].shape[-1]
            mask = valid[..., None]
            measures = {
                "latent_std": std[..., selection],
                "latent_mean_abs": mean[..., selection].abs(),
                "saturation": (actions[..., selection].abs() >= 1).float(),
            }
            for measure, value in measures.items():
                names.append(f"Policy/{name}/{measure}")
                sums.append(torch.where(mask, value, 0.0).sum())
                counts.append(count)
        totals = torch.stack((torch.stack(sums), torch.stack(counts)))
        if self.is_distributed:
            torch.distributed.all_reduce(totals)
        values = totals[0] / totals[1].clamp_min(1)
        if writer is not None:
            for name, value in zip(names, values.tolist(), strict=True):
                writer.add_scalar(name, value, iteration)

    @property
    def training_control_steps(self):
        """Derive cadence from native checkpoint progress; no second saved counter."""
        return (self.current_learning_iteration + 1) * self.cfg["num_steps_per_env"] * self.env.chunk_size

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration + 1
        self.logger.tot_timesteps = start_it * self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            self._log_action_diagnostics(it)
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )

            previous_steps = self.training_control_steps - self.cfg["num_steps_per_env"] * self.env.chunk_size
            evaluate = self.evaluation.due(previous_steps, self.training_control_steps)
            # Preserve the exact policy shown in each evaluation video, including
            # early blocks that fall between the regular checkpoint intervals.
            if evaluate or it % self.cfg["save_interval"] == 0:
                with coordinated_errors(self):
                    if self.logger.writer is not None:
                        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
            if evaluate:
                obs = self.evaluation.run(self)

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
