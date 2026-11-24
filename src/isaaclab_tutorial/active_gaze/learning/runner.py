# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Native RSL-RL training across GPUs, with NCCL bound before the simulator starts."""

import os

import torch
from rsl_rl.runners import OnPolicyRunner


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
        print(f"Distributed rank={rank} local_rank={local_rank} world_size={world_size} ready", flush=True)
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
