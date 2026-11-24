"""Measure real rollout/update memory, including allocations outside PyTorch."""

import json
import math
import threading
from pathlib import Path


def minibatch_count(num_envs: int, rollout_steps: int, minibatch_size: int) -> int:
    """Translate a sample count into RSL-RL's batch count without dropping transitions."""
    total = num_envs * rollout_steps
    if min(num_envs, rollout_steps, minibatch_size) <= 0 or total % minibatch_size:
        raise ValueError("Minibatch size must be positive and divide the complete rollout.")
    return total // minibatch_size


class TrainingMemoryRecorder:
    """Post-warmup device-wide NVML usage plus exact PyTorch allocator peaks.

    Use a dedicated GPU: device-wide memory includes RTX, PhysX, Warp, and any
    other process on that GPU. NVML UUID lookup respects CUDA device remapping.
    The runner's logging callback marks completed updates, not printed log lines.
    """

    def __init__(self, runner, output: Path, warmup: int = 4, interval: float = 0.02):
        import pynvml
        import torch

        if warmup < 1 or interval <= 0:
            raise ValueError("Memory recording requires positive warmup and polling interval.")
        self.runner, self.output, self.warmup = runner, output, warmup
        self.torch, self.nvml = torch, pynvml
        self.device = runner.device
        self.properties = torch.cuda.get_device_properties(self.device)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(self.properties.uuid))
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active = False
        self.peak = 0
        self.allocator_adjusted_peak = 0
        self.completed = 0
        self.samples = []
        self.error = None
        self.original_log = runner.logger.log
        self.thread = threading.Thread(target=self._poll, name="isaaclab_fovi-memory", daemon=True)

    def _sample(self):
        with self.lock:
            if self.active:
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                self.peak = max(self.peak, used)
                # Add any allocator peak missed between NVML polls. This is conservative:
                # non-Torch and allocator maxima need not occur at the same instant.
                reserved = self.torch.cuda.memory_reserved(self.device)
                allocator_peak = self.torch.cuda.max_memory_reserved(self.device)
                self.allocator_adjusted_peak = max(
                    self.allocator_adjusted_peak, used + max(0, allocator_peak - reserved)
                )

    def _poll(self):
        try:
            while not self.stop_event.wait(self.interval):
                self._sample()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _log(self, **kwargs):
        self.original_log(**kwargs)
        if not all(math.isfinite(float(v)) for v in kwargs["loss_dict"].values()):
            raise RuntimeError("Nonfinite training loss during memory measurement.")
        self.completed += 1
        if self.completed == self.warmup:
            self.torch.cuda.synchronize(self.device)
            with self.lock:
                self.torch.cuda.reset_peak_memory_stats(self.device)
                self.active = True
            self._sample()
        elif self.completed > self.warmup:
            self._sample()
            steps = self.runner.env.num_envs * self.runner.cfg["num_steps_per_env"]
            self.samples.append(
                {
                    "iteration": kwargs["it"],
                    "rollout_fps": steps / kwargs["collect_time"],
                    "total_fps": steps / (kwargs["collect_time"] + kwargs["learn_time"]),
                    "collection_time": kwargs["collect_time"],
                    "learning_time": kwargs["learn_time"],
                }
            )

    def __enter__(self):
        self.runner.logger.log = self._log
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.runner.logger.log = self.original_log
        try:
            self._sample()
            info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            decisions = self.runner.env.num_envs * self.runner.cfg["num_steps_per_env"] * len(self.samples)
            total_time = sum(s["collection_time"] + s["learning_time"] for s in self.samples)
            mean_fps = decisions / total_time if total_time else 0.0
            data = {
                "complete": exc_type is None and not self.error and bool(self.samples),
                "error": str(exc) if exc else self.error,
                "gpu": self.properties.name,
                "gpu_uuid": str(self.properties.uuid),
                "total_bytes": info.total,
                "peak_device_bytes": self.peak,
                "peak_conservative_bytes": self.allocator_adjusted_peak,
                "peak_torch_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "peak_torch_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
                "warmup_updates": self.warmup,
                "measured_updates": len(self.samples),
                "num_envs": self.runner.env.num_envs,
                "action_chunk_size": self.runner.env.chunk_size,
                "world_size": self.runner.gpu_world_size,
                "rank": self.runner.gpu_global_rank,
                "rollout_steps": self.runner.cfg["num_steps_per_env"],
                "minibatch_size": self.runner.env.num_envs
                * self.runner.cfg["num_steps_per_env"]
                // self.runner.alg.num_mini_batches,
                "mean_total_fps": mean_fps,
                "mean_control_fps": mean_fps * self.runner.env.chunk_size,
                "updates": self.samples,
            }
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps(data, indent=2) + "\n")
            print("ISAACLAB_FOVI_MEMORY " + json.dumps(data), flush=True)
        finally:
            self.nvml.nvmlShutdown()
        if exc_type is None and not data["complete"]:
            raise RuntimeError(f"Memory measurement incomplete: {data['error']}")
