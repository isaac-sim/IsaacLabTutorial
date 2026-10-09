"""Opt-in NVTX/CPU instrumentation for Nsight profiling of a warmed-up state rollout.

Pass ``--external_callback isaaclab_tutorial.utils.profile_so101.install_profile``
to training under ``nsys profile --capture-range=cudaProfilerApi``. Output path is
selected by SO101_PROFILE_OUTPUT; instrumentation does not change actions or physics.
"""

import json
import os
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path


def install_profile() -> list[str]:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    original_step = ManagerBasedRLEnv.step
    start_step, count = 64, 32
    state = {"step": 0, "active": False, "initialized": False}
    timings = defaultdict(lambda: {"cpu_seconds": 0.0, "calls": 0})

    def instrument(obj, attribute, name):
        original = getattr(obj, attribute)

        @wraps(original)
        def wrapped(*args, **kwargs):
            if not state["active"]:
                return original(*args, **kwargs)
            torch.cuda.nvtx.range_push(name)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[name]["cpu_seconds"] += time.perf_counter() - started
                timings[name]["calls"] += 1
                torch.cuda.nvtx.range_pop()

        setattr(obj, attribute, wrapped)

    def profiled_step(self, action):
        if not state["initialized"]:
            instrument(torch.Tensor, "new_tensor", "new_tensor_nested")
            for obj, attr, name in (
                (self.action_manager, "process_action", "action_process"),
                (self.action_manager, "apply_action", "action_apply"),
                (self.scene, "write_data_to_sim", "scene_write"),
                (self.sim, "step", "physics_step"),
                (self.scene, "update", "scene_update"),
                (self.termination_manager, "compute", "terminations"),
                (self.reward_manager, "compute", "rewards"),
                (self.observation_manager, "compute", "observations"),
                (self, "_reset_idx", "reset"),
                (self.event_manager, "apply", "events_nested_in_reset"),
            ):
                instrument(obj, attr, name)
            state["initialized"] = True
        if state["step"] == start_step:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            state["active"] = True
            state["started"] = time.perf_counter()
        if state["active"]:
            torch.cuda.nvtx.range_push("env_step")
        result = original_step(self, action)
        if state["active"]:
            torch.cuda.nvtx.range_pop()
        state["step"] += 1
        if state["step"] == start_step + count:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - state["started"]
            state["active"] = False
            torch.cuda.cudart().cudaProfilerStop()
            report = {
                "num_envs": self.num_envs,
                "sampled_env_steps": count,
                "warmup_env_steps": start_step,
                "decimation": self.cfg.decimation,
                "physics_handles_decimation": self._physics_handles_decimation,
                "profiled_wall_ms_per_env_step": elapsed * 1000 / count,
                "components": dict(timings),
                "note": "CPU timings include waits; nested ranges overlap. Nsight adds profiling overhead.",
            }
            output = Path(os.environ["SO101_PROFILE_OUTPUT"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2))
        return result

    ManagerBasedRLEnv.step = profiled_step
    return []
