"""Native RSL-RL W&B logging with portable task inputs and source provenance."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from rsl_rl.utils.wandb_log_writer import WandbLogWriter


def install_performance_metrics(runner):
    """Keep native decision FPS and add explicit, distributed control-step FPS."""
    import torch

    original = runner.logger.log

    def log(**kwargs):
        collect = kwargs["collect_time"]
        total = collect + kwargs["learn_time"]
        if runner.is_distributed:
            timings = torch.tensor([collect, total], device=runner.device)
            torch.distributed.all_reduce(timings, op=torch.distributed.ReduceOp.MAX)
            collect, total = timings.tolist()
        learn = total - collect
        kwargs = kwargs | {"collect_time": collect, "learn_time": learn}
        original(**kwargs)
        writer = runner.logger.writer
        if writer is not None:
            decisions = runner.env.num_envs * runner.cfg["num_steps_per_env"] * runner.gpu_world_size
            controls = decisions * runner.env.chunk_size
            for key, value in {
                "Perf/decision_fps": decisions / (collect + learn),
                "Perf/control_fps": controls / (collect + learn),
                "Perf/control_fps_per_gpu": controls / (collect + learn) / runner.gpu_world_size,
                "Perf/rollout_control_fps": controls / collect,
                "Perf/world_size": runner.gpu_world_size,
                "Train/control_steps": runner.logger.tot_timesteps * runner.env.chunk_size,
            }.items():
                writer.add_scalar(key, value, kwargs["it"])

    runner.logger.log = log


class WandbWriter(WandbLogWriter):
    """RSL-RL's W&B writer, with a configurable entity and group, and the run's description and inputs uploaded."""

    def __init__(self, log_dir: str, project_name: str, entity: str | None = None, group: str | None = None):
        # RSL-RL's writer reads the entity from WANDB_USERNAME; without one, W&B uses the login's default.
        if entity:
            os.environ["WANDB_USERNAME"] = entity
        if group:
            os.environ["WANDB_RUN_GROUP"] = group
        super().__init__(log_dir=log_dir, project_name=project_name)
        import wandb

        from ..configuration import RUN_FILE

        wandb.config.update({"run": json.loads((Path(log_dir) / RUN_FILE).read_text())}, allow_val_change=True)
        for name in (RUN_FILE, "agent.json", "env.yaml", "instructions.pt"):
            path = Path(log_dir) / name
            if path.is_file():
                self.save_file(str(path))
        print(f"W&B run: {wandb.run.url}", flush=True)

    def store_config(self, env_cfg, train_cfg):
        import wandb

        env = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
        wandb.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=True)
