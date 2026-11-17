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


class FoviWandbWriter(WandbLogWriter):
    def store_config(self, env_cfg, train_cfg):
        """Record the active configuration when an existing W&B run resumes."""
        import wandb

        env = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
        wandb.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=wandb.run.resumed)

    def save_video(self, path: Path, it: int):
        """Log each block once, including native RSL-RL's subsequent video scan."""
        import wandb

        identity = str(path.resolve())
        if identity not in self.logged_videos:
            wandb.log({f"Eval/observations/{path.stem}": wandb.Video(identity, format="mp4")}, step=it)
            self.logged_videos.add(identity)

    def __init__(self, log_dir: str, project_name: str, entity: str, group: str | None = None):
        # RSL-RL's writer currently reads WANDB_USERNAME, while W&B reads WANDB_ENTITY.
        os.environ["WANDB_USERNAME"] = os.environ["WANDB_ENTITY"] = entity
        if group:
            os.environ["WANDB_RUN_GROUP"] = group
        super().__init__(log_dir=log_dir, project_name=project_name)
        import wandb

        settings_path = Path(log_dir) / "config.json"
        if not settings_path.is_file():
            # The standalone SO-101 launcher preserves its existing checkpoint metadata.
            settings_path = Path(log_dir) / "example.json"
        settings = json.loads(settings_path.read_text())
        wandb.config.update({"isaaclab_fovi": settings}, allow_val_change=wandb.run.resumed)
        provenance_path = Path(log_dir) / "provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            wandb.config.update(
                {
                    "source": {
                        key: provenance.get(key)
                        for key in (
                            "bundle_id",
                            "isaaclab_commit",
                            "image",
                            "runtime",
                            "workflow_id",
                        )
                    }
                },
                allow_val_change=wandb.run.resumed,
            )
        for name in (
            "config.json",
            "example.json",
            "agent.json",
            "env.yaml",
            "instructions.pt",
            "gaze_reset_cache.pt",
            "calibration.json",
            "provenance.json",
        ):
            path = Path(log_dir) / name
            if path.is_file():
                self.save_file(str(path))
        print(f"ISAACLAB_FOVI_WANDB {wandb.run.url}", flush=True)
