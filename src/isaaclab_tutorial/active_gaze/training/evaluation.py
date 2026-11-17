"""Periodic deterministic rollout blocks, isolated from PPO collection and updates."""

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from rsl_rl.utils import check_nan


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    frequency_episodes: float = 100.0
    duration_episodes: float = 1.0
    early_episodes: list[float] = field(default_factory=list)
    video: bool = True
    video_envs: int = 3

    def __post_init__(self):
        for name in ("frequency_episodes", "duration_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"evaluation.{name} must be finite and positive")
        if type(self.enabled) is not bool or type(self.video) is not bool:
            raise ValueError("evaluation.enabled and video must be booleans")
        if type(self.video_envs) is not int or self.video_envs < 1:
            raise ValueError("evaluation.video_envs must be a positive integer")
        if not isinstance(self.early_episodes, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in self.early_episodes
        ):
            raise ValueError("evaluation.early_episodes must contain finite positive numbers")
        object.__setattr__(self, "early_episodes", sorted(set(self.early_episodes)))

    def control_steps(self, episode_length: int) -> tuple[int, int]:
        steps = []
        for value in (self.frequency_episodes, self.duration_episodes, *self.early_episodes):
            exact = value * episode_length
            count = round(exact)
            if count < 1 or not math.isclose(exact, count, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Evaluation durations must resolve to whole control steps")
            steps.append(count)
        return tuple(steps[:2])


class PeriodicEvaluation:
    """Use training control steps for cadence, independent of GPU or environment count."""

    def __init__(self, env, cfg: dict):
        self.cfg = EvaluationConfig(**cfg)
        self.frequency, self.duration = self.cfg.control_steps(env.max_episode_length)
        self.early_steps = tuple(round(value * env.max_episode_length) for value in self.cfg.early_episodes)

    def due(self, previous: int, current: int) -> bool:
        return self.cfg.enabled and (
            current // self.frequency > previous // self.frequency
            or any(previous < step <= current for step in self.early_steps)
        )

    def run(self, runner):
        env = runner.env
        with coordinated_errors(runner):
            obs, totals, elapsed, output, recorder = self._rollout(runner)
        if runner.is_distributed:
            torch.distributed.all_reduce(totals)
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        episodes, reward_sum, length_sum, successes = totals.tolist()
        metrics = {
            "Eval/completed_episodes": episodes,
            "Eval/control_steps_per_env": self.duration,
            "Eval/duration_episodes": self.duration / env.max_episode_length,
            "Eval/wall_seconds": elapsed.item(),
        }
        if episodes:
            metrics.update(
                {
                    "Eval/mean_return": reward_sum / episodes,
                    "Eval/mean_episode_control_steps": length_sum / episodes,
                    "Eval/success_rate": successes / episodes,
                }
            )
        with coordinated_errors(runner):
            if output is not None:
                metadata = {
                    "training_iteration": runner.current_learning_iteration,
                    "training_control_steps_per_env": runner.training_control_steps,
                    "policy": "deterministic actor mean; no oracle action substitution",
                    "control_fps": 1 / env.unwrapped.step_dt,
                    "policy_fps": 1 / env.decision_dt,
                    "action_chunk_size": env.chunk_size,
                    "metrics": metrics,
                }
                if recorder is not None:
                    metadata.update(
                        frame_count=recorder.frames,
                        videos={f"env_{i}": path.name for i, path in enumerate(recorder.paths)},
                        recorded_envs=recorder.num_envs,
                        instruction_segments=recorder.segments,
                        video_fps=1 / env.decision_dt,
                        source="post-chunk native foveated RGB; no intermediate camera captures",
                    )
                (output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
            writer = runner.logger.writer
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, runner.current_learning_iteration)
                if recorder is not None:
                    from .tracking import FoviWandbWriter

                    if isinstance(writer, FoviWandbWriter):
                        for path in recorder.paths:
                            writer.save_video(path, runner.current_learning_iteration)
        print(
            f"ISAACLAB_FOVI_EVAL complete rank={runner.gpu_global_rank} " + json.dumps(metrics),
            flush=True,
        )
        return obs.to(runner.device)

    def _rollout(self, runner):
        env = runner.env
        policy = runner.alg.get_policy()
        instructions = getattr(runner.alg.actor, "instructions", ())
        returns = torch.zeros(env.num_envs, device=env.device)
        lengths = torch.zeros_like(returns)
        # Completed episodes, returns, lengths, successes. Reduce sums, not rank means.
        totals = torch.zeros(4, device=env.device, dtype=torch.float64)
        recorder = None
        output = None
        count = min(self.cfg.video_envs, env.num_envs)
        if runner.gpu_global_rank == 0 and runner.logger.log_dir is not None:
            output = Path(runner.logger.log_dir) / "evaluation" / f"step_{runner.training_control_steps:09d}"
            output.mkdir(parents=True, exist_ok=True)
            if self.cfg.video:
                from .video import ObservationVideoRecorder

                recorder = ObservationVideoRecorder(output, 1 / env.decision_dt, count)

        def observe(obs, reward, done, extras):
            if runner.cfg.get("check_for_nan", True):
                check_nan(obs, reward, done)
            returns.add_(reward)
            lengths.add_(1)
            finished = done.bool()
            totals[0].add_(finished.sum())
            totals[1].add_(returns[finished].sum())
            totals[2].add_(lengths[finished].sum())
            totals[3].add_((env.unwrapped.termination_manager.get_term("success") & finished).sum())
            returns[finished] = 0
            lengths[finished] = 0

        print(
            f"ISAACLAB_FOVI_EVAL start rank={runner.gpu_global_rank} training_control_steps={runner.training_control_steps} control_steps={self.duration}",
            flush=True,
        )
        start = time.perf_counter()
        # Restore nested module modes exactly, including deliberately frozen children.
        models = [runner.alg.actor, runner.alg.critic]
        if runner.alg.rnd is not None:
            models.append(runner.alg.rnd)
        modes = {module: module.training for model in models for module in model.modules()}
        rendering = env.unwrapped.render_enabled
        try:
            runner.alg.eval_mode()
            env.unwrapped.render_enabled = True
            with torch.inference_mode():
                obs, _ = env.reset()
                policy.reset()
                for step in range(0, self.duration, env.chunk_size):
                    with torch.autocast(
                        torch.device(runner.device).type,
                        dtype=torch.bfloat16,
                        enabled=runner.alg.use_mixed_precision,
                    ):
                        actions = policy(obs.to(runner.device))
                    obs, _, done, _ = env.step(
                        actions.to(env.device),
                        control_steps=min(env.chunk_size, self.duration - step),
                        observer=observe,
                    )
                    if recorder is not None:
                        # Vision already refreshed at this boundary. Reading it
                        # in the per-control-step observer would trigger a lazy
                        # render of the entire tiled camera between decisions.
                        rgb = env.unwrapped.scene["eye"].data.output["rgb"]
                        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                        captions = [""] * count
                        if instructions:
                            indices = obs["instruction"][:count, 0].long().cpu().tolist()
                            captions = [instructions[i] for i in indices]
                        recorder.capture(rgb[:count].cpu().numpy(), captions)
                    policy.reset(done)
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                for module, training in modes.items():
                    module.training = training
                # Discard evaluation state and partial training episodes. The caller
                # must collect from this fresh observation, never its pre-eval buffer.
                env.unwrapped.render_enabled = rendering
                with torch.inference_mode():
                    obs, _ = env.reset()
                    policy.reset()
                runner.logger.cur_reward_sum.zero_()
                runner.logger.cur_episode_length.zero_()
                runner.logger.ep_extras.clear()
                if runner.alg.rnd is not None:
                    # Match the discarded training episodes in native RND logging.
                    runner.logger.cur_ereward_sum.zero_()
                    runner.logger.cur_ireward_sum.zero_()
        elapsed = torch.tensor(time.perf_counter() - start, device=env.device)
        return obs, totals, elapsed, output, recorder


@contextmanager
def coordinated_errors(runner):
    """Let every rank leave an evaluation phase before any rank resumes PPO.

    A rank-zero encoder/upload failure must not strand peers in the next gradient
    collective. Healthy ranks finish this bounded phase, then all ranks raise.
    """
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    failed = torch.tensor(int(error is not None), device=runner.device)
    if runner.is_distributed:
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if error is not None:
        raise error
    if failed.item():
        raise RuntimeError("Evaluation failed on another rank; stopping training")
