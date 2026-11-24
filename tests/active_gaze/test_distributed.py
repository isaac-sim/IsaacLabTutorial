"""Distributed calibration must qualify every GPU and train the measured recipe."""

import json
from pathlib import Path
from types import SimpleNamespace

from isaaclab_tutorial.active_gaze import calibrate, distributed


def test_launch_and_memory_boundary_proposals():
    command = distributed.training_command(8)
    assert "--nproc_per_node=8" in command
    assert command[-1] == "isaaclab_tutorial.active_gaze.train"
    assert "torch.distributed.run" not in distributed.training_command(1)
    propose = calibrate.propose_environments
    assert propose([], 45, 8192, 2048) == 2048
    assert propose([(2048, float("inf"))], 45, 8192, 2048) == 1024
    assert propose([(2560, 44), (2688, 46)], 45, 8192, 2048) is None
    assert propose([(2048, 35), (2560, 40)], 45, 8192, 2048) == 2944


def report(rank=0, world=8, envs=2048, batch=1024, measured=2, peak=44e9):
    return dict(
        complete=True,
        error=None,
        gpu="L40",
        gpu_uuid=f"gpu-{rank}",
        rank=rank,
        world_size=world,
        num_envs=envs,
        rollout_steps=64,
        action_chunk_size=1,
        minibatch_size=batch,
        measured_updates=measured,
        total_bytes=48e9,
        peak_device_bytes=peak,
        peak_conservative_bytes=peak,
        peak_torch_allocated_bytes=peak / 2,
        peak_torch_reserved_bytes=peak / 2,
        mean_total_fps=1000,
        updates=[dict(iteration=i, collection_time=1, learning_time=1) for i in range(measured)],
    )


def test_missing_or_over_budget_rank_cannot_qualify(tmp_path):
    for rank in range(8):
        directory = tmp_path if rank == 0 else tmp_path / f"rank_{rank}"
        directory.mkdir(exist_ok=True)
        if rank == 7:
            assert distributed.read_measurement(tmp_path, 8) == {}
        (directory / "memory.json").write_text(json.dumps(report(rank, peak=(46e9 if rank == 7 else 44e9))))
    data = distributed.read_measurement(tmp_path, 8)
    assert data["peak_conservative_bytes"] == 46e9
    assert not calibrate.budget_ok(data, 2, 45e9)


def test_distributed_reset_metrics_pool_unequal_episode_counts(monkeypatch):
    import torch

    from isaaclab_tutorial.active_gaze.sampling import SuccessHistory, iteration_metrics

    local, remote = SuccessHistory(1), SuccessHistory(1)
    local.update(torch.tensor([0]), torch.tensor([True]))
    remote.update(torch.tensor([0, 0, 0]), torch.tensor([False, False, False]))
    sampler = SimpleNamespace(
        history=local, sampled=torch.tensor([1]), bin_names=["home"], phase_probabilities=lambda: torch.tensor([1.0])
    )
    other = torch.stack((remote.episodes, remote.successes, torch.tensor([3])))
    other_history = torch.stack((remote.history.sum(-1), remote.counts, torch.tensor([1.0])))
    additions = iter((other, other, other_history))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: value.add_(next(additions)))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    metrics, previous = iteration_metrics(sampler, 0, distributed=True)
    assert metrics["Reset/success_rate/home"] == 0.25
    assert metrics["Reset/success_rate_rolling/home"] == 0.25
    assert metrics["Reset/episodes/home"] == 4
    assert previous[:, 0].tolist() == [1, 1, 1]
    assert local.episodes.tolist() == [1]


def test_calibration_confirms_real_trials_and_backs_off(tmp_path, monkeypatch):
    def run(command, log, timeout):
        def option(name):
            return command[command.index(name) + 1]

        envs = int(option("--num_envs"))
        batch = envs * 16 // int(option("--minibatches"))
        measured = int(option("--iterations")) - int(option("--memory_warmup"))
        output = Path(option("--output"))
        output.mkdir()
        peak = 9e9 + envs * 1e7 + batch * 3e6 + (1e9 if measured == 8 else 0)
        for rank in range(8):
            path = output if rank == 0 else output / f"rank_{rank}"
            path.mkdir(exist_ok=True)
            (path / "memory.json").write_text(
                json.dumps(report(rank, envs=envs, batch=batch, measured=measured, peak=peak))
            )
        log.write_text("completed")
        return 0

    monkeypatch.setattr(calibrate, "run_trial", run)
    args = SimpleNamespace(
        output=tmp_path / "calibration",
        resume=False,
        num_gpus=8,
        memory_budget_gb=45,
        minibatch_sizes=[1024, 2048],
        max_envs=8192,
        initial_envs=2048,
        warmup=2,
        measure=2,
        confirm=8,
        timeout=10,
    )
    result = calibrate.calibrate(args, ["presets=dino_language,chunking"], rollout_steps=16)
    selected = result["selected"]
    assert result["complete"] and selected["world_size"] == 8
    assert selected["measured_updates"] == 8 and selected["peak_conservative_bytes"] <= 45e9
    assert selected["total_environments"] == selected["num_envs"] * 8
    assert len([t for t in result["trials"] if t["measured_updates"] == 8]) > 1
