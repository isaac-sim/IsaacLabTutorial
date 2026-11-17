"""Evaluation stays isolated from training and captures RGB only at chunk boundaries."""

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import imageio.v2 as imageio
import pytest
import torch
from tensordict import TensorDict

from isaaclab_tutorial.active_gaze.learning.chunks import ActionChunkWrapper
from isaaclab_tutorial.active_gaze.learning.runner import EvaluationOnPolicyRunner
from isaaclab_tutorial.active_gaze.training.evaluation import EvaluationConfig, PeriodicEvaluation, coordinated_errors
from isaaclab_tutorial.active_gaze.training.video import ObservationVideoRecorder


class CountingCamera:
    def __init__(self):
        self.rgb = torch.zeros(3, 16, 16, 3, dtype=torch.uint8)
        self.reads = 0

    @property
    def data(self):
        self.reads += 1
        return SimpleNamespace(output={"rgb": self.rgb})


class EvaluationEnv:
    num_envs, num_actions, device, max_episode_length = 3, 3, "cpu", 5
    step_dt = 0.1

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(compute_final_obs=False, to_dict=lambda: {})
        self.render_enabled = True
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.resets = 0
        self.steps = 0
        self.calls = []
        self.action_manager = SimpleNamespace(
            get_term=lambda _: SimpleNamespace(arm_action_dim=1, motor_action_dim=1, action_dim=1)
        )
        self.termination_manager = SimpleNamespace(get_term=lambda _: torch.zeros(3, dtype=torch.bool))
        self.scene = {"eye": CountingCamera()}

    def get_observations(self):
        state = torch.stack((torch.full((3,), self.resets), self.episode_length_buf), dim=-1).float()
        return TensorDict(
            dict(
                proprio=state,
                rnd_state=state.clone(),
                privileged=state.clone(),
                instruction=torch.arange(3)[:, None],
                oracle=torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone(),
            ),
            batch_size=[3],
        )

    def reset(self):
        self.resets += 1
        self.episode_length_buf.zero_()
        return self.get_observations(), {}

    def step(self, action):
        self.calls.append((action.clone(), self.render_enabled, self.refresh_policy_vision))
        self.steps += 1
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        final = self.get_observations()
        self.episode_length_buf[done] = 0
        self.reset_terminated = torch.zeros(3, dtype=torch.bool)
        self.scene["eye"].rgb.fill_(self.steps % 250)
        return (
            self.get_observations(),
            torch.ones(3),
            done.long(),
            {"time_outs": done, "final_obs": final},
        )


def make_runner(tmp_path, *, rnd=False, **evaluation):
    env = ActionChunkWrapper(EvaluationEnv(), 4, 0.99, critic_groups=["proprio", "privileged"])
    cfg = {
        "actor": dict(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=[8],
            obs_normalization=True,
            distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.5, std_type="log"),
        ),
        "critic": dict(class_name="rsl_rl.models:MLPModel", hidden_dims=[8], obs_normalization=True),
        "obs_groups": {"actor": ["proprio"], "critic": ["proprio", "privileged"]},
        "algorithm": dict(
            class_name="isaaclab_tutorial.active_gaze.learning.oracle_ppo:OraclePPO",
            num_learning_epochs=1,
            num_mini_batches=1,
            schedule="fixed",
            action_chunk_size=4,
        ),
        "num_steps_per_env": 2,
        "save_interval": 100,
        "logger": "tensorboard",
        "evaluation": asdict(EvaluationConfig(frequency_episodes=4)) | evaluation,
    }
    if rnd:
        cfg["obs_groups"]["rnd_state"] = ["rnd_state"]
        cfg["algorithm"]["rnd_cfg"] = dict(
            num_outputs=8,
            predictor_hidden_dims=[16, 16],
            target_hidden_dims=[16],
            state_normalization=True,
            reward_normalization=True,
            weight=1.0,
        )
    runner = EvaluationOnPolicyRunner(env, cfg, str(tmp_path), "cpu")
    runner.alg.actor.instructions = ("Place the vial.", "Move to the rack.", "Release the vial.")
    return runner


def assert_state_equal(before, after):
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_state_equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        for left, right in zip(before, after, strict=True):
            assert_state_equal(left, right)
    else:
        assert before == after


def test_default_schedule_uses_training_time_and_crosses_rollout_boundaries():
    schedule = PeriodicEvaluation(SimpleNamespace(max_episode_length=360), {})
    assert (schedule.frequency, schedule.duration) == (36000, 360)
    assert not schedule.due(35936, 35968)
    assert schedule.due(35968, 36000)
    assert not schedule.due(36000, 36032)
    # An interval that doesn't divide PPO's horizon runs at the following boundary.
    schedule = PeriodicEvaluation(SimpleNamespace(max_episode_length=5), {"frequency_episodes": 4})
    assert schedule.due(16, 24)
    assert not schedule.due(24, 32)
    assert schedule.due(32, 40)
    assert not PeriodicEvaluation(SimpleNamespace(max_episode_length=5), {"enabled": False}).due(16, 24)
    for kwargs in (
        {"duration_episodes": 0},
        {"frequency_episodes": float("nan")},
        {"video_envs": 0},
    ):
        with pytest.raises(ValueError):
            EvaluationConfig(**kwargs)
    with pytest.raises(ValueError, match="whole control steps"):
        EvaluationConfig(duration_episodes=0.25).control_steps(5)


def test_early_schedule_coalesces_thresholds_and_does_not_replay_on_resume():
    schedule = PeriodicEvaluation(
        SimpleNamespace(max_episode_length=5),
        {"frequency_episodes": 4, "early_episodes": [4, 2, 1, 1]},
    )
    assert schedule.early_steps == (5, 10, 20)
    assert [schedule.due(step, step + 8) for step in range(0, 40, 8)] == [
        True,
        True,
        True,
        False,
        True,
    ]
    assert not schedule.due(24, 32)
    assert not PeriodicEvaluation(
        SimpleNamespace(max_episode_length=5),
        {"enabled": False, "frequency_episodes": 4, "early_episodes": [1]},
    ).due(0, 8)
    for values in ([0], [-1], [float("nan")], [True], ["1"], "1"):
        with pytest.raises(ValueError, match="early_episodes"):
            EvaluationConfig(early_episodes=values)
    with pytest.raises(ValueError, match="whole control steps"):
        EvaluationConfig(early_episodes=[0.25]).control_steps(5)


def test_early_blocks_record_video_save_matching_policy_and_resume_without_replay(tmp_path):
    runner = make_runner(tmp_path, early_episodes=[1, 2])
    evaluated_iterations = []
    original_run = runner.evaluation.run

    def run(active):
        iteration = active.current_learning_iteration
        checkpoint = tmp_path / f"model_{iteration}.pt"
        assert checkpoint.is_file()
        saved = torch.load(checkpoint, weights_only=False)
        assert saved["iter"] == iteration
        for key, value in active.alg.save().items():
            assert_state_equal(value, saved[key])
        evaluated_iterations.append(iteration)
        return original_run(active)

    runner.evaluation.run = run
    runner.learn(4)
    assert evaluated_iterations == [0, 1, 2]
    blocks = list((tmp_path / "evaluation").glob("*/evaluation.json"))
    assert len(blocks) == 3
    assert len(list((tmp_path / "evaluation").glob("*/env_*.mp4"))) == 9
    assert runner.env.unwrapped.steps == 32 + 3 * 5
    resumed = make_runner(tmp_path / "resumed", video=False, early_episodes=[1, 2])
    resumed.load(str(tmp_path / "model_1.pt"))
    assert resumed.training_control_steps == 16
    resumed.learn(1)
    assert resumed.env.unwrapped.steps == 8 + 5
    assert (tmp_path / "resumed/evaluation/step_000000024/evaluation.json").exists()


def test_checkpoint_failure_stops_before_evaluation(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, early_episodes=[1])

    def fail_save(*args):
        raise OSError("checkpoint storage unavailable")

    monkeypatch.setattr(runner, "save", fail_save)
    with pytest.raises(OSError, match="checkpoint storage unavailable"):
        runner.learn(1)
    assert not (tmp_path / "evaluation").exists()
    assert runner.env.unwrapped.steps == 8


@pytest.mark.parametrize("rnd", [False, True])
def test_eval_freezes_learning_resets_all_envs_and_records_at_chunk_rate(tmp_path, rnd):
    runner = make_runner(tmp_path, rnd=rnd)
    # Seed real PPO/Adam and normalization state before taking the snapshot.
    runner.learn(1)
    runner.logger.cur_reward_sum.fill_(20)
    runner.logger.cur_episode_length.fill_(3)
    runner.alg.actor.mlp.eval()  # Deliberately frozen nested mode must also survive.
    modes = {module: module.training for model in (runner.alg.actor, runner.alg.critic) for module in model.modules()}
    before = copy.deepcopy(runner.alg.save())
    if rnd:
        rnd_modes = {module: module.training for module in runner.alg.rnd.modules()}
        rnd_returns = runner.alg.rnd.reward_normalizer.disc_avg.avg.clone()
        runner.logger.cur_ereward_sum.fill_(20)
        runner.logger.cur_ireward_sum.fill_(3)
    storage = runner.alg.storage.step
    control = runner.env.unwrapped
    reset_count, control_count = control.resets, control.steps
    obs = runner.evaluation.run(runner)
    assert_state_equal(before, runner.alg.save())
    if rnd:
        assert all(module.training == value for module, value in rnd_modes.items())
        torch.testing.assert_close(rnd_returns, runner.alg.rnd.reward_normalizer.disc_avg.avg)
        assert not runner.logger.cur_ereward_sum.any() and not runner.logger.cur_ireward_sum.any()
    assert runner.alg.storage.step == storage
    assert all(module.training == value for module, value in modes.items())
    assert control.resets == reset_count + 2
    assert control.steps == control_count + 5
    assert runner.training_control_steps == 8  # Evaluation isn't training progress.
    assert [(render, fresh) for _, render, fresh in control.calls[-5:]] == [
        (False, False),
        (False, False),
        (False, False),
        (True, True),
        (True, True),
    ]
    assert (obs["proprio"][:, 0] == control.resets).all()
    assert (obs["proprio"][:, 1] == 0).all()
    assert not runner.logger.cur_reward_sum.any()
    assert not runner.logger.cur_episode_length.any()
    folder = tmp_path / "evaluation/step_000000008"
    meta = json.loads((folder / "evaluation.json").read_text())
    assert len(list(folder.glob("env_*.mp4"))) == 3
    assert meta["videos"] == {f"env_{i}": f"env_{i}.mp4" for i in range(3)}
    assert control.scene["eye"].reads == 2  # No lazy camera reads inside a chunk.
    assert meta["frame_count"] == 2
    assert meta["instruction_segments"][0]["instructions"] == list(runner.alg.actor.instructions)
    assert meta["control_fps"] == 10
    assert meta["policy_fps"] == 2.5
    assert meta["video_fps"] == 2.5
    assert meta["metrics"]["Eval/completed_episodes"] == 3
    assert meta["metrics"]["Eval/mean_return"] == 5
    with imageio.get_reader(folder / "env_0.mp4") as reader:
        frames = [frame for frame in reader]
        assert reader.get_meta_data()["fps"] == 2.5
    assert len(frames) == 2
    assert frames[0].shape == (320, 320, 3)
    assert frames[0][100, 100, 0] < frames[-1][100, 100, 0]


@pytest.mark.parametrize("video,rank", [(False, 0), (True, 1)])
def test_eval_without_recording_never_requests_intermediate_rgb(tmp_path, video, rank):
    runner = make_runner(tmp_path, video=video)
    runner.gpu_global_rank = rank
    runner.evaluation.run(runner)
    control = runner.env.unwrapped
    assert control.scene["eye"].reads == 0
    assert [(render, fresh) for _, render, fresh in control.calls] == [
        (False, False),
        (False, False),
        (False, False),
        (True, True),
        (True, True),
    ]


def test_training_uses_fresh_observations_after_eval_and_resumes_schedule(tmp_path):
    runner = make_runner(tmp_path, video=False)
    original_act = runner.alg.act
    actions = []

    def act(obs):
        torch.testing.assert_close(obs["proprio"], runner.env.get_observations()["proprio"])
        actions.append(obs["proprio"].clone())
        return original_act(obs)

    runner.alg.act = act
    runner.learn(4)
    assert len(actions) == 8
    assert runner.training_control_steps == 32
    assert runner.env.unwrapped.steps == 37  # Four PPO rollouts plus one five-step eval.
    assert actions[-1][0, 0] == 2  # Both full resets happened before training resumed.
    assert len(list((tmp_path / "evaluation").glob("*/evaluation.json"))) == 1
    resumed = make_runner(tmp_path / "resumed", video=False)
    resumed.load(str(tmp_path / "model_3.pt"))
    assert resumed.training_control_steps == 32
    resumed.learn(1)
    assert resumed.current_learning_iteration == 4
    assert resumed.training_control_steps == 40
    assert resumed.logger.tot_timesteps == 5 * 2 * resumed.env.num_envs
    assert (tmp_path / "resumed/evaluation/step_000000040/evaluation.json").exists()
    assert resumed.env.unwrapped.steps == 13


def test_recorder_failure_restores_state_and_does_not_continue_training(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    modes = [runner.alg.actor.training, runner.alg.critic.training]

    def fail(*args):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(ObservationVideoRecorder, "capture", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        runner.evaluation.run(runner)
    assert [runner.alg.actor.training, runner.alg.critic.training] == modes
    assert runner.env.unwrapped.resets == 2
    assert runner.env.unwrapped.render_enabled and runner.env.unwrapped.refresh_policy_vision


def test_peer_evaluation_failure_is_propagated_before_training(monkeypatch):
    runner = SimpleNamespace(is_distributed=True, device="cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda failed, **kwargs: failed.fill_(1))
    with pytest.raises(RuntimeError, match="another rank"):
        with coordinated_errors(runner):
            pass


def _distributed_evaluation_worker(rank, rendezvous, directory, fail_encoder):
    from datetime import timedelta
    from pathlib import Path

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        runner = make_runner(Path(directory) / f"rank{rank}", video=fail_encoder)
        runner.is_distributed = True
        runner.gpu_global_rank = rank
        if fail_encoder and rank == 0:

            def fail(*args):
                raise RuntimeError("encoder failed")

            ObservationVideoRecorder.capture = fail
        if fail_encoder:
            with pytest.raises(RuntimeError, match="encoder failed|another rank"):
                runner.evaluation.run(runner)
        else:
            runner.evaluation.run(runner)
        assert runner.env.unwrapped.resets == 2
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fail_encoder", [False, True])
def test_distributed_evaluation_reduces_metrics_and_coordinates_rank_zero_failure(tmp_path, fail_encoder):
    torch.multiprocessing.spawn(
        _distributed_evaluation_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), fail_encoder),
        nprocs=2,
        join=True,
    )
    if not fail_encoder:
        data = json.loads((tmp_path / "rank0/evaluation/step_000000000/evaluation.json").read_text())
        assert data["metrics"]["Eval/completed_episodes"] == 6
        assert data["metrics"]["Eval/mean_return"] == 5
        assert not (tmp_path / "rank1/evaluation").exists()


def _distributed_rnd_worker(rank, rendezvous, directory):
    from datetime import timedelta
    from pathlib import Path

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(10 + rank)
        runner = make_runner(Path(directory) / f"rank{rank}", rnd=True, enabled=False)
        runner.is_distributed = runner.alg.is_multi_gpu = True
        runner.gpu_global_rank = runner.alg.gpu_global_rank = rank
        runner.gpu_world_size = runner.alg.gpu_world_size = 2
        # Different local novelty streams must still produce shared models/moments.
        runner.env.unwrapped.resets = rank + 1
        runner.learn(1)
        for model in (runner.alg.actor, runner.alg.critic, runner.alg.rnd):
            for value in model.state_dict().values():
                if isinstance(value, torch.Tensor):
                    reference = value.clone()
                    torch.distributed.broadcast(reference, 0)
                    torch.testing.assert_close(value, reference)
        assert all(p.grad is None for p in runner.alg.rnd.target.parameters())
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_rnd_synchronizes_target_predictor_gradients_and_moments(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_rnd_worker,
        args=(f"file://{tmp_path / 'rnd-rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )


def test_wandb_records_each_block_once_despite_native_video_scan(tmp_path, monkeypatch):
    import wandb

    from isaaclab_tutorial.active_gaze.training.tracking import FoviWandbWriter

    writer = object.__new__(FoviWandbWriter)
    writer.logged_videos = set()
    logged = []
    monkeypatch.setattr(wandb, "Video", lambda path, **kwargs: path)
    monkeypatch.setattr(wandb, "log", lambda values, **kwargs: logged.append((values, kwargs)))
    first = tmp_path / "step_001/env_0.mp4"
    second = tmp_path / "step_002/env_0.mp4"
    writer.save_video(first, 1)
    writer.save_video(first, 2)  # Native logger scans old and new MP4 files.
    writer.save_video(second, 2)
    third = tmp_path / "step_002/env_1.mp4"
    writer.save_video(third, 2)
    assert logged[-1][0]["Eval/observations/env_1"] == str(third.resolve())
    assert len(logged) == 3
    assert logged[0][0]["Eval/observations/env_0"] == str(first.resolve())
    assert logged[1][0]["Eval/observations/env_0"] == str(second.resolve())


@pytest.mark.parametrize("settings_file", ["config.json", "example.json"])
def test_resumed_wandb_writer_updates_active_configuration(tmp_path, monkeypatch, settings_file):
    import wandb
    from rsl_rl.utils.wandb_log_writer import WandbLogWriter

    from isaaclab_tutorial.active_gaze.training.tracking import FoviWandbWriter

    updates = []
    saved = []
    monkeypatch.setattr(wandb, "run", SimpleNamespace(resumed=True, url="test"))
    monkeypatch.setattr(wandb, "config", SimpleNamespace(update=lambda data, **kw: updates.append((data, kw))))
    monkeypatch.setattr(WandbLogWriter, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(FoviWandbWriter, "save_file", lambda self, path: saved.append(path))
    (tmp_path / settings_file).write_text('{"camera": "hybrid"}')
    (tmp_path / "gaze_reset_cache.pt").write_bytes(b"test cache")
    (tmp_path / "provenance.json").write_text('{"workflow_id": "resumed-workflow"}')
    writer = FoviWandbWriter(str(tmp_path), "test", "nvidia")
    config = SimpleNamespace(to_dict=lambda: {"num_envs": 4})
    writer.store_config(config, {"evaluation": {"enabled": True}})
    assert len(updates) == 3
    assert updates[0][0]["isaaclab_fovi"] == {"camera": "hybrid"}
    assert str(tmp_path / settings_file) in saved
    assert str(tmp_path / "gaze_reset_cache.pt") in saved
    assert all(kwargs == {"allow_val_change": True} for _, kwargs in updates)
    assert updates[-1][0]["train_cfg"]["evaluation"]["enabled"]


@pytest.mark.parametrize("rnd", [False, True])
def test_eval_resets_cannot_cross_gae_rollout_boundaries(tmp_path, rnd):
    """Check real PPO targets on both sides of eval against an independent GAE.

    Evaluation changes the reset generation and reward scale. Any leaked eval
    transition, stale bootstrap observation or carried advantage is observable.
    """
    runner = make_runner(tmp_path, rnd=rnd, video=False, frequency_episodes=1.6)
    control = runner.env.unwrapped
    step = control.step

    def distinct_reward(action):
        obs, reward, done, extras = step(action)
        return obs, reward * (1 + 100 * control.resets), done, extras

    control.step = distinct_reward
    compute = runner.alg.compute_returns
    evaluate = runner.evaluation.run
    records = []
    eval_count = 0

    def checked_returns(obs):
        st = runner.alg.storage
        assert st.step == runner.cfg["num_steps_per_env"]
        torch.testing.assert_close(obs["proprio"], runner.env.get_observations()["proprio"])
        assert (st.observations["proprio"][..., 0] == 2 * eval_count).all()
        bootstrap = runner.alg.critic(obs).detach()
        rewards, values, dones = (x.clone() for x in (st.rewards, st.values, st.dones))
        expected = torch.empty_like(values)
        # Explicit sum of future TD residuals within this rollout only.
        next_values = torch.cat((values[1:], bootstrap.unsqueeze(0)))
        alive = 1 - dones.float()
        deltas = rewards + runner.alg.gamma * alive * next_values - values
        for t in range(st.num_transitions_per_env):
            advantage = torch.zeros_like(bootstrap)
            discount = torch.ones_like(bootstrap)
            for k in range(t, st.num_transitions_per_env):
                advantage += discount * deltas[k]
                discount *= runner.alg.gamma * runner.alg.lam * alive[k]
            expected[t] = advantage + values[t]
        compute(obs)
        torch.testing.assert_close(st.returns, expected)
        advantages = expected - values
        if not runner.alg.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        torch.testing.assert_close(st.advantages, advantages)
        records.append((st.returns.clone(), st.advantages.clone()))

    def checked_evaluation(current):
        nonlocal eval_count
        st = runner.alg.storage
        assert st.step == 0  # Optimizer consumed the completed rollout first.
        assert current.alg.transition.observations is None
        before = (st.returns.clone(), st.advantages.clone())
        fresh = evaluate(current)
        assert st.step == 0
        for saved, actual in zip(before, (st.returns, st.advantages), strict=True):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        eval_count += 1
        assert (fresh["proprio"][:, 0] == 2 * eval_count).all()
        return fresh

    runner.alg.compute_returns = checked_returns
    runner.evaluation.run = checked_evaluation
    runner.learn(3)
    assert len(records) == eval_count == 3
    assert not torch.allclose(records[0][0], records[1][0])
