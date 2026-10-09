"""A training launch is not qualification, and every child must see exactly one GPU."""

import json
import sys
from types import SimpleNamespace

import pytest

from isaaclab_tutorial.utils import train_transfer


@pytest.mark.parametrize("qualification_rate", [0.96, 0.80])
@pytest.mark.parametrize("warm_start", [True, False])
@pytest.mark.parametrize("kind", ["state", "vision"])
def test_campaign_gpu_scope_and_independent_qualification(tmp_path, monkeypatch, qualification_rate, warm_start, kind):
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_transfer",
            "--gpu",
            "3",
            "--kind",
            kind,
            "--full-colliders",
            *(["--checkpoint", str(checkpoint)] if warm_start else ["--uniform-starts"]),
            "--output",
            str(output),
            "--seed",
            "9",
            "--blocks",
            "1",
            "--shoulder-scale",
            "0.04",
            "--steps-per-env",
            "32",
        ],
    )
    audits = []

    def run(command, *, cwd, env, stdout, stderr):
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        assert "env.actions.arm_action.scale.shoulder_lift=0.04" in command
        spawn = (
            "camera_env_cfg:_spawn_so101_for_wrist_camera"
            if kind == "vision"
            else "env_cfg:_spawn_so101_with_camera_overrides"
        )
        assert f"env.scene.robot.spawn.func=isaaclab_tutorial.tasks.place_vial.config.so101.{spawn}" in command
        if "train" in command:
            assert "agent.num_steps_per_env=32" in command
            assert ("--checkpoint" in command) == warm_start
            assert ("--reset_optimizer" in command) == warm_start
            if not warm_start:
                assert "env.events.reset_from_dataset.params.phase_weights=[1,1,1,1,1,1,1,1]" in command
            model = cwd / "logs/rsl_rl/test/run/model_199.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"trained checkpoint")
        else:
            count = int(env["SO101_EVALUATION_EPISODES"])
            audits.append((count, command[command.index("--seed") + 1]))
            rate = 0.96 if count == 256 else qualification_rate
            (cwd / "evaluation.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "episodes": count,
                            "successes": int(count * rate),
                            "success_rate": rate,
                        }
                    }
                )
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_transfer.subprocess, "run", run)
    train_transfer.main()
    manifest = json.loads((output / "campaign.json").read_text())
    expected = [256, 256, 1024, 1024, 1024, 1024] if kind == "vision" else [256, 1024, 1024]
    assert len(audits) == len(expected) and len({seed for _, seed in audits}) == len(expected)
    assert [count for count, _ in audits] == expected
    if qualification_rate >= 0.95:
        assert manifest["status"] == "qualified"
        assert "selected_checkpoint" in manifest
    else:
        assert manifest["status"] == "budget_exhausted_unqualified"
        assert "selected_checkpoint" not in manifest
