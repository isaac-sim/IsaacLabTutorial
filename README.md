# IsaacLab Tutorial - SO101 Vial Place

<p align="center">
  <img src="media/demo.gif" alt="SO-101 arm placing a vial into a four-hole rack" width="100%">
</p>

This Isaac Lab 3.0 tutorial trains an SO-101 arm to pick up a vial and place it in a four-hole rack. It is an
installable downstream task package: the shared Isaac Lab CLI discovers the package through the `isaaclab.tasks`
entry-point group, so this repository does not carry copies of the train, play, or benchmark launchers.

The registered tasks are:

- `IsaacTutorial-Place-Vial-SO101` for state observations.
- `IsaacTutorial-Place-Vial-SO101-Camera` for the direct-from-scratch 64 × 48 visual PPO baseline.
- `IsaacTutorial-Place-Vial-SO101-Camera-Distillation` for training that visual policy from a state teacher.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Isaac Lab and its runtime dependencies are built from
the upstream Git repository's `develop` branch, so no sibling Isaac Lab checkout is required.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

Confirm that the installed Isaac Lab command sees the downstream tasks:

```bash
uv run isaaclab --help
uv run python -c 'import importlib.metadata; print([e.name for e in importlib.metadata.entry_points(group="isaaclab.tasks")])'
```

## Smoke test and benchmark

```bash
uv run isaaclab zero_agent \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 8 --visualizer none presets=newton_mjwarp

uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --num_steps 1000 --warmup_steps 50 \
  --visualizer none presets=newton_mjwarp
```

For the camera task, add the renderer preset and choose an environment count that fits the GPU:

```bash
uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 1024 --num_steps 1000 --warmup_steps 50 \
  --visualizer none presets=newton_mjwarp,newton_renderer
```

## Train the state teacher

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 800 --seed 42 \
  --run_name so101_vial_seed42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1 --checkpoint /path/to/state_model.pt --deterministic \
  --visualizer newton presets=newton_mjwarp
```

Training writes checkpoints under `logs/rsl_rl/so101_vial_state/<run>/`; the final checkpoint is `model_799.pt`.

The optional evaluation callback runs the tracked phase-zero starts once each and prints an `SO101_EVAL_RESULT` JSON
record:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 --checkpoint /path/to/state_model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp
```

## Distill the wrist-camera policy

Pass the finished state checkpoint to the dedicated distillation task. Distillation is single-GPU; use
`CUDA_VISIBLE_DEVICES` to choose that GPU.

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 1024 --max_iterations 800 --seed 42 \
  --checkpoint /path/to/state_teacher.pt \
  --run_name wrist_distillation_seed42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp,newton_renderer
```

The distillation runner uses bounded replay DAgger: it begins with coherent teacher trajectories, gradually adds
student recovery states, and retains a 25% teacher-action floor. The replay buffer preserves recent complete task
trajectories instead of forgetting whichever motion phase was seen in the latest update. A
training-only geometry head gives the camera encoder a dense localization target; the exported actor still consumes
only wrist RGB and proprioception. The last 200 iterations also maintain a sparse stochastic weight average, so the
final `model_799.pt` is stable despite ordinary supervised-learning checkpoint noise. Distillation checkpoints are
written under `logs/rsl_rl/so101_vial_camera_distillation/<run>/`.

Play the distilled policy interactively:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 1 --checkpoint /path/to/distilled_model.pt --deterministic \
  --visualizer newton presets=newton_mjwarp,newton_renderer
```

Evaluate a distilled checkpoint with the same exact 1,024-start contract:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 1024 --checkpoint /path/to/distilled_model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp,newton_renderer
```

As a reference, a fresh seed-42 run on an RTX 6000 Ada took about 37 minutes for state training and 22 minutes for
distillation. Its state teacher succeeded on 1,018 of 1,024 starts (99.4%). Repeated audits of the distilled policy
were 67.6–72.9% successful with no unsafe rack contacts. Small variation between vision audits is expected from the
renderer and observation randomization even with deterministic policy actions.

A reference rollout is available at [`media/demo.mp4`](media/demo.mp4).

## Reset dataset maintenance

The checked-in reset dataset is ready for training. To regenerate or inspect a separate candidate without
overwriting it:

```bash
uv run generate-so101-resets \
  --output checkpoints/reset_poses.pt --device cuda:0 \
  --visualizer none presets=newton_mjwarp

uv run view-so101-resets \
  --dataset checkpoints/reset_poses.pt --device cuda:0 \
  --visualizer newton presets=newton_mjwarp
```

## Task design

The policy produces five bounded, measured-relative arm-joint position increments and one bounded jaw increment at
30 Hz. The robot asset, Sys-ID dynamics, and limits come from Isaac Lab's `SO101_CFG`. The vial remains a free rigid
body throughout each episode; task code does not attach it to the gripper or write its pose after reset.

Training samples a tracked reset dataset across eight physically reached phases:

```text
home → pregrasp → grasp → lift → reorient → transport → insert → release
```

The task-wide MDP and reset implementation is shared by every robot configuration. Robot-specific scenes, controls,
physics settings, registrations, and agent configurations live under `config/so101`.

```text
src/isaaclab_tutorial/
  assets/                         workshop and reset assets
  tasks/
    place_vial/
      mdp/                        actions, events, observations, rewards, terminations
      reset/                      reset dataset and generation logic
      config/
        so101/
          agents/                 RSL-RL configurations and models
          env_cfg.py              task and physics configuration
          camera_env_cfg.py       wrist-camera task configuration
  utils/                          exact rollout evaluation helpers
tests/                            behavioral and configuration contracts
```

To add another robot, create a sibling of `config/so101` and register its task IDs there. To add another task, create
a sibling of `tasks/place_vial` with its own MDP and robot configurations.
