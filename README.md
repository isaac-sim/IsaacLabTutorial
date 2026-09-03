# SO-101 vial placement

This Isaac Lab 3.0 tutorial trains an SO-101 arm to pick up a vial and place it in a four-hole rack. It is an
installable downstream task package: the shared Isaac Lab CLI discovers the package through the `isaaclab.tasks`
entry-point group, so this repository does not carry copies of the train, play, or benchmark launchers.

The registered tasks are:

- `IsaacTutorial-Place-Vial-SO101` for state observations.
- `IsaacTutorial-Place-Vial-SO101-Camera` for 64 × 64 wrist RGB and proprioception.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. This checkout resolves the Isaac Lab packages from an
upstream `develop` checkout at `../IsaacLab`; keep the two repositories next to one another.

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

## Train and play

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 800 --seed 42 \
  --run_name so101_vial_seed42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1 --checkpoint checkpoints/model.pt --deterministic \
  --visualizer newton presets=newton_mjwarp
```

The optional evaluation callback runs the tracked phase-zero starts once each and prints an `SO101_EVAL_RESULT` JSON
record:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 --checkpoint checkpoints/model.pt --deterministic \
  --external_callback so101_vial_place.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp
```

A reference rollout is available at [`media/demo.mp4`](media/demo.mp4).

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
src/so101_vial_place/
  assets/                         workshop and reset assets
  tasks/
    place_vial/
      mdp/                        actions, events, observations, rewards, terminations
      reset/                      reset dataset and generation logic
      config/
        so101/
          agents/                 RSL-RL configurations and models
          state_env_cfg.py        state task configuration
          camera_env_cfg.py       wrist-camera task configuration
          control.py              SO-101 command conventions
          physics.py              SO-101 contact material setup
  utils/                          evaluation and maintainer utilities
tests/                            behavioral and configuration contracts
```

To add another robot, create a sibling of `config/so101` and register its task IDs there. To add another task, create
a sibling of `tasks/place_vial` with its own MDP and robot configurations.
