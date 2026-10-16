# Isaac Lab Tutorial Learning Path

This branch is the exercise scaffold for the completed code on `main`.
Follow the **Train a Robot Arm** module, applying its marked regions in lesson order.
Imports, workshop assets, the reset dataset, and supporting utilities are supplied.
The initial scene intentionally has no robot. Lesson 02 makes inspection runnable;
lesson 04 adds actions, lesson 05 the state teacher, and lesson 06 camera distillation.
Run the upstream tests after completing lesson 06, not against unfinished exercises.

Compare with the completed reference without switching branches:

```bash
git diff main -- src/
```

The reference documentation below describes the completed task and includes
optional upstream workflows outside the course.

# SO-101 vial placement

<p align="center"><img src="media/demo.gif" alt="SO-101 workshop demonstration" width="100%"></p>

An Isaac Lab tutorial that trains an SO-101 arm to pick up a vial lying on the desk and place it upright in
**any of the four holes** of a rack. A state policy learns the task with PPO. A wrist-camera policy is then
distilled from it, and that camera policy is the one that runs on the real robot.
Training uses **Newton MJWarp physics** and the **Newton renderer** for the camera.

## Setup

Linux x86-64 or Windows 11, Python 3.12, Git LFS, an NVIDIA GPU with a CUDA 13-compatible driver, and `uv`.
Every command below is a single line, so it runs unchanged in bash, PowerShell and Command Prompt.

```bash
git lfs pull
uv sync --locked
uv run pytest -q
```

Check that the task builds and steps before training:

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp
```

## Tasks

| Task | Inputs | Training |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | Full state | PPO |
| `IsaacTutorial-Place-Vial-SO101-Camera` | Wrist RGB (2 frames, 64×48) + proprioception | Distillation from a state teacher, or PPO from scratch |

Add `presets=sim2real` to either task to enable robot-transfer randomization. The physical randomization is
identical in both tasks, so a teacher and its student train under the same physics. See
[docs/SIM2REAL.md](docs/SIM2REAL.md).

## Train

```bash
# State teacher
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp

# Camera student, distilled from the teacher checkpoint
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint /abs/path/to/state/model_799.pt --visualizer none presets=newton_mjwarp,newton_renderer
```

During distillation the student acts and the frozen teacher labels every state it visits. The labels are
clipped to the executed `[-1, 1]` action range. Checkpoints go to `logs/rsl_rl/`.

The camera policy can also be trained from scratch with PPO, without a teacher. The actor sees the same wrist
images and proprioception as the student; the critic sees the privileged state. It reaches about 98% in 2,500
iterations.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --agent rsl_rl_ppo_cfg_entry_point --num_envs 1024 --visualizer none presets=newton_mjwarp,newton_renderer
```

## Evaluate

Play starts every episode from the home pose. The exact-evaluation callback counts one episode per
environment and prints a `SO101_EVAL_RESULT` JSON line. Set `SO101_EVALUATION_OUTPUT=path.json` to also save
per-episode outcomes.

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 1024 --checkpoint /abs/path/to/model.pt --visualizer none --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter presets=newton_mjwarp
```

To evaluate a camera policy, use the `-Camera` task with `presets=newton_mjwarp,newton_renderer`, and add
`--agent rsl_rl_ppo_cfg_entry_point` for one trained from scratch.

Do not pass `--deterministic`. On this pin it also enables Warp-wide deterministic physics, which overflows a
collision-kernel buffer in this scene. Playback already uses deterministic policy actions.

## Task design

**Control and actions**
- The policy runs at 30 Hz, and physics runs at 120 Hz.
- Actions are Isaac Lab relative joint-position targets: at every physics step the target is the
  *measured* joint position plus the scaled action, which matches the real servo interface. The scale is
  0.033 rad for the arm joints and 0.02 rad for the gripper.
- A small penalty discourages pushing against the soft joint limits.

**Rewards and success**
- Three physical milestones (grasp, lift, insertion) each pay once per episode.
- Dense shaping guides the approach to the cap and the held vial toward the nearest hole.
- Success means the vial is seated upright, released and almost still for ten consecutive control steps.

**Resets**
- Episodes start from a physics-validated reset dataset (`assets/reset_poses.pt`). It covers eight phases of
  the task, from the home pose through to release.
- Training samples all phases; evaluation starts from home.

To regenerate or inspect the reset dataset:

```bash
uv run generate-so101-resets --output outputs/reset_poses.pt --visualizer none presets=newton_mjwarp
uv run view-so101-resets --dataset outputs/reset_poses.pt --visualizer newton presets=newton_mjwarp
```

## Layout

```
src/isaaclab_tutorial/
  tasks/place_vial/config/so101/env_cfg.py          state task and the sim2real preset
  tasks/place_vial/config/so101/camera_env_cfg.py   wrist-camera task (inherits the state task)
  tasks/place_vial/config/so101/agents/             RSL-RL PPO and distillation configs
  tasks/place_vial/mdp/                             observations, rewards, terminations, events
  tasks/place_vial/reset/                           reset dataset format and generator
  utils/                                            evaluation, LEAPP export, real-robot scripts
docs/SIM2REAL.md                                    training for and deploying to the real robot
```

PhysX and OVRTX are available for sim-to-sim checks: install them with `uv sync --extra ovphysx --extra ovrtx`
and run with `presets=physx,ovrtx`. The tutorial is not tuned for those backends.
