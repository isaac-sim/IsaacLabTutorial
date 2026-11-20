# Isaac Lab Tutorial

Train and evaluate policies for two robotics tasks in Isaac Lab:

- [**SO-101 vial placement**](#so-101-vial-placement). Teach a robot arm to
  pick up a vial and place it upright in a rack, then distill the state
  teacher into a wrist-camera policy.
- [**Microduck locomotion**](#microduck-locomotion). Teach a two-legged robot
  to balance, walk at a requested speed, and turn. Compare how changes to
  the rewards affect its gait.

The [Getting Started With Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html)
course walks through both tasks in **Train a Robot Arm** and **Train a
Microduck**.

## Setup

Linux x86-64 or Windows 11, Python 3.12, Git LFS, an NVIDIA GPU with a CUDA 13-compatible driver, and `uv`.

Install [Git Large File Storage (Git LFS)](https://git-lfs.com/) on the machine
where you'll clone the project. Git LFS is required for the repository's media
and assets, including when course checkout commands skip downloading them.
On Ubuntu, install it with:

```bash
sudo apt-get update
sudo apt-get install git-lfs
```

For Windows or other Linux distributions, follow the
[Git LFS installation instructions](https://git-lfs.com/).
Confirm that `git lfs version` prints a version before cloning.

Follow [Project Setup](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/01-project-setup.html) in **Train a Robot Arm** to
check out the pinned scaffold and set up the environment.

Both projects can use the same `IsaacLabTutorial` directory. The robot arm
lessons use the local `learning-path` branch, and the Microduck lessons use
`microduck-course`. Commit your lesson edits before switching branches.

### Optional Recording and Experiment Tracking

Add `--extra video` to `uv run` commands that record videos with `--video` or
`record_video`. This installs Isaac Lab's MoviePy dependency through `isaaclab[video]`.

For Weights & Biases (W&B) experiment tracking, sign in with:

```bash
uv run --extra wandb wandb login
```

Then add `--extra wandb` to the training command and set `agent.logger=wandb` and
`agent.wandb_project=YOUR_PROJECT`. The tutorial's `wandb` extra forwards to `isaaclab[wandb]`
and temporarily pins a compatible W&B version in [pyproject.toml](pyproject.toml), because
the Isaac Lab logger uses the legacy `wandb.util.generate_id` API. Both extras can be
combined as `uv run --extra video --extra wandb` when recording and tracking a run.

## SO-101 Vial Placement

This branch provides the SO-101 exercises. The starter scene has no robot.
Lesson 02 makes inspection runnable, lesson 04 adds actions, lesson 05
adds the state teacher, and lesson 06 adds camera distillation. Use the
commands below after completing the corresponding lessons. Run
`uv run pytest -q` after lesson 06.

To compare your work with the pinned reference tag created in Project Setup:

```bash
git diff tutorial-reference -- src/
```

<p align="center"><img src="media/demo.gif" alt="SO-101 workshop demonstration" width="100%"></p>

The SO-101 picks up a vial lying on the desk and places it upright in any of
the four holes in the rack. A state policy learns the task with PPO. A
wrist-camera policy is then distilled from it and can run on the real robot.
Training uses Newton MJWarp physics and the Newton renderer for the camera.

Check that the task builds and steps before training:

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp
```

### Task Variants

| Task | Inputs | Training |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | Full state | PPO |
| `IsaacTutorial-Place-Vial-SO101-Camera` | Wrist RGB (2 frames, 64×48) + proprioception | Distillation from a state teacher, or PPO from scratch |

Add `presets=sim2real` to either task to enable robot-transfer randomization. The physical randomization is
identical in both tasks, so a teacher and its student train under the same physics. See
[docs/SIM2REAL.md](docs/SIM2REAL.md).

### Train

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

### Evaluate

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

### Task design

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

## Microduck Locomotion

Train a two-legged robot to balance and follow walking commands, then compare
how different rewards affect its gait. To build the Microduck tasks, follow
**Train a Microduck** and use the
[`learning-path-microduck` scaffold](https://github.com/anthonyjclark/IsaacLabTutorial/tree/learning-path-microduck).
See [Microduck Locomotion in the completed tutorial](https://github.com/anthonyjclark/IsaacLabTutorial/blob/lp/adding-microduck/README.md#microduck-locomotion)
for playback, training, and evaluation commands.

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
and run with `presets=physx,ovrtx`. The SO-101 task is not tuned for those backends.
