# IsaacLab Tutorial - SO101 Vial Place

<p align="center">
  <img src="media/demo.gif" alt="SO-101 arm placing a vial into a four-hole rack" width="100%">
</p>

This Isaac Lab 3.0 tutorial trains an SO-101 arm to pick up a vial and place it in a four-hole rack, three ways:

| Task ID | Policy inputs | Method |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | full state | PPO |
| `IsaacTutorial-Place-Vial-SO101-Camera-Distillation` | wrist RGB + proprioception | distilled from the state policy |
| `IsaacTutorial-Place-Vial-SO101-Camera` | wrist RGB + proprioception | PPO from scratch |

The repository is an installable downstream task package: Isaac Lab's CLI discovers it through the `isaaclab.tasks`
entry-point group, so it carries no copies of the train or play launchers. The robot asset, its Sys-ID dynamics, and the
wrist camera come from Isaac Lab's `SO101_CFG`, and the scene matches the real workshop setup for sim-to-real transfer.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Isaac Lab and its runtime dependencies are built from
the upstream Git repository's `develop` branch.

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

Smoke test and benchmark:

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --visualizer none presets=newton_mjwarp

uv run isaaclab benchmark runtime --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --num_steps 1000 --warmup_steps 50 --visualizer none presets=newton_mjwarp
```

Camera tasks add the renderer preset: `presets=newton_mjwarp,newton_renderer`.

## 1. Train the state policy

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 800 --seed 42 --run_name state \
  --visualizer none presets=newton_mjwarp
```

Checkpoints are written to `logs/rsl_rl/so101_vial_state/<run>/`; the final one is `model_799.pt`.

## 2. Distill it into a wrist-camera policy

Pass the finished state checkpoint as the teacher:

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 1024 --max_iterations 1600 --seed 42 --run_name distillation \
  --checkpoint logs/rsl_rl/so101_vial_state/<run>/model_799.pt \
  --visualizer none presets=newton_mjwarp,newton_renderer
```

This is RSL-RL's standard student–teacher distillation (DAgger): the camera student acts in the environment, the state
teacher labels every visited state with the action it would take, and the student regresses those labels. The only
addition is that labels are clipped to the `[-1, 1]` action range the environment actually executes
(`agents/distillation.py`). Because training episodes start from every phase of the task, the student sees the whole
task from the first iteration without teacher-driven rollouts, replay buffers, or schedules. Checkpoints are written to
`logs/rsl_rl/so101_vial_camera_distillation/<run>/`.

## 3. Train a wrist-camera policy from scratch

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 1024 --max_iterations 5000 --seed 42 --run_name vision \
  --visualizer none presets=newton_mjwarp,newton_renderer
```

The actor is the same CNN as the distillation student; the critic is an MLP on the privileged state (asymmetric
actor–critic). Checkpoints are written to `logs/rsl_rl/so101_vial_camera/<run>/`.

## Play and evaluate

Watch a policy (16 environments for state, 8 for camera tasks):

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 \
  --checkpoint /path/to/model.pt --deterministic --visualizer newton presets=newton_mjwarp
```

Measure success on the acceptance audit — the 128 canonical home-pose starts, each played eight times, one episode
per environment. The callback stops play after exactly 1,024 episodes and prints one `SO101_EVAL_RESULT` JSON line:

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 --checkpoint /path/to/model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp
```

Use the matching task ID (and add `newton_renderer`) to evaluate the camera policies.

### Reference results

Seed 42, one RTX 6000 Ada per run:

| Policy | Success (1,024 episodes) | Grasp | Lift | Insert | Unsafe rack impacts | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| State (PPO) | 99.2% (1016) | 99.4% | 99.4% | 99.2% | 0 | 34 min |
| Wrist camera, distilled | 98.4% (1008) | 100% | 92.2% | 98.6% | 0 | 31 min |
| Wrist camera, PPO from scratch | 99.0% (1014) | 100% | 98.7% | 99.5% | 0 | 3.3 h |

The table is one fresh end-to-end run from a clean checkout. Camera policies vary noticeably between training runs:
five distillation runs (seeds 42, 42, 43, 44, 45) audited at 98.4%, 91.7%, 97.6%, 97.6%, and 91.9%, and two further
from-scratch visual runs (seed 42 again, seed 43) audited at 98.8% and 93.1%. State policies vary by a few tenths of a percent (99.2–99.9%). Repeated audits of one checkpoint differ by
a few tenths of a percent because the renderer and observation noise are not seeded with the policy.

## Task design

**Control.** At 30 Hz the policy outputs five bounded arm-joint position increments (0.033 rad per step) and one
bounded jaw increment (0.02 rad per step). Targets are relative to the measured joint positions, matching the real
controller's interface. The vial is a free rigid body: nothing attaches it to the gripper or writes its pose after reset.

**Observations.** The state policy sees joint positions, velocities, targets, the previous action, the gripper and
vial poses in the robot frame, the vial in the rack frame, and the three latched milestone flags below. The camera
policies see a 64 × 48 wrist RGB image and proprioception (joint positions, velocities, targets, previous action).
Image exposure, contrast, white balance, and brightness are randomized per episode; proprioception carries small
uniform noise. The critic of every task uses the privileged state plus jaw-contact flags.

**Milestones and success.** Three physical facts are latched once per episode and each pays a one-time reward:

```text
grasped   both jaws touch the vial while it is held ≥ 6 mm above the mat        +10
lifted    the vial's lowest point has cleared the top of the rack                  +20
inserted  the vial's tip is inside the target opening and the vial is upright      +40
```

Success is purely physical: the released vial rests upright inside the target opening for ten consecutive steps
(+200, and the episode ends). Two dense shaping terms guide the sparse milestones. `approach_progress` (weight 1.0)
pays the decrease in jaw-to-cap distance until the grasp: as a difference of potentials it has no incentive to hover,
and it has the same gradient 25 cm from the vial as 2 cm away, which is what lets a pixels-only policy discover the
approach from the home pose. `held_goal` (weight 0.1) is a Gaussian bump that brings the held vial to the insertion
pose until its tip enters the opening. There is deliberately no such bump around the vial before the grasp: a visual
policy whose grasp is still unreliable learns to hover in it instead of grasping. Losing the vial costs −50, and small
action-rate and joint-velocity penalties keep motion smooth.
Hard vial–rack impacts (> 20 N) are logged as a safety metric.

**Exploration.** Both PPO actors use a Gaussian head whose standard deviation is clamped to [0.05, 0.3]
(`BoundedGaussianDistributionCfg`). With clipped actions and an entropy bonus, an unclamped std inflates without bound
(the state policy drifted past 1.5, the visual policy past 10), and noise that large random-walks the gripper by
centimetres over the approach, so a visual policy that has not yet learned to reach the vial never can.

**Resets.** Training episodes start from a checked-in dataset of 1,024 physics-validated states, 128 for each of
eight task phases reached by executing the task from the real robot's home pose:

```text
home → pregrasp → grasp → lift → reorient → transport → insert → release
```

Sampling every phase uniformly gives dense learning signal along the whole task from the first iteration; evaluation
and play use only the home-pose starts. Milestones already true in a loaded state are restored but never rewarded.

## Layout

```text
src/isaaclab_tutorial/
  assets/                         workshop USD assets and the reset dataset
  tasks/place_vial/
    mdp/                          actions, reset events, observations, rewards, milestones, terminations
    reset/                        reset dataset format, phase weights, and the reset generator
    config/so101/
      env_cfg.py                  scene, control, physics, and the state task
      camera_env_cfg.py           wrist-camera and distillation tasks
      agents/                     RSL-RL PPO and distillation configurations
  utils/evaluation.py             exact 1,024-episode audit for `isaaclab play`
tests/                            configuration and behavioural contracts
```

To add another robot, create a sibling of `config/so101` and register its task IDs there. To add another task, create
a sibling of `tasks/place_vial` with its own MDP and robot configurations.

## Reset dataset maintenance

The checked-in reset dataset is ready for training. To regenerate or inspect a candidate without overwriting it:

```bash
uv run generate-so101-resets --output checkpoints/reset_poses.pt --visualizer none presets=newton_mjwarp
uv run view-so101-resets --dataset checkpoints/reset_poses.pt --visualizer newton presets=newton_mjwarp
```
