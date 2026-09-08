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

### Sim2sim: play Newton policies on PhysX (and back)

The task runs on two physics backends selected from the command line: `presets=newton_mjwarp` (default, Newton with
the MuJoCo-Warp solver) and `presets=physx`, which in this repository means the standalone **OV PhysX** runtime
(`ovphysx`, no Isaac Sim required; with Isaac Sim installed the same preset selects Isaac Sim PhysX). All PhysX
numbers below are OV PhysX. Install the standalone runtime and its renderer with the optional extras:

```bash
uv sync --extra ovphysx --extra ovrtx   # add --extra video for MP4 export of the rollout comparisons
```

Then audit any checkpoint on the other backend by changing only the preset:

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 --checkpoint /path/to/newton_model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=physx
```

The PhysX preset (`WORKSHOP_SO101_CFG`, `PhysicsCfg` and the vial spawn in `env_cfg.py`) maps the Newton
system-identification values onto PhysX and fixes two solver-level differences that otherwise make every grasp fail:

| Newton quantity | PhysX equivalent in the preset | Why |
| --- | --- | --- |
| `newton:armature`, `newton:friction`, `newton:damping` (per joint) | actuator `armature`, `friction`/`dynamic_friction`, `viscous_friction` (`newton:damping` is per degree; the preset converts it) | same joint dynamics parameters |
| USD variant `Physics=physics` | `Physics=physx` | the asset ships one layer per backend |
| MJWarp Newton solver, 100 iterations, 2 substeps | `OvPhysxCfg(rigid_body_position_iteration_count=128, articulation_position_iteration_count=128)` on the `physx` preset (scene-wide defaults, added to Isaac Lab in the fork) | at PhysX's default 4 rigid-body iterations the jaw/vial pinch is under-converged: the jaw closes through the vial regardless of friction; 64 is not enough, 128 holds |
| Newton contact stiffness/damping 1.57e5 / 1.12e3 | compliant PhysX contact material with the same stiffness/damping, friction combine `max` | rigid PhysX contacts release the vial as soon as the pinch relaxes during the lift; letting the pads sink 1-2 mm into the cap keeps it on the cap ledge like Newton |
| vial friction randomization range 0.7-1.3 | x0.5 on OV PhysX (`PHYSX_VIAL_FRICTION_SCALE`; pair friction then equals the pads' 0.7 under the `max` combine rule) | with the Newton range the closing jaws drag the vial into a different in-hand pose on OV PhysX (20% success from the pregrasp phase vs ~100% on Newton); lowering the vial's friction to the pads' level gives 31% from pregrasp and 55% from home; lowering the pads too (all shapes x0.5) drops the vial (19%) |
| gripper drive stiffness (USD: 68.25 N m/rad) | x0.6 on OV PhysX (`PHYSX_GRIPPER_STIFFNESS_SCALE`, separate `gripper` actuator group) | the gripper stalls against the vial at +0.065 rad on Newton but at +0.02 rad on OV PhysX with the compliant contact, so the vial sits deeper in the jaws and the gripper observation is off-distribution; x0.6 reproduces Newton's stall angle and retention curve within 0.01 rad and raised the Newton policy's success on OV PhysX from 13.7% to 32% |
| sys-ID joint friction / viscous friction | the same values times per-joint factors (`PHYSX_JOINT_FRICTION_SCALE`: x1.5 pan/shoulder/roll, x2 gripper; `PHYSX_JOINT_VISCOUS_SCALE`: x2 pan/roll/gripper, x3 shoulder/wrist_flex, x1.5 elbow) | with identical numbers the OV PhysX joints respond faster than Newton's, most of all the loaded shoulder (2.4x); the factors were fitted against Newton's step and sinusoid responses and bring the joint traces to 0.001-0.05 rad RMSE (see `SIM2SIM_ISAACLAB_ISSUES.md (issue 3)`) |

Newton is untouched by these settings (it ignores `physx*` schema attributes), and its audits are unchanged.

Side-by-side rollouts from identical canonical starts (four environments, 10 s at 30 Hz):

- `docs/sim2sim/newton_policy_newton_vs_physx.mp4` - the Newton-trained state policy on Newton (left) and on PhysX (right)
- `docs/sim2sim/physx_policy_physx_vs_newton.mp4` - the PhysX-trained state policy on PhysX (left) and on Newton (right)
- `docs/sim2sim/newton_dr_contact_policy_newton_vs_physx.mp4` - the Newton-trained contact-DR state policy (`-DR-Contact`,
  1,600 iterations; 76% on PhysX) on Newton (left) and on PhysX (right)

Record side-by-side rollouts with the `-Record` task (the state task plus a fixed third-person camera; any state
checkpoint plays) and compose them into a video (`uv sync --extra video` for MP4 output, GIF needs no extra):

```bash
RECORD_OUT=/tmp/rec/newton_on_newton RECORD_ENVS=6 RECORD_STEPS=300 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Record --num_envs 6 --checkpoint /path/to/newton_model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.sim2sim_video.install_recorder --visualizer none presets=newton_mjwarp,newton_renderer
RECORD_OUT=/tmp/rec/newton_on_physx RECORD_ENVS=6 RECORD_STEPS=300 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Record --num_envs 6 --checkpoint /path/to/newton_model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.sim2sim_video.install_recorder --visualizer none presets=physx,ovrtx
uv run python -m isaaclab_tutorial.utils.sim2sim_video /tmp/rec/newton_on_newton /tmp/rec/newton_on_physx \
  --labels "Newton policy on Newton" "Newton policy on PhysX" --out newton_policy_sim2sim.mp4
```

Canonical starts are sequential, so environment *i* starts from the same state on both backends.

Two more gaps sit outside the physics preset:

- **Renderer.** The standalone OVRTX renderer (`presets=physx,ovrtx`) cannot compile the `OmniPBR` MDL materials the
  workshop assets used to carry (the vendored `OmniPBR.mdl` imports modules that are not shipped next to it) and painted
  the vial and rack red. The assets now define plain `UsdPreviewSurface` materials with the same colours, which both
  renderers draw identically.
- **Speed.** OV PhysX throughput at 4,096 environments is set by the solver iterations the pinch needs: 80k steps/s at
  PhysX's default 4 iterations (Newton: 89k), 47k at 32, 18k at the 128 the vial and the articulation both require
  (fewer iterations, PGS, or external-forces-every-iteration all let the jaw close through the vial). A PhysX
  state-policy run therefore takes roughly 3 h instead of 34 min; the compliant contact material costs only ~4%.

Cross-backend audits of the state policy (1,024 episodes each, `--deterministic`):

| Policy | Trained on | Audited on | Success | Grasp | Lift | Insert |
| --- | --- | --- | --- | --- | --- | --- |
| State | Newton | Newton | 99.4% (1018) | 99.5% | 99.5% | 99.4% |
| State | PhysX (`presets=physx`, 800 iterations) | PhysX | 98.4% (1008) | 98.8% | 98.8% | 98.4% |
| State | PhysX | Newton | 66.5% (681) | 87.4% | 85.7% | 72.8% |
| State | Newton | PhysX | 55.4% (567) | 94.3% | 90.0% | 61.7% |
| Wrist camera, distilled | Newton | PhysX + OVRTX | 3.0% (31) | 64.9% | 32.8% | 5.5% |
| State, actuator domain randomization (`-DR`, 800 it) | Newton | Newton | 76.8% (786) | 94.7% | 92.9% | 81.1% |
| State, actuator domain randomization (`-DR`) | Newton | PhysX | 2.9% (30) | 68.1% | 34.3% | 4.8% |
| State, wide domain randomization (`-DR-Wide`, 800 it) | Newton | Newton | 95.3% (976) | 98.4% | 97.7% | 96.9% |
| State, wide domain randomization (`-DR-Wide`) | Newton | PhysX | 2.5% (26) | 51.3% | 27.2% | 6.5% |
| State, contact domain randomization (`-DR-Contact`, 800 it) | Newton | Newton | 45.1% (462) | 100% | 100% | 62.4% |
| State, contact domain randomization (`-DR-Contact`, 800 it) | Newton | PhysX | 57.7% (591) | 90.4% | 89.6% | 63.5% |
| State, contact domain randomization (`-DR-Contact`, 1,600 it) | Newton | Newton | 99.7% (1021) | 100% | 99.9% | 99.7% |
| State, contact domain randomization (`-DR-Contact`, 1,600 it) | Newton | PhysX | 76.0% (778) | 95.4% | 90.7% | 78.4% |
| State, MuJoCo `condim = 4` (torsional friction active, 800 it) | Newton | Newton | 99.8% (1022) | 100% | 100% | 98.7% |
| State, MuJoCo `condim = 4` | Newton | PhysX | 36.2% (371) | 66.9% | 58.6% | 45.5% |
| Wrist camera, PPO from scratch | Newton | PhysX + OVRTX | 0.2% (2) | 46.7% | 28.1% | 0.4% |
| Wrist camera, distilled (from the PhysX state policy, 1,600 iterations) | PhysX + OVRTX | PhysX + OVRTX | 93.0% (952) | 98.9% | 84.4% | 94.1% |
| Wrist camera, distilled (PhysX teacher) | PhysX + OVRTX | Newton | 13.2% (135) | 69.0% | 54.0% | 21.6% |
| Wrist camera, PPO from scratch, 1,000 of 5,000 iterations | PhysX + OVRTX | PhysX + OVRTX | 0% | 33.9% | 23.8% | 0% |

Without the solver fixes the PhysX-trained state policy reached only 33.9% on PhysX itself and 0% on Newton; with the
128-iteration fix alone it reached 98.4% on PhysX and 68.3% on Newton. Newton-trained policies remain hard to move to
PhysX. Per-phase diagnostics locate the whole residual gap before the lift: episodes that start after the grasp succeed
91-100% on PhysX, home starts 54-60% and pregrasp starts 35%. In the pregrasp states the jaws are already clamped on
the vial on both backends, but PhysX reports ~28 N of jaw force where Newton reports ~17 N at the same gripper angle
(both drives saturate at the 3.35 N m effort limit), and the Newton policy's learned reorientation (a slightly
off-centre grasp that lets the vial swing upright while lifting) does not happen on PhysX: the vial stays horizontal
and ends lying on the rack. Mat friction, jaw-pad friction, PhysX torsional patch radius, vial angular damping and a
Newton retrain with torsional friction active (`condim = 4`, see issue 13) all made transfer worse or left it unchanged.
The sim2sim route here is domain randomization only (no training on the target backend). Randomizing what
actually differs between the backends (the vial's friction, 0.2-1.3 per reset, and its mass, 12-30 g; `-DR-Contact`)
and training to convergence lifts Newton -> PhysX from 55.4% to 76.0% while keeping 99.7% on Newton. Actuator
randomization alone (`-DR`, `-DR-Wide`) makes transfer worse (2-3%): the gap is in the contact, not the actuator.
PhysX -> Newton stands at 66.5% without any randomization; a PhysX-trained `-DR-Contact` policy is the open experiment.


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
