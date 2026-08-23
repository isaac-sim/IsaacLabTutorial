# SO-101 vial placement with physical task-horizon resets

This is a Newton-native Isaac Lab 3.0 tutorial task for learning an SO-101 robot to pick up a workshop vial and
place it into a four-hole rack. The public task name is:

```text
IsaacTutorial-Place-Vial-SO101
```

The design goal is a policy that can transfer to the real robot. Robot motion uses ordinary bounded joint-position
increments and the actuator dynamics authored in the supplied Sys-ID USD. The vial remains a free rigid body: there
are no grasp constraints, object writes during an episode, collision proxies, orientation scripts, or hidden motion
controllers.

The state policy is still under development. The latest clean sparse baseline solves transport, insertion, and
release resets, but not the canonical start. See [MULTI_GPU_HANDOFF.md](MULTI_GPU_HANDOFF.md) for exact results and the
next experiment plan. Do not treat older files under `checkpoints/` as accepted results.

## Install and test

This project expects the current Isaac Lab 3.0 development stack and Python 3.12.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

Smoke-test the state and wrist-camera environments:

```bash
uv run isaaclab zero_agent \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 8 presets=newton_mjwarp

uv run isaaclab zero_agent \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 8 presets=newton_mjwarp,newton_renderer
```

Measure end-to-end environment throughput after a CUDA warmup:

```bash
uv run isaaclab benchmark --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 presets=newton_mjwarp

uv run isaaclab benchmark --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 1024 presets=newton_mjwarp,newton_renderer
```

The benchmark reports aggregate control FPS (complete 30 Hz environment
steps) and aggregate physics FPS (120 Hz steps). Both include actions,
physics, sensors, observations, rewards, terminations, and episode resets.
On the reference RTX Pro 6000, 4,096 state environments measured 166k control
FPS / 663k physics FPS. With one fresh 64x48 wrist RGB image per control step,
1,024 camera environments measured 52k image/control FPS / 209k physics FPS.
Rendering—not collision—is the remaining vision bottleneck.

## Why reset across the task horizon

Pick-and-place is a long-horizon task for a small arm with compliant, identified drives. Waiting for random PPO
exploration to discover grasp, lift, reorientation, transport, insertion, and release in one uninterrupted episode
is inefficient.

The reset generator follows one connected physical trajectory and stores valid states throughout it. Every state is
reached through robot targets, gravity, contacts, and the untouched actuator model. Candidates are rejected when the
grasp is not load-bearing, the robot is unstable, the vial is lost, rack contact is unsafe, or the endpoint is not
physically valid.

The tracked reset dataset contains 1,024 states, 128 for each phase:

```text
approach -> pregrasp -> grasp -> lift -> reorient -> transport -> insert -> release
```

Training samples these phases uniformly. Canonical evaluation samples only `approach`, from the real workshop
controller's farther operational start.

Generate a replacement dataset only when physics, assets, or reset logic intentionally change:

```bash
uv run isaaclab generate_resets \
  --batch_size 128 --poses_per_phase 128 \
  --output src/so101_vial_place/assets/reset_poses.pt \
  --visualizer none presets=newton_mjwarp
```

Inspect the stored states independently of training:

```bash
uv run isaaclab view_resets --visualizer newton
```

## Physics, assets, and control

The environment loads the SO-101, vial, rack, and mat from tracked local USD assets. The visual rack retains its four
holes, while eleven box primitives reproduce its base, supports, and four-hole top lattice for collision. Newton does
not build an SDF or run Co-ACD on the rack. The vial uses two cylinder primitives matching its body and cap shoulder.

The robot has one `ImplicitActuatorCfg` so Isaac Lab can route joint targets, but every dynamics field is `None`.
Stiffness, damping, armature, friction, effort limits, and velocity limits therefore load directly from the Sys-ID
USD. A configuration guard rejects Python actuator overrides.

The policy produces six actions at 30 Hz:

- five measured-relative arm-joint position increments, limited to 0.03 rad per command;
- one measured-relative jaw increment, limited to 0.02 rad per command.

All dynamics and soft-limit behavior are resolved by the authored robot model. The policy never commands torques or
object motion.

Newton runs at 120 Hz with two solver substeps. The solver configuration follows the Isaac Lab manipulation examples
while retaining an elliptic friction cone for the two-pad grasp. Light rack guidance is expected during insertion;
only forces above the explicit hard-impact threshold are classified as unsafe diagnostics.

Held insertion and final seating are intentionally different. The gripper lowers the held vial until its tip is
inside the selected opening, opens above the rack, and lets gravity seat the vial more deeply. Success requires a
stable released vial in the deep rack-local bounds for ten consecutive samples.

## A small MDP

The actor observes joint state and target, previous action, end-effector state, vial state, rack-relative target,
placement features, and physical milestone flags. The asymmetric critic additionally observes contacts.

The active sparse baseline reward has six terms:

- a small pre-grasp reach reward;
- one-time physical grasp, lift, and held-insertion milestone rewards;
- terminal physically confirmed success;
- actual vial-loss penalty;
- small action-rate and joint-velocity costs.

There is no staged curriculum, transport teacher, waypoint potential, collision-avoidance reward, or action-specific
opening/closing penalty. The next planned experiment adds one symmetry-aware vial-to-final-goal pose reward because
the sparse baseline does not learn the early horizon quickly enough.

## Train and evaluate the state policy

Train one ordinary PPO job on the full reset distribution:

```bash
env SO101_RESET_CURRICULUM=horizon \
  uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 2000 \
  --run_name state_horizon \
  --visualizer none presets=newton_mjwarp
```

Benchmark the best per-job environment count on the target GPU; the primitive object colliders support the usual
2,048--4,096 state environments when memory permits. On a multi-GPU machine, use each GPU for an independent seed or
reward ablation; distributed PPO is not required.

Evaluate the 128 canonical reset states exactly once each:

```bash
env SO101_RESET_CURRICULUM=initial \
  SO101_EVAL_EPISODES=128 \
  SO101_EVAL_ONCE_PER_ENV=1 \
  SO101_EVAL_SEQUENTIAL=1 \
  uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 128 --checkpoint <checkpoint.pt> --deterministic \
  --external_callback so101_vial_place.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp
```

The evaluator prints one `SO101_EVAL_RESULT` JSON object with grasp, lift, held insertion, success, timeout, loss,
unsafe contact, rack force, and reset-phase metrics. Acceptance must use complete episodes, not training-time
per-step occupancy metrics.

Record a frontal Newton rollout after a policy passes evaluation:

```bash
env SO101_RESET_CURRICULUM=initial \
  SO101_EVAL_EPISODES=1 SO101_EVAL_SEQUENTIAL=0 \
  SO101_VIDEO_OUTPUT_DIR=checkpoints/videos/state \
  SO101_VIDEO_PREFIX=state_seed42 \
  uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 1 \
  --seed 42 --checkpoint <checkpoint.pt> --deterministic \
  --video --video_length 600 \
  --external_callback so101_vial_place.evaluation.install_state_episode_counter \
  --visualizer newton presets=newton_mjwarp
```

## Wrist-camera policy

The camera task uses one physical wrist camera at 64x48 RGB and 30 Hz. It has measured OpenCV intrinsics and
distortion and a fixed, buildable side-bracket transform. The deployed actor receives only:

- wrist RGB;
- joint position and velocity;
- joint target;
- previous action.

The camera actor receives no vial pose, rack pose, contact, phase, or milestone state. Training may use a privileged
asymmetric critic. Episode-consistent exposure, contrast, white-balance, brightness, pixel noise, and proprioceptive
noise are implemented.

Capture samples from the actual policy camera:

```bash
uv run isaaclab capture_wrist \
  --output_dir checkpoints/screenshots/wrist \
  --visualizer none
```

Vision work starts only after a state policy passes the physical acceptance suite. The intended order is:

1. distill the state policy into wrist RGB + proprioception using student-visited states;
2. evaluate and refine that student with a privileged critic;
3. train the same vision actor from scratch as a separate comparison.

The full vision-from-scratch policy must not load or query the state teacher. A privileged critic is still allowed
because it is discarded at deployment.

## Scene screenshots and diagnostics

```bash
uv run isaaclab capture_scene \
  --output_dir checkpoints/screenshots/scene \
  --visualizer newton

uv run isaaclab inspect_robot \
  --task IsaacTutorial-Place-Vial-SO101 \
  --visualizer none presets=newton_mjwarp
```

The default rollout camera is raised, frontal, and angled down at the robot, vial, and selected rack opening.

## Repository map

```text
src/so101_vial_place/
  agents/                 PPO and distillation configuration
  assets/                 local SO-101 and workshop USD assets; reset_poses.pt
  mdp/                    actions, events, geometry, observations/rewards/terminations
  reset/                  reset dataset loader, generator, and reset distribution
  camera_env_cfg.py       wrist-camera task
  control.py              real workshop command conventions and canonical poses
  env_cfg.py              state environment, physics, MDP configuration
  evaluation.py           exact episodic evaluator
  physics.py              Newton shape-contact material setup
  benchmark.py            state and wrist-render throughput measurement
  scene_preview.py        Newton scene screenshot utility
  wrist_preview.py        calibrated wrist-image capture utility
tests/                    configuration and behavioral contracts
MULTI_GPU_HANDOFF.md      current results, findings, and next experiment plan
```

## Current acceptance criteria

A final state policy must achieve at least 90% success from canonical starts over at least 1,000 episodes and several
seeds, with no physics cheats and no obvious irregularity in multiple videos. Report every phase, loss/timeout rates,
and rack forces. Only after that result is accepted should the checkpoint, exports, manifest, and tutorial videos be
promoted.
