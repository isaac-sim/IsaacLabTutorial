# Place a vial with SO-101: state and vision RL on Newton

This tutorial builds one manager-based Isaac Lab task in two policy variants. The robot must pick up a horizontal
vial, rotate it upright, move it into the rack, and release it. Both variants use Newton/MJWarp at 120 Hz, a 30 Hz
control loop, and RSL-RL PPO:

- `Isaac-Place-Vial-SO101`: state actor and critic, 4,096 environments by default.
- `Isaac-Place-Vial-SO101-Camera`: dual-RGB actor and privileged state critic, 4,096 environments by default.

The tutorial deliberately covers reinforcement learning only. Teleoperation, imitation learning, LeRobot, GR00T,
and hardware deployment are outside its scope.

## Quick start

Python 3.12 and `uv` are required. Resolve and install the locked project:

```bash
uv sync
```

Smoke-test the state task on a small GPU allocation:

```bash
uv run isaaclab zero_agent --task so101_vial_lift:Isaac-Place-Vial-SO101 --num_envs 8 presets=newton_mjwarp
uv run isaaclab random_agent --task so101_vial_lift:Isaac-Place-Vial-SO101 --num_envs 8 presets=newton_mjwarp
```

Train and play a state policy:

```bash
uv run isaaclab train --task so101_vial_lift:Isaac-Place-Vial-SO101 presets=newton_mjwarp
uv run isaaclab play --task so101_vial_lift:Isaac-Place-Vial-SO101 presets=newton_mjwarp --checkpoint logs/rsl_rl/so101_vial_state/<run>/model_<iteration>.pt
```

For smaller GPUs, append `--num_envs 64` (or any suitable value). Both checked-in training defaults are 4,096 as
the throughput-oriented target. The dual-camera run needs substantially more memory than the state run, so use an
explicit smaller override if 4,096 rendered environments do not fit the available GPU.

## 1. Reproducible setup with `uv`

The root `pyproject.toml` requires Python 3.12 and sources the Isaac Lab packages from the upstream `develop`
branch. `uv.lock` freezes the resolved commit, Newton release, RSL-RL version, and upstream compatibility
overrides. Refresh the resolution only when intentionally updating the tutorial:

```bash
uv lock --upgrade
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run codespell
```

The module prefix in `so101_vial_lift:Isaac-...` makes Gym import this external package before looking up the task.
There are no copied training launchers or simulator shell wrappers.

## 2. Inspect the multiphysics SO-101 asset

The complete supplied archive is extracted under `src/so101_vial_lift/assets/so101/`; the original ZIP remains local
and ignored. Large USD and image files are tracked with Git LFS. The interface layer selects:

- `Robot=robot` for Isaac robot and joint metadata.
- `Sensor=sensors` for the calibrated gripper camera payload.
- `Physics=physics` for Newton-authored joint dynamics.

The robot exposes five arm joints plus `gripper`. The asset-validation tests walk every relative USD dependency and
verify the variants, camera prim, and local prop files:

```bash
uv run pytest tests/test_assets.py
```

The workshop vial, rack, and mat are local Apache-2.0 USD assets, so training never depends on a workstation-only
path or asset server. Attribution is in `THIRD_PARTY_NOTICES.md`.

The original high-detail collision meshes remain in the archive but are disabled for this task: several contain
degenerate planar decomposition fragments that MJWarp cannot solve reliably. One compact box proxy on each
fingertip provides stable, filtered grasp contacts while preserving the supplied visual, joint, dynamics, and sensor
data. A high-grip task material is bound only to those two proxies; the mat and vial retain the scene default
friction. No arm-link, wrist, camera-mount, or decorative hand geometry participates in collision detection.

## 3. Construct the Newton scene

`SO101SceneCfg` creates one active vial, one kinematic rack, the mat, the SO-101, and filtered contact sensors on
the fixed and moving jaws. The vial starts horizontal at a randomized reachable pose. At each reset its XY location
and yaw change while the rack stays fixed, which makes the target simple to interpret in rack-local coordinates.

`PhysicsCfg` exposes the `newton_mjwarp` preset. Simulation runs at 120 Hz with 12 Newton substeps; action
decimation 4 produces the 30 Hz control rate. Start with a few environments when debugging geometry:

```bash
uv run isaaclab random_agent --task so101_vial_lift:Isaac-Place-Vial-SO101 --num_envs 4 presets=newton_mjwarp --visualizer newton
```

## 4. Actions, observations, resets, rewards, and success

The six policy outputs are relative joint-position increments. Arm deltas are scaled by 0.05 rad, the gripper
delta by 0.10 rad, and persistent targets are clamped to the asset's soft limits. The task uses the residual-RL
controller decomposition common in contact-rich manipulation: a finite-state controller closes and lifts after
physical bilateral contact, supplies a safe joint-space transport reference, and holds the arm during release.
PPO supplies bounded transport residuals and the gripper release action. This keeps exploration from repeatedly
destroying grasps while leaving contact acquisition, residual positioning, and release timing in the MDP.

MJWarp does not currently expose a batched per-environment fixed-joint API. After bilateral jaw contact, the action
term therefore maintains the measured vial-to-gripper transform at simulation rate until the policy opens the
gripper at a valid insertion pose. Once released inside the rack capture volume, the kinematic fixture holds the
vial during the 15-step confirmation dwell. These two explicit constraints are part of the task definition, not
privileged observations; state and camera actors use the same controller.

The state actor and critic each receive joint position and velocity, previous action, end-effector state, vial
state, the vial position in rack coordinates, and irreversible grasp/lift flags. Resets perturb robot joints and the
horizontal vial pose without changing other environments in the batch.

Rewards progress through reach, bilateral grasp acquisition and retention, lift height, upright rotation, transport,
insertion, and released placement. Potential-difference terms reward progress toward the transport joint goal and
rack-local target without paying the policy for remaining still. Later dense terms switch off after their milestone,
and every stage after grasping is gated by stored episode progress. This prevents reward farming and prevents a
policy from earning placement success by pushing an unlifted vial into the rack. Action changes, excessive joint
velocity, drops, and workspace loss are penalized.

Success is an instance-owned manager term. It requires a prior bilateral grasp and 5 cm lift, rack-local placement,
vertical alignment above 0.8, release, speed below 0.1 m/s, and 15 consecutive control steps. Partial environment
resets clear only the selected history rows. Success, timeout, vial loss, and unstable joint velocity terminate an
episode. The environment logs grasp, lift, valid-placement, confirmed-success, reward-component, and
time-to-success metrics.

The pure tensor tests cover transforms, bounds, gating, consecutive confirmation, failure interruption, and partial
resets:

```bash
uv run pytest tests/test_geometry.py tests/test_progress.py
```

## 5. Train and evaluate the state policy

The state PPO actor and critic use `[256, 256, 128]` MLPs. Training collects 32 steps per environment, performs five
epochs over four minibatches, and uses `gamma=0.99`, `lambda=0.95`, and an adaptive learning rate starting at
`1e-3`.

```bash
uv run isaaclab train --task so101_vial_lift:Isaac-Place-Vial-SO101 --num_envs 4096 --max_iterations 10 --run_name final_canonical_from_scratch presets=newton_mjwarp
uv run isaaclab train --task so101_vial_lift:Isaac-Place-Vial-SO101 presets=newton_mjwarp --checkpoint latest
```

The exact evaluation callback counts terminal episodes rather than averaging per-step logger values. It stops after
exactly 1,000 randomized episodes and prints one `SO101_EVAL_RESULT` JSON record:

```bash
uv run env SO101_EVAL_EPISODES=1000 isaaclab play --task so101_vial_lift:Isaac-Place-Vial-SO101 --num_envs 1024 --checkpoint logs/rsl_rl/so101_vial_state/2026-08-19_22-56-19_final_canonical_from_scratch/model_9.pt --deterministic --external_callback so101_vial_lift.evaluation.install_episode_counter --visualizer none presets=newton_mjwarp
```

The clean seed-42 run reached 997/1,000 confirmed placements (99.7%) from scratch. The local checkpoint is ignored
by Git; its relative path and SHA-256 are recorded in `checkpoints/manifest.json`. Upload it to the Isaac Lab Nucleus
checkpoint root and add the resulting URI to the manifest when Nucleus credentials are available.

## 6. Add dual cameras and an asymmetric CNN policy

The camera actor receives only two normalized CHW RGB tensors, joint positions and velocities, and the previous
action. It never receives vial pose, rack pose, or progress flags. The critic keeps the full privileged state group.

Both views are 64×64 at 30 Hz. `ego_camera` reads the calibrated `wowrobo_2MP_camera` prim under the gripper, retaining
its intrinsics and mount translation with an optical-axis correction for the Newton renderer; the external camera
uses the workshop D455-style pose. `renderer=newton_renderer` selects the native Newton Warp renderer.
RSL-RL creates one CNN encoder per image key, concatenates both embeddings with proprioception, and applies the MLP
head. Camera PPO uses eight minibatches and a fixed `1e-4` learning rate.

```bash
uv run isaaclab zero_agent --task so101_vial_lift:Isaac-Place-Vial-SO101-Camera --num_envs 8 presets=newton_mjwarp,newton_renderer
uv run isaaclab random_agent --task so101_vial_lift:Isaac-Place-Vial-SO101-Camera --num_envs 8 presets=newton_mjwarp,newton_renderer
uv run isaaclab train --task so101_vial_lift:Isaac-Place-Vial-SO101-Camera --num_envs 4096 --max_iterations 10 --run_name camera_fixed_view_from_scratch presets=newton_mjwarp,newton_renderer
uv run env SO101_EVAL_EPISODES=1000 isaaclab play --task so101_vial_lift:Isaac-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint logs/rsl_rl/so101_vial_camera/2026-08-19_23-22-35_camera_fixed_view_from_scratch/model_9.pt --deterministic --external_callback so101_vial_lift.evaluation.install_episode_counter --visualizer none presets=newton_mjwarp,newton_renderer
```

The seed-42 camera run reached 996/1,000 confirmed placements (99.6%). The actor receives only the two images and
proprioception; the full task state remains confined to the critic during training.

## 7. Debugging checklist

- Contacts: inspect each filtered sensor independently. If bilateral contact never rises, first verify the vial body
  path and jaw collision geometry; do not lower the force threshold blindly.
- Camera framing: run one environment with the Newton visualizer and confirm both images change after a reset and
  after arm movement. A static image usually means a prim path or render preset mismatch.
- Rewards: plot each `Episode_Reward/*` component with the progress metrics. Transport or placement reward before
  `lift_rate` indicates broken gating.
- Resets: watch for success counters leaking across only some reset environments. `tests/test_progress.py` is the
  minimal regression test.
- Training: check observations and rewards for finite values with zero/random agents before starting PPO. Resume a
  short run and play its saved model before committing to a long experiment.
- Scaling: reduce `--num_envs` first for out-of-memory failures. Camera tensors and per-view CNN activations make the
  vision task substantially heavier than the state task.

Run the complete local verification suite before training:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run codespell
```
