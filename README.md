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

The state task is solved with one compact object-centric shaping term and otherwise ordinary PPO. Canonical evaluation
begins at the exact workshop home pose, with no approach progress serialized in a phase-zero reset. The reproducible
recipe is 600 full-horizon updates followed by 100 canonical-home polishing updates. Fresh seeds 42, 43, and 45
achieved 97.17%, 99.61%, and 99.41% exact canonical success over 1,024 episodes, with no unsafe rack impacts. The
historical 100% teacher was recovered losslessly from the teacher weights stored in both native distillation
checkpoints. Exact
metadata is recorded in `checkpoints/manifest.json`. Its original optimizer was not retained; the three fresh state
checkpoints, rather than the recovered actor, are the end-to-end state reproducibility evidence.

The arm increment is hardcoded at 0.033 rad per 30 Hz command. The reproduced policies complete the task in a mean
9.35--10.42 s while retaining the accepted success and contact-force margins. Larger 0.035 and 0.040 rad steps were
rejected because success and contact-force margins degraded.

Example policy rollouts are stored under `checkpoints/videos/state/`, including
`state_vial_farther_seed45_0000.mp4`. Earlier state and camera checkpoints trained against randomized phase-zero
approach progress are invalidated; vision work will resume only from a teacher trained on the corrected home reset.

The two retained spatial-softmax students still achieve 92.87% and 90.82% exact canonical success using only 64x64
wrist RGB plus proprioception at deployment. Their training is not yet reproducible, however: cleanup omitted the
teacher-rollout warm start and mislabeled two five-update branches from one parent as independent seeds. Fresh exact
audits are recorded below. Vision from scratch also remains open: the retained diagnostic policy achieves 97.56%
grasp and 96.48% lift from canonical home but 0% final placement. See `VISION_SCRATCH_HANDOFF.md` for the evidence.

## Install and test

This project expects the current Isaac Lab 3.0 development stack and Python 3.12.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

Smoke-test the state and wrist-camera environments:

```bash
uv run so101-vial zero_agent \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 8 presets=newton_mjwarp

uv run so101-vial zero_agent \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 8 presets=newton_mjwarp,newton_renderer
```

Measure end-to-end environment throughput after a CUDA warmup:

```bash
uv run so101-vial benchmark --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 presets=newton_mjwarp

uv run so101-vial benchmark --task IsaacTutorial-Place-Vial-SO101-Camera \
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
canonical home -> pregrasp/closure -> grasp -> lift -> reorient -> transport -> insert -> release
```

Training samples these phases uniformly. Canonical evaluation samples only `canonical home`: every row uses the exact
real workshop operational joint pose, with the vial visibly outside the open jaws. The policy performs the complete
home-to-overhead-to-pregrasp approach; no approach progress is hidden in the reset.

Generate a replacement dataset only when physics, assets, or reset logic intentionally change:

```bash
uv run so101-vial generate_resets \
  --batch_size 128 --poses_per_phase 128 \
  --output src/so101_vial_place/assets/reset_poses.pt \
  --device cuda:0 --visualizer none
```

The tracked artifact is an ordinary Git blob of only about 95 KiB (not a Git LFS object). A full 1,024-state
regeneration took about three minutes on an RTX 6000 Ada in the reproducibility audit, and contact-validation
acceptance varied across otherwise identical Newton runs. Keep the tracked, hash-checked artifact for training
reproduction; generation is a maintainer workflow, not a fresh installation prerequisite.

Inspect the stored states independently of training:

```bash
uv run so101-vial view_resets --visualizer newton
```

## Physics, assets, and control

The environment loads the SO-101, vial, rack, and mat from tracked local USD assets. The visual rack retains its four
holes, while eleven box primitives reproduce its base, supports, and four-hole top lattice for collision. Newton does
not build an SDF or run Co-ACD on the rack. The vial uses two cylinder primitives matching its body and cap shoulder.

The robot has one `ImplicitActuatorCfg` so Isaac Lab can route joint targets, but every dynamics field is `None`.
Stiffness, damping, armature, friction, effort limits, and velocity limits therefore load directly from the Sys-ID
USD. A configuration guard rejects Python actuator overrides.

The policy produces six actions at 30 Hz:

- five measured-relative arm-joint position increments, limited to 0.033 rad per command;
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

The active state reward has six terms:

- one compact object-centric term: pre-grasp reach, then symmetry-aware vial center/endpoints error to the final
  held insertion pose while a live bilateral hold remains;
- one-time physical grasp, lift, and held-insertion milestone rewards;
- terminal physically confirmed success;
- actual vial-loss penalty;
- small action-rate and joint-velocity costs.

There is no staged curriculum, transport teacher, waypoint potential, collision-avoidance reward, or action-specific
opening/closing penalty.

## Train and evaluate the state policy

Train ordinary PPO for 400 updates on the hardcoded full-horizon reset distribution, retain the optimizer for another
200 full-horizon updates, then polish the complete canonical-home task for 100 updates. `--max_iterations` is the
number of additional updates when a checkpoint is loaded, so the stage finals are models 399, 598, and 697:

```bash
CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 400 --seed 42 \
  --run_name state_horizon_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 --max_iterations 200 --seed 42 \
  --checkpoint <stage-1-run>/model_399.pt \
  --run_name state_horizon_continue_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Canonical \
  --num_envs 4096 --max_iterations 100 --seed 42 \
  --checkpoint <stage-2-run>/model_598.pt \
  --run_name state_canonical_polish_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp
```

Benchmark the best per-job environment count on the target GPU; the primitive object colliders support the usual
2,048--4,096 state environments when memory permits. On a multi-GPU machine, use each GPU for an independent seed or
reward ablation; distributed PPO is not required.

Evaluate all 1,024 canonical-start episodes exactly once each. The callback
hardcodes the episode count, phase-zero reset filter, sequential sampling, and
policy action mode:

```bash
uv run so101-vial play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 --checkpoint <checkpoint.pt> --deterministic \
  --external_callback so101_vial_place.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp
```

The evaluator prints one `SO101_EVAL_RESULT` JSON object with grasp, lift, held insertion, success, timeout, loss,
unsafe contact, rack force, and reset-phase metrics. Acceptance must use complete episodes, not training-time
per-step occupancy metrics.

Record a frontal Newton rollout after a policy passes evaluation:

```bash
uv run so101-vial play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 1 \
  --seed 42 --checkpoint <checkpoint.pt> --deterministic \
  --video --video_length 600 \
  --external_callback so101_vial_place.evaluation.install_state_episode_counter \
  --visualizer newton presets=newton_mjwarp
```

## Wrist-camera policy

The camera task uses one physical wrist camera at 64x64 RGB and 30 Hz. It has measured OpenCV intrinsics and
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
uv run so101-vial capture_wrist \
  --output_dir checkpoints/screenshots/wrist \
  --visualizer none
```

Vision work starts only after a state policy passes the physical acceptance suite. The intended order is:

1. distill the state policy into wrist RGB + proprioception using student-visited states;
2. evaluate and refine that student with a privileged critic;
3. train the same vision actor from scratch as a separate comparison.

The full vision-from-scratch policy must not load or query the state teacher. A privileged critic is still allowed
because it is discarded at deployment.

### Distillation reproduction audit

The previously documented one-stage, 4,096-environment command is invalid. Fresh seeds 42 and 45 finished all 1,110
updates but reached only 5.27% and 43.65% exact canonical success. Reconstructing the historical 1,024-environment
stages produced 45.41% and 42.68%; a dense sweep of all 22 final refinement checkpoints found no result above 45.90%.
The retained students remain valid inference artifacts, but they are not evidence of reproducible training.

The recovered historical logs establish the actual five-stage experiment below. It uses teacher-controlled images
for the initial visual warm start, then student-visited DAgger. All configuration values are hardcoded in the named
agent configurations; `--max_iterations` counts additional updates after loading a checkpoint. This recipe reproduces
the failed audit above, not the retained >90% result:

```bash
CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_geometry_spatial_teacher_rollout_cfg_entry_point \
  --num_envs 1024 --max_iterations 600 --seed 42 \
  --checkpoint checkpoints/candidates/recovered_state_vial_farther_teacher.pt \
  --run_name distill_teacher_rollout_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_strong_geometry_spatial_distillation_cfg_entry_point \
  --num_envs 1024 --max_iterations 302 --seed 42 \
  --checkpoint <stage-1-run>/model_599.pt \
  --run_name distill_strong_to900_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_strong_geometry_spatial_dense_distillation_cfg_entry_point \
  --num_envs 1024 --max_iterations 201 --seed 42 \
  --checkpoint <stage-2-run>/model_900.pt \
  --run_name distill_strong_dense_to1100_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_low_rate_geometry_spatial_distillation_cfg_entry_point \
  --num_envs 1024 --max_iterations 6 --seed 42 \
  --checkpoint <stage-3-run>/model_1100.pt \
  --run_name distill_low_rate_to1105_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_ultra_low_rate_geometry_spatial_distillation_cfg_entry_point \
  --num_envs 1024 --max_iterations 5 --seed 42 \
  --checkpoint <stage-4-run>/model_1105.pt \
  --run_name distill_ultra_to1109_s42 --device cuda:0 \
  --visualizer none presets=newton_mjwarp
```

Evaluate the native distillation checkpoint without converting it to PPO:

```bash
CUDA_VISIBLE_DEVICES=0 uv run so101-vial play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera \
  --agent rsl_rl_ultra_low_rate_geometry_spatial_distillation_cfg_entry_point \
  --num_envs 1024 --checkpoint <distillation-run>/model_1109.pt \
  --deterministic \
  --external_callback so101_vial_place.evaluation.install_episode_counter \
  --device cuda:0 --visualizer none presets=newton_mjwarp
```

## Scene screenshots and diagnostics

```bash
uv run so101-vial capture_scene \
  --output_dir checkpoints/screenshots/scene \
  --visualizer newton

uv run so101-vial inspect_robot \
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
