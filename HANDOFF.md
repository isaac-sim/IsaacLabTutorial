# SO-101 vial placement: implementation and parallel-GPU handoff

Date: 2026-08-22  
Repository: `/home/mhaiderbhai/code/IsaacLabTutorial`  
Task ID: `IsaacTutorial-Place-Vial-SO101`

## 2026-08-24 authoritative continuation

This section supersedes the older status narrative below. The older material is retained only as historical context.
The final pause-point evidence, retained artifacts, rejected last batch, exact commands, and restart plan are in
`VISION_SCRATCH_HANDOFF.md`; that file supersedes the active-job details in this document.

### Accepted results

- State is solved from the farther canonical home start. Fresh corrected three-stage reproductions reached 97.17%,
  99.61%, and 99.41% exact canonical success in seeds 42, 43, and 45. The actual historical 100% teacher used by
  distillation is
  `checkpoints/candidates/recovered_state_vial_farther_teacher.pt`; it was recovered losslessly from the identical
  teacher weights embedded in both native distillation checkpoints. Its original optimizer was not recovered. The
  previously retained model-399 checkpoint is a pre-farther-layout diagnostic and reaches only 74.22% now.
- Retained spatial-softmax state-to-vision distillation inference is solved with wrist RGB plus proprioception:
  checkpoints labeled seeds 42 and 45 reach 92.87% and 90.82% exact canonical success. They are five-update branches
  from one parent, not independent full training runs. Fresh one-stage training reached 5.27%/43.65%; reconstructed
  staged training reached 45.41%/42.68%, so distillation training is not currently reproducible. State and retained
  distillation rollout-video candidates exist under `checkpoints/videos`.
- Full canonical vision-from-scratch is not solved. Do not promote a scratch checkpoint until it passes the fixed
  1,024-episode exact evaluator and independent seeds.

### Vision-from-scratch evidence

- Resolution is fixed at 64x64. The actor consumes the fabricated wrist-camera view plus 24 proprioceptive values;
  the privileged state is critic-only.
- Canonical scratch acquisition/lift specialist:
  `checkpoints/candidates/vision_transport_alignment_m1850_std003.pt`, approximately 98% grasp and 97% lift.
- Phase-five insertion/release specialist:
  `checkpoints/candidates/vision_scratch_release_direct_seed413_model260.pt`, 90.3% exact phase-five success and
  99.9% held insertion.
- Joint BC, geometry-augmented BC, one DAgger relabel pass, full-network PPO, split-gripper PPO, and post-lift
  residual PPO have not connected those specialists. The best residuals preserve grasp/lift but trade upright
  alignment for modest radial progress. Exact success remains 0%.
- A post-lift residual boundary using measured jaw/shoulder proprioception preserves acquisition, but exact audits
  showed the correction still drifts over long rollouts. Do not claim mixed-reset training metrics as canonical
  evidence.
- Uniform scratch PPO checkpoint seed 701 at iterations 100 and 300 had 0% grasp and 100% vial loss on 1,024 exact
  canonical starts, despite strong downstream reset metrics. A 90%-canonical run at iteration 100 also had 0%
  canonical grasp. This proves reset-phase aggregate metrics can be badly misleading.
- Longer uniform controls remained canonical-zero: seed 701 at iteration 700 had 0% grasp and 99.8% loss, and stable
  seed 704 at iteration 800 had 0% grasp and 99.9% loss, each over 1,024 exact canonical episodes.
- The historical scratch curriculum is physically valid. Exact canonical audits measured 60.8% grasp at the
  canonical-close model 500, 100% grasp at the acquisition model 700, and 96.8% grasp/lift at the lift-progress
  model 1699. The unresolved boundary is therefore specifically post-lift transport.
- Raising arm exploration to 0.10 destroyed the lift skill within 25 PPO updates. Keeping exploration at 0.03
  preserved 97.0% grasp and 96.0% lift after 25 balanced-horizon updates, but improved upright alignment at the cost
  of worse radial distance. Low-noise transport consolidation is the active direction.
- Exact exploration/rate ablations selected `0.03` action noise and a fixed `1e-4` PPO rate. Arm noise `0.05`
  worsened radial distance and introduced vial loss; rates `3e-5` and `1e-5` preserved lift but made no transport
  progress. Canonical dense `1e-4` reduced exact radial distance to about 18.0 cm in two PPO seeds while retaining
  at least 95.5% grasp/lift. This is real but still far from insertion and is not a solved scratch policy.

### Active single-GPU jobs

There are no active jobs at the pause point. The final four-run fixed-mixture batch completed and was rejected; see
`VISION_SCRATCH_HANDOFF.md` for its exact canonical and bridge results.

The proven historical scratch acquisition chain began with
`2026-08-23_14-42-46_vision_tuned_stable_canonical_close_s114` from random initialization. Its grasp signal emerged
around iterations 400--500, followed by `Discovery-CloseLift` and `Discovery-CanonicalLiftProgress`. Exact-audit the
new canonical-close runs near that window before continuing the same compact stage sequence.

### Important fixes in the current worktree

- Transport shaping now uses irreversible physical grasp/lift history instead of instantaneous bilateral contact;
  success and termination remain contact validated.
- Release progress/action credit is physically gated by irreversible held-insertion history.
- Non-finite terminal physics values are neutralized in reward terms so 4,096-environment runs do not fail with NaN
  rewards. Invalid geometric errors map to a safely distant value rather than an accidental perfect score.
- Added hardcoded slow/midrate residual PPO configs and fixed reset-distribution ablations; no reward values depend on
  shell environment variables.
- Added one plain spatial-softmax fine-tuning configuration with 64-step rollouts, fixed `1e-4` learning rate, and
  `gamma=0.995` for the longer post-lift credit horizon. It uses ordinary PPO and adds no policy logic.
- Added matching `3e-5` and `1e-5` rate ablations; exact audits rejected both for lack of radial progress. Added one
  strong-dense task that uniformly scales the same three post-lift geometry rewards by 5x; its first audit is pending.
- Added full-horizon canonical rollout recording and residual consolidation/relabel tooling. These are diagnostic,
  not accepted policy machinery.
- Newton allocator startup faults (`exit 134/139`, `malloc(): unaligned tcache chunk`) are transient; retry the exact
  command. After interrupting a run, check `nvidia-smi` because a wrapper interruption once left a Python child
  orphaned on GPU 3.
- `uv run ruff check .` passes and the full suite has 85 passing tests.

### Current acceptance workflow

Use the fixed evaluator, with no environment variables:

```bash
CUDA_VISIBLE_DEVICES=<gpu> uv run so101-vial play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Discovery-HorizonReleaseDirect \
  --agent <matching-agent-entry-point> --num_envs 1024 --checkpoint <checkpoint.pt> \
  --deterministic --external_callback so101_vial_place.evaluation.install_episode_counter \
  --device cuda:0 --visualizer none presets=newton_mjwarp
```

The final scratch acceptance remains at least 90% canonical success over at least 1,024 episodes and multiple seeds,
followed by camera/rollout videos and a clean from-zero reproduction audit.

## Executive status

The environment, physical assets, canonical robot start, wrist camera, reset generator, reset viewer, exact evaluator,
and Newton configuration are working. The current training code has been simplified to the intended tutorial
foundation:

- five bounded relative arm-joint commands plus one bounded relative jaw command;
- actuator dynamics loaded unchanged from the SO-101 Sys-ID USD;
- a uniform distribution over eight physics-validated reset phases;
- a six-term reward containing reach, one-time physical milestones, terminal success/loss, and two tiny smoothness
  costs;
- standard PPO from scratch with an asymmetric critic;
- no transport teacher, scripted orientation, object-state writes during episodes, artificial attachment, collision
  disabling, staged difficulty filters, or checkpoint-preservation curriculum.

This foundation is promising but the state task is **not solved from the canonical start yet**. A 4.096-million-step
sparse baseline learned phases 5--7 extremely well but did not propagate to phases 0--3. This is useful evidence:
joint-relative control is sufficient for physical transport, insertion, and release; the next experiment should add
one compact object-centric shaping term for approach/lift/reorientation.

Do not present any existing policy as the final 90% result.

## Current exact results

Checkpoint:

`logs/rsl_rl/so101_vial_state/2026-08-22_15-23-23_01_sparse_joint_horizon/model_499.pt`

Training configuration:

- 128 environments, 64 steps/environment/update;
- 500 iterations, 4,096,000 transitions;
- 11 minutes 27 seconds, approximately 6,000 simulator steps/s;
- uniform sampling over all eight reset phases;
- deterministic checkpoint audit after training.

The standalone canonical-start audit used all 128 phase-zero rows once:

| Metric | Result |
|---|---:|
| Episodes | 128 |
| Grasp | 0.0% |
| Lift | 0.0% |
| Success | 0.0% |
| Timeout | 93.75% |
| Vial lost | 6.25% |
| Unsafe rack impact | 0.0% |

The 1,024-episode sequential horizon audit reported:

| Reset phase | Episodes counted | Grasp | Lift | Held insertion | Success | Timeout | Lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 approach | 133 | 0.0% | 0.0% | 0.0% | 0.0% | 92.48% | 7.52% |
| 1 pregrasp/closure | 128 | 85.16% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 2 grasp | 128 | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 3 lift | 128 | 100.0% | 31.25% | 0.0% | 0.0% | 100.0% | 0.0% |
| 4 reorient | 128 | 100.0% | 100.0% | 28.13% | 27.34% | 72.66% | 0.0% |
| 5 transport | 128 | 100.0% | 100.0% | 99.22% | 99.22% | 0.78% | 0.0% |
| 6 insert | 123 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| 7 release | 128 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| Aggregate | 1,024 | 85.16% | 53.42% | 40.43% | 40.33% | 58.69% | 0.98% |

The evaluator stops when the aggregate target is reached, while environments finish asynchronously. That produced
133 phase-zero and 123 phase-six episodes rather than exactly 128 of each. The separate phase-zero audit is exact;
the other phase rates are strong diagnostics but should be rerun with a phase-filtered evaluator before final
acceptance.

Peak rack force over the full audit was 10.27 N, below the 20 N hard-impact diagnostic threshold. There were no
unsafe rack impacts. Ordinary rack guidance contact is allowed.

## What is implemented

### Repository and task organization

- Python package: `src/so101_vial_place`
- Public state task: `IsaacTutorial-Place-Vial-SO101`
- Public camera task: `IsaacTutorial-Place-Vial-SO101-Camera`
- Project distribution: `so101-vial-place`
- The superseded `so101_vial_lift` package has been removed.

### Robot and control

- Robot asset: `src/so101_vial_place/assets/so101/so101_new_calib.usda`
- The environment asserts that every actuator field in `ImplicitActuatorCfg` is `None`; stiffness, damping,
  armature, friction, effort limits, and velocity limits therefore come directly from the authored Sys-ID USD.
- Canonical start is the workshop controller's real operational initial pose:
  `(-0.122107, -0.906685, 0.190088, 1.479793, -0.804401)` plus the ordinary open jaw target.
- Policy actions are now five relative joint-position increments at 0.03 rad/full-scale and one relative jaw
  increment at 0.02 rad/full-scale, at 30 Hz.
- This removes a custom 6-DoF Cartesian IK command from a five-DoF arm and matches an interface that can be
  reproduced directly on hardware.

### Vial, rack, collision, and seating

- Visual assets are the SO-101 workshop vial and four-hole rack, not proxy cylinders/blocks.
- Newton preserves the detailed rack mesh, repairs it to a watertight mesh, and builds a narrow-band SDF in about
  0.03 seconds. Co-ACD is not used for the rack.
- A physical centered free-drop test established the released root target near rack-local `z=0.031 m`.
- Held insertion is distinct from final seating: the jaws stop near root `z=0.060 m`, open, and gravity seats the
  vial. This prevents both shallow success and gripper/rack interpenetration.
- Success requires a released, stable, upright vial inside the deep seated bounds. Object pose and velocity are
  never written after reset.

### Physics

- Newton MJWarp, 120 Hz physics, control decimation 4 (30 Hz policy), two solver substeps.
- Solver settings follow the Isaac Lab Franka/Kuka manipulation family: Newton/implicit-fast, 100 Newton iterations,
  15 line-search iterations, elliptic cone, `impratio=10`, and `update_data_interval=2`.
- Detailed-rack broad-phase capacity is 2.5 million triangle pairs.
- The runtime prints a benign fixed-articulation root-COM velocity warning.

### Reset generator and dataset

- Generator logic is isolated under `src/so101_vial_place/reset/`.
- Resets are reached by connected dynamic rollouts through the untouched drives and contacts. Candidate states are
  rejected for failed grasp, instability, invalid IK, unsafe contact, poor seating, or loss.
- Dataset: `src/so101_vial_place/assets/reset_poses.pt`
- Schema version 2, 1,024 rows, 128 rows for each of:
  `approach`, `pregrasp`, `grasp`, `lift`, `reorient`, `transport`, `insert`, `release`.
- Dataset content hash: `3704912036a4ac3de628ba1c71c3545a0e579534c8b1ec8f5e1cc56d62b3a282`
- File SHA-256: `8a6c1fbdce85210ba81218e7689fdc5b8ce557e6e1ef742d178b784a47d7ab9b`
- Metadata explicitly records gravity enabled and `object_state_writes_after_reset=False`.
- The reset viewer cycles stored poses in the Newton visualizer independently of training:

```bash
uv run isaaclab view_resets --visualizer newton
```

### Current MDP

Actor state includes joint state/target, previous action, end-effector state, vial state, rack-relative target,
placement features, and three physical progress flags. The critic receives the same state plus contact observations.

Active reward terms:

| Term | Weight | Meaning |
|---|---:|---|
| reaching | 0.1 | jaw midpoint approaches vial grasp point before grasp |
| milestones | 10.0 | one-time grasp, lift, held-insertion pulses in ratio 1:2:4 |
| success | 200.0 | physically confirmed released seating |
| vial_lost | -50.0 | actual drop/non-finite/unreachable object |
| action_rate | -0.002 | small command smoothness cost |
| joint_velocity | -0.0002 | small motion regularizer |

The active reset choices are intentionally only `horizon` for training and `initial` for canonical evaluation. There
are no staged phase weights or within-phase difficulty filters.

PPO is conventional rather than specialized: 64-step rollouts, 256/256/128 ELU actor and critic, observation
normalization, initial action standard deviation 0.2, learning rate `3e-4` with adaptive schedule, entropy `0.005`,
gamma `0.995`, lambda `0.95`, five epochs and eight minibatches.

### Wrist camera and vision boundary

- The camera is rigidly attached to the wrist/gripper, not a world overview camera.
- RGB is 64x48 at 30 Hz with measured OpenCV intrinsics/distortion and a buildable 55 mm side bracket pose.
- The deployed vision actor is limited to wrist RGB, joint position/velocity/target, and previous action.
- The vision critic is already asymmetric and may use full privileged state/contact during training.
- Episode-consistent exposure, contrast, white balance, brightness, pixel noise, and proprioceptive noise are
  implemented.
- Camera previews were visually checked and found viable. Vision training should wait for a strong state policy.

### Evaluation and presentation

- Exact episode callback reports grasp, lift, held insertion, success, loss, timeout, unsafe contact, contact force,
  and per-reset-phase breakdown.
- The default Newton rollout camera is frontal, raised, and angled down at the task.
- Scene and wrist screenshot tools exist via `capture_scene` and `capture_wrist`.

## Why training used 128 environments

This was a conservative per-GPU engineering choice, not a belief that PPO prefers 128 environments. The detailed
four-hole rack is expensive per replicated world. At 256 environments, a prior insertion run measured about 1.57
million triangle-pair candidates. Naive scaling suggests roughly 12.5 million at 2,048 environments and 25 million
at 4,096, before solver/contact state. Earlier work also encountered host-side collision initialization and memory
pressure.

At 128 environments and 64 rollout steps, each update already contains 8,192 transitions and runs near 6,000
simulator steps/s. On the new machine, benchmark 128/256/512/1,024 environments **per independent job** and select
the count with best stable transitions/second. Do not use distributed PPO. Use each GPU for a separate job.

## Recommended parallel-GPU plan

### 1. Reproduce and benchmark

On one GPU, install from the lockfile, run the test suite, launch a five-iteration smoke train, inspect the reset
viewer, and reproduce the phase-5/6/7 behavior of `model_499.pt`. On otherwise idle GPUs, benchmark environment
counts of 128, 256, 512, and 1,024 for 20 iterations. Record startup time, peak GPU memory, steps/s, and any dropped
triangle-pair/contact warnings. Pick one per-job batch size; do not distribute one run across GPUs.

### 2. Add one object-centric task reward

Keep the current milestone and terminal terms. Add one symmetry-aware geometric reward modeled on ADEPT's simple
object-to-goal objective:

- before grasp: exponential jaw-midpoint to vial-grasp-point distance;
- while physically held: exponential vial-to-final-held-goal error;
- represent vial pose with center plus its axial endpoints and minimize over the two endpoint assignments, so vial
  yaw and unsigned-axis symmetry are treated correctly;
- use the final held insertion pose directly, not a hand-authored transport waypoint;
- gate the goal term by live physical grasp/contact;
- do not penalize ordinary rack contact.

The primary ablation should remain very small. Suggested independent jobs:

| GPU/job | Experiment |
|---|---|
| A | Current sparse baseline, new seed, to measure variance |
| B | ADEPT-style reach + symmetry-aware held object-to-goal reward |
| C | Job B plus one bounded held-vial clearance/height term |
| D | Job B with a second seed |

The clearance term in C is a diagnostic, not an assumed requirement. If B solves phases 0--7, omit it. Avoid
phase-specific rewards, reward switches based on reset phase, action-command penalties, waypoint potentials, and
special training callbacks.

Run 1,500--2,500 iterations initially. Audit saved checkpoints periodically with deterministic complete episodes;
rank by canonical phase-zero success, then full-horizon minimum phase success, loss rate, impact force, and video
quality—not shaped return.

### 3. Confirm robustness

For the best formulation, run at least three independent seeds in parallel. Acceptance for the state policy:

- at least 90% canonical-start success over at least 1,000 episodes and several seeds;
- strong success in every reset phase, not only aggregate success;
- no object writes, constraints, attachments, actuator overrides, collision disabling, or visual/physics mismatch;
- inspect multiple frontal videos for smooth approach, credible bilateral grasp, lift, upright transport, controlled
  rack guidance, full gravity seating, and retreat;
- report peak/mean rack forces and loss/timeouts.

Only then promote a checkpoint, export it, update the manifest/results, and record final videos.

### 4. Vision sequence after state acceptance

1. Distill the accepted state actor into the wrist-RGB + proprioception student using student-state rollouts.
2. Keep the privileged asymmetric critic; it is training-only.
3. Add symmetry-aware vial-axis/center and selected-hole auxiliary prediction if action cloning alone has poor visual
   geometry.
4. Evaluate distillation from every reset phase and canonical starts.
5. For full vision-from-scratch, reuse the same horizon reset distribution and compact reward. Launch independent
   seeds on separate GPUs. A vision-only actor and critic is an ablation; the preferred full-RL setup may still use
   a privileged critic while keeping the deployed actor strictly wrist RGB + proprioception.

## Known issues and cleanup status

- The working tree is not committed. Commit or archive it before moving machines.
- The `model_499.pt` sparse checkpoint is a diagnostic candidate, not an accepted policy.
- Incompatible seven-action checkpoints, exports, evaluation logs, videos, and rejected experiment logs were moved
  to the desktop trash during cleanup. They are recoverable there if historical debugging is ever necessary.
- `checkpoints/manifest.json` now has no accepted checkpoint and lists only the current six-action diagnostic model.
  `README.md` documents the simplified workflow and deliberately makes no success claim.
- Dead transport/acquisition teacher files, behavior-cloning helper, custom Cartesian IK implementation, complex
  potential reward, command-specific penalties, and narrow staged curriculum entries have been removed from source.
- Python `__pycache__` may still contain bytecode named after removed modules; it is ignored and has no runtime role.
- Isaac Lab emits deprecation warnings for generic rigid-body/articulation schema configs. They are upstream API
  migration warnings, not observed task failures.
- Newton emits a benign fixed-base root-COM velocity warning and can emit a sensor destructor warning at interpreter
  shutdown. Neither changed rollout behavior.
- A possible mismatch between Isaac Lab's manager reset order and one Newton regression-test reset pattern was
  considered but not proven to affect this task. Do not claim it as an Isaac Lab bug without a minimal reproducer.

## Fresh-context continuation notes

1. Trust exact episodic evaluation, not per-step `Metrics/success_rate` or aggregate termination metrics. The latter
   made several weak policies look solved.
2. Held insertion and released seating are different physical states. The robot should stop near local `z=0.060`,
   open, and let the vial settle near `z=0.031`.
3. The sparse joint policy proves that phases 5--7 are easy with the clean interface: 99.2%, 100%, 100%. Preserve
   this simplicity.
4. The missing gradient is specifically canonical approach, lift, and reorientation. Add a compact object-goal
   reward there; do not build another transport teacher or phase ladder.
5. A previous dense potential paid for waiting/holding and created timeout optima. A later regression-aware version
   still required multiple gates and difficulty filters. Those approaches were removed.
6. A bounded Cartesian teacher also failed because compliant object lag and underactuated IK caused oscillatory
   corrections. This reinforced switching to direct joint increments.
7. Old Cartesian checkpoints have seven actions and cannot initialize the new six-action actor.
8. Reset phases must remain connected physical states. Generator waypoints are allowed only to create reset data;
   the learned policy receives no waypoint or phase script during episodes.
9. The rack's visual mesh and collision SDF both retain four holes. Never replace it with a block/proxy collider for
   training convenience.
10. Do not override actuator gains. The real Sys-ID USD is the sim-to-real contract.
11. Light rack contact is expected during insertion. Track forces and hard impacts, but do not prohibit all contact.
12. The canonical start is intentionally farther from the vial and is the only final acceptance distribution.
13. Use the extra GPUs for parallel ablations/seeds, one ordinary PPO job per GPU. This keeps comparisons simple and
    avoids distributed optimizer changes.
14. ADEPT's most relevant lessons are: simple object-centric rewards can beat elaborate shaping; rank by task
    success; use asymmetric critics; preserve physical contact; and for vision distillation train on student-visited
    states with an auxiliary object-geometry loss.

## Historical command warning

The 2026-08-22 shell-environment recipes that originally followed this section are obsolete. The current task does
not read `SO101_*` configuration environment variables. Use the hardcoded, single-GPU training and exact-evaluation
commands in `README.md`; they are the commands exercised by the current reproducibility audit.
