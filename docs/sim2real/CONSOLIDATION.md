# Local and multi-GPU consolidation — 2026-10-08

Work branch: `feat/so101-consolidated-sim2real`.
Multi-GPU source: `origin/overhaul/so101-vial-newton` at `86167f0`.
Local source: `feat/so101-sim2real-multigpu` at `b4d8226`, plus the previously unstaged changes.
The complete local source and edits are preserved in commit `560ff59` on
`archive/so101-local-before-consolidation-20261008`. No old model or experiment log was deleted.

## Selected first deployment scope

The user selected **match the proven setup first, then expand the workspace**. The first policy uses
an initially empty four-hole rack at the reference pose. Any hole counts. Home starts sample the
128-row reset distribution (nominal XY half-ranges 30/40 mm), plus independent ±20 mm jitter with
rack-overlap rejection. This is not an arbitrary-workspace policy. The earlier broad rack/vial
sampling and visibility-latched target experiments remain available on the archive branch.

## Reconciliation decisions

| Difference | Consolidated decision and reason |
| --- | --- |
| Architecture and rewards | Use the multi-GPU branch's compact two-frame CNN, any-hole geometry, bounded exploration, DAgger and PPO handoff. Do not reintroduce failed local GRU/selected-hole experiments. |
| Camera | Retain physical mount randomization, overscan/projection, material and photometric DR, cache invalidation, and camera-housing self-occlusion fix. |
| Dependency | Retain the exact latest multi-GPU pin `a5edcec21741e98c5918f4c8ae88341fcd357a5f`. Historical >90% scores used an older runtime; the combined runtime needs fresh qualification. |
| Vial | Restore measured 28.9 mm body, 35.4 mm cap and three-primitive geometry, with matching visual scales. Update clearance bounds and cap grasp point. |
| Elbow | Restore +6.4° coordinate/frame correction. Shift reset positions, targets and generator waypoints together. |
| Gripper | Use this follower's 1449-tick/127.3846° calibrated span and the earlier provisional fitted zero. Physical joint-map verification remains required. |
| Self-contact | Restore local robot self-collision. Validate reset generation and fresh training with it enabled. |
| Action timing | Retain the proven 30 Hz policy / 120 Hz measured-relative target controller and matching deployment loop. The local held-target 30 Hz controller is incompatible and is not merged. Real serial timing remains to be measured. |
| Rolling/torsional contacts | Keep the multi-GPU contact model for initial qualification; local six-dimensional rolling resistance is an unqualified alternative. Current condim=3 does not use rolling/torsional coefficients. |
| Broader randomization | Preserve multi-GPU ranges and restore ±0.5 mm body/cap diameter DR at Newton construction, with matching render scale. Broad workspace, gravity variation and extra latency DR remain later controlled experiments. |
| Deployment dependencies | Keep LeRobot's isolated CPU environment; retain `uv sync --extra sim2real` as an alias for simulation-side LEAPP dependencies. This avoids incompatible training/LeRobot Torch requirements. |
| Documentation | Restore the complete local working log and findings, explicitly historical. Use this document for the new combined status. |

## Validation and fresh-training plan

1. Baseline on the fetched multi-GPU source: **100 tests passed**, deployment tests skipped because
   OpenCV is intentionally outside the training environment.
2. Reconcile geometry and regenerate/validate staged reset states under the combined physics model.
   Merely shifting old elbow coordinates does not revalidate old grasps after collider changes.
3. Fresh state actor/critic: bootstrap 800 updates; then randomized continuation in 200-update blocks.
4. Audit complete randomized home-start placement before using the teacher for a new visual student.
5. Fresh visual actor: 400 DAgger updates, then 200 PPO updates as the initial proven recipe.
   Extend only in response to measured evaluation results.
6. Qualify the frozen selected camera policy on separate clean/noisy seeds with 1,024 first episodes,
   retaining force, loss, timeout and duration statistics. Export only a qualifying model and verify
   LEAPP parity and CPU execution.
7. Verify physical joint mapping, camera projection and feedback timing before supervised robot trials.

The host has one RTX PRO 6000 Blackwell GPU with 96 GB memory. Pruning unused uv cache recovered
66.7 GiB and left roughly 75 GB available; model/dataset artifacts were retained. At inventory time
the Sonix wrist camera was present, but no follower/leader serial devices appeared under
`/dev/serial/by-id`. No motor writes have been performed during consolidation.

## Current status

Consolidation and geometry validation are in progress. No newly trained policy has been qualified.
The reset-generator CLI had not resolved `PresetCfg` after the dependency update; it now accepts
`--presets` and resolves them before launch. The README generator commands were corrected.
Generated checks and future training artifacts live in `outputs/consolidation_20261008/` and `logs/`.

### Validation update

The combined model passed all eight phases of a 64-row reset-generation probe, including physical
insertion and release (dataset content SHA-256 `0a98389100be625890f7158d68f5cd9008d98bcb6f5722bb672c0011629e9fdc`).
A fresh 64-environment state PPO smoke test completed two updates. The software suite passed
100 tests after the geometry changes. Reset generation additionally required the current
`isaaclab_newton.controllers.ik` import and clone-plan path lookup API; both are migrated.
The full 1,024-row reset dataset and randomized camera smoke test are in progress.

The randomized camera PPO smoke completed two updates; inspected 16 rendered views contain the rack,
vial and jaws under mount/appearance variation. Native Newton construction checks confirmed independent
per-world diameter variation in the requested ranges, matching body/bottom radii. The current suite
remains at 100 passing tests. Runtime package versions are saved in the local artifact directory.

The isolated CPU deployment environment resolves successfully; its three preprocessing/control tests
passed. These tests made no motor connections. Diameter probing over 16 worlds measured body diameters
28.431–29.374 mm and caps 34.917–35.869 mm, inside the requested ±0.5 mm intervals.

### Full reset validation and training launch

Regenerated and installed all 1,024 reset rows (128 per phase) on the corrected model. Content SHA-256:
`916e90a973cb7def1dbee83240006badef6f700b5c91315e127a4da6425149dc`.
The regression suite passes 100 tests in 4.09 s; all three deployment tests pass separately.
Ruff, formatting, spelling and locked dependency checks pass. No old trained weights are used for
the new bootstrap. The first training command is:

```bash
TMPDIR="$PWD/.cache/tmp" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1 \
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --max_iterations 800 \
  --seed 42 --run_name consolidated_fresh_state --visualizer none presets=newton_mjwarp
```

### Solver and hardware checks

The new MuJoCo stack exhausted the inherited 15-iteration line-search budget frequently. A matched
64-environment/two-update probe on the new reset dataset produced 2,577 warnings at 15 and zero at 50.
Raised the budget to 50; this follows the current upstream default. Stopped the initial bootstrap at
iteration 53/800 and will restart from random weights. Replayed all 1,024 reset states for 0.5 seconds
under the new solver budget: all remained finite; all 640 grasp-through-insertion states retained
bilateral contact. The 95th-percentile drift was ≤5.01 mm in those held phases.

The user connected the robot and confirmed unchanged calibration and camera mount. Persistent IDs
identify follower `5AE6079843` (currently `/dev/ttyACM1`) and leader `5AB0179854` (currently `/dev/ttyACM0`).
The running shell lacked the already-granted dialout group; `sg dialout` supplies that group without
changing device permissions. Read-only bus checks confirm matching stored calibration. Joint reads
took 1.40/1.69/1.78 ms at the 50th/95th/99th percentiles; this excludes position writes and inference.
A focused 640×480 image was saved as `outputs/consolidation_20261008/real_camera_current.png`.

The current elbow read 102.59° exceeds the calibration interval's 92.48° positive half-span. Matching
calibration registers therefore does not establish a verified map or an execution-ready starting pose.
Added an execution-only check that rejects a measured pose outside the intersected limits before
torque configuration and on each feedback cycle. Read-only checks remain available. No motor writes
or torque changes were made. Physical zero/sign and home-pose checks remain pending.

### Active fresh campaign and first development audit

The solver-50 bootstrap restarted from random weights in
`logs/rsl_rl/so101_vial_state/2026-10-08_14-36-10_consolidated_fresh_state_solver50/`.
Its planned final checkpoint is `model_799.pt`. The local supervisor
`outputs/consolidation_20261008/continue_training.py` waits for this process, then runs the
randomized teacher, fresh visual distillation, visual PPO, independent audits and LEAPP export.
Exact commands, stage status and results are recorded in `campaign.json` beside the supervisor.
It stops on failed commands or acceptance gates; starting the campaign is not evidence of success.

An early development audit of bootstrap checkpoint 200 on the randomized task scored **0/128**
complete home-start placements: 83 lost vials and 45 timeouts, with no successful grasps or lifts.
Peak rack contact was 6.60 N. This checkpoint has not yet received the planned randomized continuation.
Mixed-stage training metrics must not be interpreted as complete home-start success. The teacher
must pass a 90% development gate and an independent 1,024-episode qualification before distillation.
The selected frozen vision actor must pass four separate 1,024-episode clean/noisy audits at 90%
before export. Export parity and physical mapping/timing checks remain separate requirements.

### Calibration range investigation

The workshop's ten-follower calibration reference has a mean elbow travel of 2,221.7 encoder ticks;
this follower's saved travel is 2,104 ticks (about 10.35° narrower). Its shoulder-pan range is also
narrower than that reference population. Together with the live elbow reading outside its saved
interval, this warrants checking whether the original manual sweep reached both mechanical stops.
Population statistics do not establish the correct limits for this particular arm. We have not
replaced calibration values with reference averages or changed motor settings.

The raw comparison is saved in `outputs/consolidation_20261008/calibration_reference_comparison.json`.
The user was asked whether the elbow calibration sweep covered its full mechanical travel. Any
recalibration must preserve the original file and be followed by a new measured joint-map check;
calibration matching alone does not verify physical joint zeros.

The user confirmed that an endpoint may have been missed. The original calibration is backed up at
`outputs/consolidation_20261008/calibration_before_resweep.json`. The isolated calibration entry point
was checked with `--help` successfully; no calibration was executed by the agent. From the repo root:

```bash
sg dialout -c 'uv run --script src/isaaclab_tutorial/utils/calibrate.py --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00 --robot.id=wowrobo_follower'
```

Support the arm before running this: calibration disables torque. Type `c` at the existing-file prompt
to perform a new calibration. At the midpoint prompt, match the workshop's
`../sim-to-real-so-101-workshop/docs/images/calibration_pose.jpg` and `wrist_center.jpg` references.
Then follow LeRobot's instructions: sweep each joint **except wrist_roll** through its full mechanical
travel, one at a time. Ensure cables or nearby objects do not create false endpoints; do not force a
joint past its stop. Include the gripper's full range. Press Enter after all ranges are recorded.
Only the follower needs this check now. Afterward re-read calibration and physical pose, compare the
old/new spans and centers, and revalidate the deployment mapping before any policy execution.

### Completed follower resweep

The user completed recalibration. A subsequent read-only check confirms the saved calibration matches
the motor registers. Elbow travel increased from 2,104 to **2,211 ticks** (184.97° to **194.37°**),
recovering 9.41° of previously unrecorded travel. Its current reading is 23.87°, within the new ±97.19°
calibrated interval. Joint read latency is 1.41/1.67/1.72 ms at p50/p95/p99. No motion commands were
sent during this verification.

| Joint | Previous span, ticks | New span, ticks |
| --- | ---: | ---: |
| Shoulder pan | 2664 | 2635 |
| Shoulder lift | 2344 | 2347 |
| Elbow | 2104 | 2211 |
| Wrist flex | 2294 | 2288 |
| Wrist roll | 4095 | 4095 |
| Gripper | 1449 | 1442 |

Saved the new calibration, comparison, bus report and camera image as `calibration_after_resweep.json`,
`calibration_resweep_comparison.json`, `hardware_after_resweep.json` and `real_camera_after_resweep.png`
in the campaign artifact directory. The shoulder-pan span remains narrower than the workshop reference;
the new sweep does not establish physical zero/sign correctness for any joint.

Prepared `joint_map_after_resweep.unverified.json` with gripper scale based on **1,442 ticks / 126.77°**.
It deliberately retains `verified: false` and the provisional gripper zero. The active training process
retains its original 1,449-tick simulated gripper span; changing source physics during a campaign would
mix training/evaluation models. The 0.615° upper-travel discrepancy must be included in deployment
qualification. Physical reference-pose checks, gripper zero verification and inference/control timing
remain outstanding. A fully closed resting gripper can also be outside the simulation's soft limits;
do not treat a valid calibration as authorization to bypass the initial-pose guard.

### First full campaign result

The fresh bootstrap completed 800 updates. Three 200-update randomized continuation blocks reached
84.77%, 89.45% and 90.23% on 256 development episodes. The selected state teacher then passed its
independent 1,024-episode qualification: **932/1,024 (91.02%)** complete home-start placements.

Fresh visual distillation completed 400 updates, followed by 500 total visual PPO updates. Development
results, each over 256 first episodes per observation condition:

| Visual PPO updates | Clean observations | Noisy observations |
| ---: | ---: | ---: |
| 200 | 82.42% | 84.38% |
| 300 | 85.16% | 86.72% |
| 400 | 85.55% | 89.45% |
| 500 | 85.16% | 89.45% |

The supervisor stopped at its explicit visual acceptance gate. **No new qualifying visual export was
produced and no real policy rollout was performed.** The final clean audit had 38 failures: 22 never
registered a grasp, seven registered grasp but not lift, eight lift but not insertion, and one insertion
without final success. The noisy audit had 27 failures, 17 without a registered grasp. These diagnostic
flags suggest pickup is the main remaining problem, but do not establish its physical or visual cause.
They are not a strict success-stage hierarchy: placement can succeed without every diagnostic flag.

A fresh-seed 1,024-episode clean diagnostic of the final checkpoint was launched to check the apparent
plateau beyond the reused 256-episode development sample. Results go to
`outputs/consolidation_20261008/vision_diagnostic_clean_1024.json`. This is a diagnostic of an unqualified
model, not a replacement for the planned frozen-model clean/noisy qualification suite.

That diagnostic completed at **879/1,024 (85.84%)**, confirming the visual plateau on a fresh seed.
There were 78 lost vials and 68 timeouts (termination flags can overlap), and one episode exceeded
the 20 N rack-contact diagnostic, peaking at **20.16 N**. Mean successful duration was 14.66 seconds.
The model remains below the 90% acceptance threshold; the next investigation should target pickup
failures rather than describe this checkpoint as ready for real deployment.

### User-requested supervised trial preparation

The user chose to try the current visual checkpoint before it reaches the original acceptance gate.
Prepared a separate `supervised_trial/` bundle, explicitly marked experimental, without relabeling
the failed qualification. Instructions are in [the SO-101 working guide, section 11](../SO101_SIM2REAL.md#11-current-supervised-real-trial--2026-10-08).

Export parity passed with zero error on eight inputs. LEAPP's initial CUDA/CPU validation mismatch
was resolved by hiding CUDA during CPU export. Independent Torch 2.10 CPU runtime parity versus the
Torch 2.13 training actor passed at maximum absolute error 5.37e-7. Real-camera/read-only inference
completed 2,400 feedback steps over 20 seconds with zero missed 120 Hz deadlines; inference p95 was
2.27 ms and overall control work p95 was 3.62 ms. Motor writes and tracking remain untested.

Added a read-only joint inspector, local trial launcher, calibration-file fingerprint checking and
an optional start-pose guard. The prepared launcher enforces the actual training home within 0.035
radians. Rejected startup checks do not change torque. Updated isolated deployment tests pass five
tests. Reference renders and measured startup errors were saved beside the bundle; the current
resting arm is not at home. Physical joint-zero/gripper alignment and camera-view comparison remain
required before setting the map verified. No motor motion or torque changes were performed.

### Guide layout restored

The user requested continuing the existing guide and using direct uv commands. Checked both sibling
repositories: `../mustafa_isaaclab2/SO101_SIM2REAL.md` is already a redirect to this repository's
`docs/SO101_SIM2REAL.md`; no second guide was found in `mustafa_isaaclab3`. Integrated the complete
trial procedure into section 11 of that existing working log and refreshed its current-status summary.
The standalone trial document is now only a redirect. Removed the newly introduced shell launcher;
current instructions use `uv run --script` directly, with `newgrp dialout` only if the user's terminal
has not picked up its existing group membership. Preserved historical commands as dated records and
explicitly distinguished the current isolated hardware environment from the old shared environment.
