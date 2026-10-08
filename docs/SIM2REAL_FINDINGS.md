# SO-101 sim2real findings for comparison

> Historical local-branch record. The current consolidation status and commands are tracked in
> [sim2real/CONSOLIDATION.md](sim2real/CONSOLIDATION.md) and the repository README.

Recorded 2026-10-07. These are findings from this attempt, not an assessment of another agent's solution.
**I did not obtain a qualified high-success full-task wrist-camera policy.**

## Task and evaluation contract

The current local `preset=sim2real` accepts placement into **any of the four initially empty holes**.
A separate teacher/reward goal chooses an in-frame opening near the image centre and keeps that physical
opening until reset. That choice uses projected simulator geometry, not an RGB hole detector; projection
alone does not prove a hole is unoccluded. The camera actor must use RGB and proprioception only.

Rack yaw remains ±15°. Placement sampling uses a 0.16–0.38 m radius and ±100° sector, collision rejection,
and a camera-visible initial vial filter. Vial heading remains unrestricted. The wrist observation is
64×48 RGB, with 30 Hz control and four physics steps per command. Episodes start at home and last up to
40 seconds. Success requires physical insertion, upright seating, release and ten stable control steps;
grasping, lifting or starting near the release pose is not whole-task success.

## Measured findings

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Same state checkpoint 499 and seed 68, require all four hole centres in frame | Target acquired 17/128; placed 5/128 | Strict visibility gate blocked many attempts |
| Same checkpoint and seed, permit a partially framed rack | Target acquired 40/128; placed 16/128 | Controlled evidence that relaxing the gate helped; still low success |
| State teacher 2999, current any-hole goal, seed 73 | Grasp 121/128; lift 116/128; insert 103/128; place 90/128 (70.3%) | Privileged teacher, not a vision-policy score |
| Camera PPO 499, broad home starts, seed 57 | Grasp 120/128; lift 119/128; place 0/128 | Good pickup did not translate into placement |
| Larger demonstration-corpus camera fit 40000, seed 55 | Place 2/128 | Low training action error did not predict successful rollouts |
| Camera release-stage clone 100, release-only starts | Place 123/128 (96.1%) | Only the release stage; same policy failed pickup from home |

The 90/128 run used a different checkpoint and seed from the visibility ablation, so its improvement
cannot be attributed solely to accepting any hole. It also had one episode exceeding 20 N rack contact
(peak 28.94 N). A subsequent local fix delays goal latching until episode step 2; the 90/128 audit
predates that fix and is not a rerun of the final source.

Requiring all four centres was an unnecessary constraint I introduced. Accepting any physical hole
while keeping supervision consistent within an attempt removes that restriction. Merely changing
success acceptance does not guarantee that the camera policy can locate and align with a hole.

## Perception and data lessons

- Camera FK matched native simulation over 1000 steps: maximum position error about 4.03e-7 m,
  quaternion-component error 7.44e-7 and no intrinsics difference. This rules out that particular
  simulation projection discrepancy; it does not validate real camera intrinsics/extrinsics.
- Pose-observer prototypes had root-position RMSE 23–27 mm and 95th-percentile errors 54–68 mm.
  A CNN's median error was only 5.3 mm, hiding large outliers. Those tails are unsuitable for precise insertion.
  Hole detection also had roughly 13.6 mm median and 44.5 mm 95th-percentile position error.
  These were prototype validation measurements, not independent complete-task success rates.
- One vectorized-data labeling bug omitted environment origins. It initially produced metre-scale
  root-label errors; origin subtraction fixed it. The corrected observer still had the outliers above.
- Reset camera frames can initially be zero. The current local goal selection waits two steps.
- 309 successful demonstrations from 512 privileged-controller episodes supplied useful training data,
  but neither demonstration count nor offline action MSE established camera-only rollout quality.

## Simulation and deployment changes worth retaining

Compared against the adjacent workshop checkout and our actual calibration:

- Corrected elbow coordinates by +6.4° with matching asset-frame/limit changes to preserve FK.
- Changed nominal vial grasping diameter to 28.9 mm and cap diameter to 35.4 mm; diameter randomization
  covers approximately ±0.5 mm. The workshop report's larger old body collider was relevant to grasping.
- Used **our follower's measured 127.3846° gripper span**, rather than copying the reported 134.6° from
  another robot. Its calibration spans 1449 ticks; real mapping still needs physical verification.
- Applied cylindrical body/cap plus spherical-bottom collision geometry, nominal friction 0.7,
  six-dimensional contacts and rolling/torsional friction 0.0005.
- Held position goals across the four physics steps and bounded per-control-step target changes.
  Retained self-collision and contact-force diagnostics.
- Added mass/inertia, friction, actuation, delay, measurement and appearance randomization.
  Image transforms approximate camera variation; they do not reproduce all physical extrinsic/parallax changes.

The user verified teleoperation and corrected camera focus. Real joint direction/zero alignment,
camera calibration and autonomous policy rollouts remain unverified. LEAPP numerical/export checks
passed for smoke and release-stage examples; those checks do not qualify a complete policy.

## Useful comparisons with the other solution

Compare its checkpoint on independent, randomized **home-start** episodes, stating task/preset, seed,
number of episodes and observation inputs. Separate full-task placement from pickup, insertion and
release-only success. Check that the exported actor needs no privileged object or target coordinates,
and that its action units, joint ordering, limits and control rate match the real interface.

Play currently retains physical/reset randomization but disables observation corruption; a separate
corruption-enabled audit is needed to measure that robustness. Record contact-force and failure rates
alongside success, and inspect rollout video for stable released seating in a physical opening.

## Provenance and current workspace

Published branch: `feat/so101-sim2real-multigpu`, last published commit
`b4d8226e83cf1cda85f047b4cc078c66bfeddcd0`. That snapshot predates the latest any-hole changes.
The current any-hole implementation and this note are local changes. The software suite passed 80 tests;
that is software validation, not policy acceptance. No training is currently running; the superseded
bottom-right home-only experiment was interrupted after iteration 883/1000.

Full chronology: [SO101_SIM2REAL.md](SO101_SIM2REAL.md).
Local audit logs (not committed):

- `/tmp/so101_corner_failure_audit.log`
- `/tmp/so101_corner_centre_gate_audit.log`
- `/tmp/so101_any_hole_latched_teacher_audit.log`
- `/tmp/so101_corner_home_teacher.log`

State checkpoint for the 90/128 audit:
`logs/rsl_rl/so101_vial_state/2026-10-07_01-14-38_sim2real_camera_grasp_teacher/model_2999.pt`.
