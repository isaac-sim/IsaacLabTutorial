# First supervised real trial — 2026-10-08

The user requested trying the current vision checkpoint despite its result being below the original
90% simulation acceptance gate. This is an experimental supervised trial, not a qualified deployment.
The checkpoint scored **879/1,024 (85.84%)** on a fresh clean simulated audit. One episode reached
20.16 N rack contact. Pickup is the main observed failure stage. It has not run on the physical arm.

## Prepared files and checks

Local bundle: `outputs/consolidation_20261008/supervised_trial/`.

- `model.pt`: final 500-update visual PPO checkpoint, sourced from
  `2026-10-08_16-40-56_consolidated_visual_ppo_4/model_496.pt`.
- `leapp/leapp.yaml`, `leapp/visual_actor.pt`, `leapp/contract.json`: executable visual policy.
- `joint_map.json`: recalibrated gripper scale, calibration-file fingerprint, **verified: false**.
- `start_pose.json`: actual reset-dataset home row and 0.035 rad (approximately 2°) tolerance.
- `manifest.json`, `cpu_runtime_parity.json`, `read_only_runtime.log`: provenance and validation.
- `home_overview.png`, `home_top.png`, `home_wrist_camera.png`, `reference_scene.json`: simulated setup.

Training-runtime export parity: maximum absolute error **0** on eight input pairs. Separate Torch
2.10 CPU deployment runtime versus Torch 2.13 training outputs: **5.37e-7**, within 1e-6 tolerance.
LEAPP auto-selected CUDA on the first export validation attempt; exporting with `CUDA_VISIBLE_DEVICES=''`
resolved the CPU/CUDA mismatch. That failed export is retained separately for diagnosis.

The physical camera/joint read-only inference test ran 20 seconds, 2,400 feedback steps, with **zero
missed 120 Hz deadlines**. Control work p95 was 3.62 ms and inference p95 was 2.27 ms. This excludes
motor-write latency and motion tracking; it is not a closed-loop hardware qualification. Read-only
mode can observe stale motor goal registers after calibration, so its predicted actions do not
validate task behavior from that resting pose.

## 1. Match the physical setup

Use the correctly powered follower, the unchanged focused wrist camera, an empty yellow rack and one
matching vial. Secure the arm base. Close teleoperation and other camera/serial applications. Keep the
leader out of the follower's working area. The first target is the fixed-rack training setup; arbitrary
rack positions and rotations were not trained in this campaign.

Open the rendered references from the repo root:

```bash
xdg-open outputs/consolidation_20261008/supervised_trial/home_overview.png
xdg-open outputs/consolidation_20261008/supervised_trial/home_top.png
```

Match the rack orientation and lying vial shown there. For interpreting the simulation layout,
world +X runs toward the objects and +Y is the lateral direction toward the rack. Relative to the
robot's CAD root at (-0.05, 0, 0), the four opening centers in the table plane are
(23, 8), (29, 8), (29, 14), (23, 14) cm. The rack center is therefore (26, 11) cm, **not** (23, 8).
These are CAD-frame measurements, not distances from the edge of the physical mounting bracket;
confirm the base-frame alignment with the reference images before using them as physical offsets.
The exact reference vial pose is in `reference_scene.json`. Place it in the same orientation and
region first; expand placement only after a successful repeatable baseline.

The 64×48 nominal wrist reference can be opened with:

```bash
xdg-open outputs/consolidation_20261008/supervised_trial/home_wrist_camera.png
```

It shows the rack toward the upper left and the jaws at the bottom. The reference vial is not fully
visible at this starting pose; this is the trained home distribution, not a claim of guaranteed
initial vial visibility. Do not rotate/remount the camera to center the rack or change home pose
without checking the resulting training/deployment mismatch.

## 2. Manually match and verify home

With the arm supported and freely movable after calibration, run:

```bash
sg dialout -c './scripts/so101_trial.sh inspect'
```

This only reads encoders. It does **not** release an already powered/holding arm; do not force joints
against active motors. It prints measured simulation degrees, signed home errors and whether all
errors are within 2°. Move the joints gently, supporting the arm, to match both the numerical target
and rendered physical pose. Press Ctrl+C to stop inspection.

| Joint | Target simulation degrees | LeRobot native target |
| --- | ---: | ---: |
| Shoulder pan | -7.00 | -7.00° |
| Shoulder lift | -51.88 | -51.88° |
| Elbow | 17.88 | 17.88° |
| Wrist flex | 84.87 | 84.87° |
| Wrist roll | -46.09 | -46.09° |
| Gripper | 14.64 | approximately 21.41% |

Numerical agreement alone does not verify the map. Confirm that the actual linkage shape and camera
orientation match the simulated references. In particular the elbow frame correction and gripper's
provisional -12.50575° zero still need physical confirmation. Share a photo of the arm at this pose
and the inspection readings so we can compare them; a new wrist-camera capture can then be taken
without disturbing the mount. If the numbers match but the geometry does not, correct the map before
motion. Do not merely change `verified` to bypass the check.

The prepared map remains unverified until this comparison is completed. A changed calibration file
also invalidates the map automatically. The gripper's newly measured span is 126.77°; the training
model retained 127.38°. This 0.615° full-span difference is recorded rather than hidden.

## 3. Read-only inference

After arranging the scene, this command can be run without enabling motor writes:

```bash
sg dialout -c './scripts/so101_trial.sh dry-run 20'
```

It uses the real wrist camera, two-frame raw RGB history, joint feedback, and exported policy, then
prints timing statistics. There are no motor configuration, torque or target writes in this mode.

## 4. First motion, after physical mapping verification

Once the map is physically checked and marked verified, start with a five-second supervised trial:

```bash
sg dialout -c './scripts/so101_trial.sh run 5'
```

The runner checks calibration identity, common joint limits and the home pose before enabling motion.
It initializes motor goals to the measured pose before configuring/enabling the motors. There is no
automatic homing move. The controller uses the trained 30 Hz policy and 120 Hz measured-relative
targets; this is **full trained action scale**, not a slow-motion mode. A five-second trial is too short
to expect task completion (simulated successes average about 14.7 seconds).

Stay at the power switch with the arm's path clear. Ctrl+C stops the loop; software cannot guarantee
an instantaneous physical stop, so use the power switch if motion is wrong. On normal completion,
Ctrl+C or a caught exception, cleanup disables torque and the arm may drop. Arrange a clear padded
resting area and support it once motion has stopped, keeping fingers out of joints and jaws.

Review approach direction, collisions, gripper alignment and timing before continuing. To attempt a
complete placement, manually reset the arm and objects to the verified starting arrangement, then:

```bash
sg dialout -c './scripts/so101_trial.sh run 30'
```

The real runner has no task-success detector and will continue until its time limit or interruption;
stop it after a successful placement. Recheck reported deadlines after the first motion trial because
read-only timings exclude goal writes. Record each attempt's outcome, failure stage and video when
available in this guide. No real success rate is claimed until those trials have been performed.

## Reproducing the export

The artifact directory is local and ignored by Git; the deployment scripts and this guide are tracked.
For a new export, choose an empty output directory and run:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m isaaclab_tutorial.utils.export_leapp \
  --model logs/rsl_rl/so101_vial_camera/2026-10-08_16-40-56_consolidated_visual_ppo_4/exported/policy.pt \
  --output /absolute/path/to/new/leapp
```
