# SO-101 sim-to-real

How to train a camera policy for the real WowRobo SO-101 follower and run it on the robot. The deployed
policy sees only wrist RGB and joint feedback, never the vial or rack state.

Two environments are involved:

- **Training machine:** the repository's locked uv environment (`uv sync --locked --extra leapp`).
- **Robot computer (Linux):** each hardware script pins its own CPU dependencies inline (LeRobot 0.6.1, CPU Torch,
  LEAPP 0.7.2), so run it with `uv run --script ...`. Isaac Lab is not needed there. The hardware sections
  (2–8) use Linux device paths and shell syntax.

Run every command from the repository root. The training and export commands are single lines and work on Linux
and Windows.

## 1. Training

`presets=sim2real` enables the transfer randomization on both tasks, so the teacher and the student train
under the same physics. The camera task also randomizes appearance and the camera.

| Randomized in both tasks | Added for the camera task |
| --- | --- |
| Vial, jaw, rack and mat friction | Robot, desk, rack and vial colors |
| Vial mass and diameter | Camera mount (±3 mm, ±3°) |
| Joint friction, armature and viscous damping | Focal length, principal point, radial distortion |
| Arm and gripper drive gains | Exposure, contrast, white balance, gamma, blur |
| Encoder bias of ±0.01 rad | |
| Vial and rack placement, mat height | |

The preset also changes the training setup:
- Training uses the measured shoulder/elbow response of this follower.
- The robot base sits on the real 35 mm tabletop, on the gray mat.
- Shoulder and elbow get larger action scales (0.12 and 0.07 rad), because small offsets stall the real joints.
- Episodes start from the home pose only.

```bash
# 1. State teacher on the nominal task.
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp

# 2. Continue the teacher under sim2real randomization (no rendering needed).
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --checkpoint /abs/path/state/model_799.pt --visualizer none presets=newton_mjwarp,sim2real

# 3. Distill a camera student from that teacher under the same randomization.
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint /abs/path/sim2real_state/model_XXX.pt --visualizer none presets=newton_mjwarp,newton_renderer,sim2real
```

To evaluate in simulation, run play from home starts and count exactly one episode per environment. The
sim2real preset keeps its physics randomization during evaluation.

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint /abs/path/distillation/model_XXX.pt --visualizer none --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter presets=newton_mjwarp,newton_renderer,sim2real
```

## 2. Physical setup

Training assumes the following setup. If you change any of it, the robot is outside the training distribution.

- **Robot:** an orange printed SO-101 follower. Its foot sits directly on a level wooden desk with no spacer,
  and the base is secured. Keep the leader arm out of the workspace.
- **Mat and objects:**
  - A dark gray mat about 390 mm square and 2–4 mm thick, under the rack and vial only.
  - The yellow four-opening rack and a vial with a blue cap.
  - Start near the reference layout. The policy only covers small offsets from it (vial ±20 mm / ±15°,
    rack ±5 mm / ±5°).
- **Camera:** the WowRobo/Sonix USB2.0 wrist camera on its standard gripper mount.
  - Use the `video-index0` capture node; `video-index1` is metadata only.
  - The camera must deliver **640×480 (4:3)** at 30 fps.
  - Focus is manual. Twist the lens until the vial is sharp at grasping distance:
    `ffplay -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0`. Close the preview before running
    anything else that opens the camera.
  - Do not rotate or re-mount the camera.
- **Lighting:** keep the scene evenly lit. Training varies global exposure and white balance, but not moving
  shadows, reflections or background clutter.
- **Ports:** use the stable `/dev/serial/by-id/...` and `/dev/v4l/by-id/...` paths, because the
  `/dev/ttyACM*` numbers change between sessions. `Permission denied` means your shell lacks the `dialout`
  group. Run `newgrp dialout`, then check with `id -nG`.

```bash
export SO101_FOLLOWER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00
export SO101_CAMERA=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
export SO101_JOINT_MAP=/absolute/path/to/joint_map.json
```

## 3. Robot calibration

The scripts reuse LeRobot's saved calibration for robot id `wowrobo_follower`, which is stored at
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/wowrobo_follower.json`. No script
calibrates on its own. To recalibrate:

```bash
uv run --script src/isaaclab_tutorial/utils/calibrate.py --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id=wowrobo_follower
```

Calibration turns torque off, so support the arm first.
1. For the middle pose, match the workshop's calibration-pose and wrist-centre reference images.
2. Sweep every joint except `wrist_roll` through its **full** mechanical travel, including the gripper. A
   missed endpoint shrinks the usable range.

**Every recalibration invalidates the joint map.**

## 4. Joint map

The joint map converts LeRobot readings into simulation radians: `sim_rad = scale * lerobot_value + offset`.
Copy `config/so101_joint_map.example.json` and fill it in.

| Field | Meaning |
| --- | --- |
| `units` | `"lerobot_degrees_and_gripper_percent"`: arm joints in degrees, gripper 0–100 %. |
| `joints.<arm>.scale/offset` | `π/180` and `0`. LeRobot degrees are centred on the calibration middle pose, which matches the simulation zero. |
| `joints.gripper.scale/offset` | `scale = (range_max − range_min) · 360/4095 / 100 · π/180`, using the calibration file (for example, 1442 ticks gives 0.0221254). `offset = −0.21826654582754615` rad, the fitted jaw zero. |
| `joints.<name>.limits` | Simulation joint limits in radians; keep the example's values. `deploy.py` narrows them further to the calibrated motor travel. |
| `calibration_sha256` | `sha256sum` of the calibration file. Homing requires it, and both scripts refuse to run if the file has changed. |
| `verified` | `deploy.py --execute` refuses to run unless this is `true`. Set it only after the checks below. |

**Verify the map.** `inspect_so101.py` only reads the motors and never changes torque:

```bash
uv run --script src/isaaclab_tutorial/utils/inspect_so101.py --joint-map "$SO101_JOINT_MAP" --start-pose config/so101_start_pose.json --port "$SO101_FOLLOWER_PORT" --watch
```

1. **Signs:** with torque off, support the arm and move each joint by hand. The printed simulation angle must
   move in the same direction as the simulated joint.
2. **Zeros:** home the arm (section 5). The readings must be within tolerance of the start pose.
   - Compare an external photo of the arm and a wrist-camera frame against a simulator render at the same pose.
     Linkage shape, jaw opening and camera orientation should all match.
   - If the numbers match but the shape does not, fix the map. Never set `verified` just to get past a check.

## 5. Home pose

`config/so101_start_pose.json` holds the training home pose. It contains six simulation-radian values in the
order pan, lift, elbow, wrist flex, wrist roll, gripper (−7.0°, −51.9°, 17.9°, 84.9°, −46.1°, 14.6°),
plus a tolerance.

`home_so101.py` moves the arm to this pose. It interpolates joint targets and does **not** avoid collisions,
so clear the rack and vial from the path first. Always preview before executing; the preview makes no motor
writes:

```bash
uv run --script src/isaaclab_tutorial/utils/home_so101.py --joint-map "$SO101_JOINT_MAP" --start-pose config/so101_start_pose.json --port "$SO101_FOLLOWER_PORT"
# then the same command with --execute   (optional: --speed-deg-s 3, at most 10; default 5)
```

- **Success:** all joints stay within 2° for 0.5 s. The script prints **Home reached** and the **motors stay
  on, holding the pose**.
- **Settling timeout:** if the error is at most 10°, the arm holds its measured pose and the script reports
  *home not confirmed*.
- **Torque turns off, and the arm can fall,** on any of:
  - a tracking error above 10°
  - feedback older than 100 ms
  - Ctrl+C

Afterwards, put the rack and vial back in the reference layout.

## 6. Export the policy (training machine)

1. **Write TorchScript.** Run `isaaclab play` on the selected checkpoint, with the same task and presets it was
   trained with. Play writes `<run_dir>/exported/policy.pt`.
2. **Package it for LEAPP.** Pass the TorchScript file, not a training checkpoint:

```bash
uv run --extra leapp python -m isaaclab_tutorial.utils.export_leapp --model <run_dir>/exported/policy.pt --output /absolute/path/to/new_bundle --history 2 --action-scale 0.033 0.12 0.07 0.033 0.033 0.02
```

- **`--action-scale`:** must equal the trained task's per-joint scales in radians. The values above are the
  sim2real preset's. Check `actions.*.scale` in the run's `params/env.yaml`, because a wrong scale silently
  changes the controller the policy sees.
- **`--history`:** the number of stacked camera frames. Both camera configurations use 2.
- **Output:** the directory must be new or empty. The bundle contains `<name>.yaml`, `visual_actor.pt` and
  `contract.json`.

Copy the whole bundle directory to the robot computer. `deploy.py` reads the scales, history and image
preprocessing from `contract.json`.

## 7. Run the policy

**Dry run first.** It opens the motor bus read-only (no configuration, torque or goal writes) and runs the real
camera and policy:

```bash
uv run --script src/isaaclab_tutorial/utils/deploy.py --bundle /abs/new_bundle/new_bundle.yaml --joint-map "$SO101_JOINT_MAP" --start-pose config/so101_start_pose.json --port "$SO101_FOLLOWER_PORT" --camera "$SO101_CAMERA" --duration 10
```

Check that `missed_120hz_deadlines` is about 0 and that the control-work p95 is a few milliseconds. Dry-run
actions are not meaningful, because their target comes from stale goal registers.

**Supervised motion:**
1. Home the arm (section 5) and reset the objects.
2. Run the same command with `--duration 5 --execute`.
3. If the approach looks right, home again, reset the objects and run `--duration 30 --execute`, the sim2real
   episode length.

Stop the run yourself once the vial is placed; there is no success detector. Every run must start from home
with fresh image and action history.

| Flag | Effect |
| --- | --- |
| `--execute` | Sends motor commands. Requires `verified: true`. Before enabling torque it checks the calibration hash, joint limits and start pose, then seeds each motor goal with its current position. |
| `--start-pose FILE` | With `--execute`, refuses to start unless every joint is within `tolerance_rad` of home. Always pass it. |
| `--trace NEW.npz` | Saves the 120 Hz measured/target/action trace and the policy-resolution RGB frames, including on Ctrl+C. The file must not already exist. |
| `--hold-at-end` | After a *normal* timeout, keeps torque on and holds the measured pose. Errors and Ctrl+C still release torque. |

The control loop matches simulation:
- Policy inference runs at 30 Hz.
- Every 1/120 s the joint target is `measured + scale · action`, clipped to the common simulation/hardware
  limits and sent.

## 8. Safety and pitfalls

- **Torque on exit:** after `--execute`, both a normal exit and Ctrl+C or an error **turn torque off**, and the
  arm drops unless you passed `--hold-at-end`. Keep a padded resting area and a hand on the power switch.
- **Torque after homing:** successful homing leaves torque **on**. Do not back-drive a holding arm by hand.
- **`Calibration changed since this joint map was prepared`:** the calibration file was rewritten. Re-verify
  the map (section 4); do not just copy the new hash.
- **`Robot calibration must already match its motors`:** recalibrate (section 3).
- **`Camera capture must have 4:3 aspect ratio`:** the camera is not delivering 640×480. Check the device node
  and that no other process holds the camera.
- **`Camera receipt is stale` / `Joint-feedback step exceeded 100 ms`:** the camera is shared, USB bandwidth is
  saturated, or the host is overloaded. The run aborts rather than send a stale command.
- **`outside the verified simulation/hardware limits`:** home the arm first. A fully closed gripper can sit
  outside the simulation limits.
- **Jaws pressing on the mat:** check that the base sits directly on the desk and that the mat thickness is
  right. Never compensate with a joint-map offset. Diagnose with `--trace` before changing anything else.
- **Small commands stall:** the shoulder and elbow do not move for target offsets of about 3° or less at some
  poses. This is why the sim2real preset uses the measured response and larger scales. Never raise the scales
  of an already exported policy.
