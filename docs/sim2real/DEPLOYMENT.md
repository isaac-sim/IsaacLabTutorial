# LEAPP and LeRobot deployment

For the current local experimental checkpoint and physical setup steps, use
[the SO-101 working guide, section 11](../SO101_SIM2REAL.md#11-current-supervised-real-trial--2026-10-08). Its 85.84% simulated success is distinct from the
historical multi-GPU results below.

> This report describes the multi-GPU source branch. The measured-model consolidation and new
> qualification status are recorded in [CONSOLIDATION.md](CONSOLIDATION.md).

`src/isaaclab_tutorial/utils/deploy.py` runs the explicit visual actor bundle described in
[DOMAIN_RANDOMIZATION.md](DOMAIN_RANDOMIZATION.md). It consumes real RGB and joint feedback;
it does not require privileged vial/rack state. It supports both image preprocessing modes and
repeats the first frame to initialize the same oldest-first history used in training.

Run the controller in its isolated CPU environment. Its inline dependency metadata pins LEAPP 0.7.1,
LeRobot 0.6.1, Torch 2.10 and torchvision 0.25. This avoids LeRobot's Hugging Face dependency conflict
with the pinned Isaac Lab environment. No simulation installation is needed on the robot computer.

```bash
uv run --script src/isaaclab_tutorial/utils/deploy.py \
  --bundle /absolute/path/to/bundle/leapp.yaml \
  --joint-map /absolute/path/to/verified_joint_map.json \
  --port /dev/serial/by-id/YOUR_FOLLOWER \
  --camera /dev/v4l/by-id/YOUR_WRIST_CAMERA --duration 30
```

This is a **read-only dry run**: it opens the motor bus without configuring motors or writing targets.
It reports control-work/inference percentiles and missed 120 Hz deadlines. Add `--execute` to send
commands only after verifying the joint map and setup. The script never automatically calibrates.
`config/so101_joint_map.example.json` preserves the earlier branch's unverified template; it is not
an accepted calibration for the current simulation. Check signs, zeros, limits, and especially the
gripper (LeRobot uses percent, not degrees) against actual measurements. The script intersects map
limits with calibrated motor travel. A `verified` flag alone is not evidence of a correct mapping.

The policy runs at 30 Hz. Its clipped action is held while targets are recomputed from measured
joint position at 120 Hz, matching the current simulation's four feedback substeps. Velocity is
estimated from successive joint reads and clipped to the training observation bounds. The target
observation is the last command actually sent. If the serial bus cannot sustain this loop, retrain
with the measured control cadence rather than assuming a slower loop is equivalent.

A background thread retains the latest 640×480 RGB camera frame. The controller resizes to 64×48;
the capture must be 4:3. It rejects camera receipts older than 100 ms, non-finite observations/actions,
and feedback steps exceeding 100 ms. Receipt time does not measure exposure or internal USB latency.
No frame-delay randomization or hardware latency measurement is claimed yet.

Offline tests cover image/history ordering, proprioception, action clipping, changing measured-relative
targets, signed joint transforms and invalid inputs. The isolated runtime imported LeRobot and ran the
actual selected visual actor. Eight varied input pairs matched the Torch 2.13 training runtime exactly
(maximum absolute error zero). **No motor connection or physical policy trial was performed.**
The current qualified bundle is `camera_randomization_20261007/selected/vision/leapp/leapp.yaml`
in the external artifact directory. Its contract specifies two **raw RGB** frames, without max-channel
intensity normalization. Fresh simulated audits exceed 94% with and without observation noise;
physical calibration and real-robot validation remain outstanding.

A second bundle validates the complete fresh-teacher/student pipeline:
`from_scratch_20261007/selected/vision/leapp/leapp.yaml`. It uses the same two-frame raw-RGB
contract and confirms 91.31% clean / 91.31% noisy simulated success. Its training and isolated CPU
runtime parity checks also match exactly; the earlier bundle remains the stronger qualified policy.
