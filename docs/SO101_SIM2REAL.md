# Wowrobo SO-101 sim-to-real working log

This is the continuing working guide originally started in `../mustafa_isaaclab2/SO101_SIM2REAL.md`.
That file already points here; the complete setup and experiment history is retained below.

Started: 2026-10-06. Update this file as we complete each step; proposed commands are not completed steps.

## Current status — 2026-10-08

The branches have been consolidated on `feat/so101-consolidated-sim2real`. Fresh training produced a
**91.02% state teacher** and an **85.84% vision policy**, each measured on 1,024 independent simulated
home-start attempts. The user selected the fixed-rack setup first and requested a supervised real trial
of this visual policy despite its result being below the original 90% acceptance threshold.

The LEAPP bundle is exported, CPU parity checks pass, and a 20-second real-camera/read-only inference
run completed without missed 120 Hz deadlines. The follower was recalibrated and its saved calibration
matches the motors. **Physical joint-map verification and real policy trials remain pending.** Motor-driven homing now passes with bounded feedback trim and holds the pose; see the homing
verification entry below.
Training is stopped. No real placement success rate is claimed.

Use [section 11: current supervised real trial](#11-current-supervised-real-trial--2026-10-08) for the
current setup and direct `uv` commands. Training and hardware use separate uv-managed environments;
the old shared-environment commands below record earlier configurations and are not current launch instructions.
Detailed consolidation evidence is in [sim2real/CONSOLIDATION.md](sim2real/CONSOLIDATION.md).

## Historical status before consolidation

The following status, setup and experiment entries preserve the earlier session history. Their
commands and results apply to the revisions and environments recorded at the time.

Historical status: teleoperation and focused camera capture work. The elbow, vial dimensions, calibrated gripper span,
control cadence, broad collision-filtered placement and camera-visible vial starts are implemented. The software
suite has 80 passing tests, and a recurrent LEAPP smoke export passes numerical validation. Current best audited
historical any-hole state-teacher placement is **93/128 broad home-start episodes (72.66%)** with self-collision enabled;
a usable visual policy has **not** been accepted.
The previous state teacher is trained. The superseded bottom-right adaptation has been stopped; no training is currently running. Wrist-camera control remains unqualified.
The 1,000-update broad-workspace camera PPO run finished without passing placement acceptance.
The current local `sim2real` task restricts rack yaw to −15°…+15° and accepts any empty hole.
Teacher/reward supervision latches an in-frame opening near the image centre. The latest teacher audit placed 90/128;
it predates the final two-step goal-latching delay. Section 10 records the evidence and artifacts.
A concise handoff is available in [SIM2REAL_FINDINGS.md](SIM2REAL_FINDINGS.md).

| Latest evidence | Measured result | Meaning |
| --- | --- | --- |
| Local any-hole state teacher 2999, seed 73 | 90/128 placements; one contact above 20 N, peak 28.94 N | Privileged teacher; audit predates final two-step latch delay |
| Bottom-right target, adapted state teacher100, seed67 | 3/128 placements; one contact above 20 N, peak39.17 N | First adaptation checkpoint did not improve baseline |
| Bottom-right target, old state teacher, seed67 | 5/128 placements; no contacts above 20 N | Historical bottom-right baseline; adaptation stopped |
| Bottom-right target, teacher with scripted search, seed65 | 6/16 placements; no contacts above 20 N | Small native geometry gate; no camera-policy qualification |
| Any-hole information-milestone camera PPO499, seed57 | 120 grasps,119 lifts,0/128 placements | Training completed; placement acceptance failed |
| Any-hole broad camera PPO999, seed57 | 115 grasps,111 lifts,0/128 placements | Training completed; placement acceptance failed |
| Visible-opening teacher, seed50 | 309/512 placements (60.35%); one contact above 20 N | Training-only source; 309 successful camera demonstrations saved |
| Larger-corpus camera fit40000, seed55 | 2/128 placements; 69 grasps; 50 lifts | Failed acceptance; subsequent PPO also failed placement |
| State teacher 2999, home seed45 | 93/128 placements; no contacts above 20 N | Privileged simulation teacher; not the physical camera policy |
| State teacher2350/2600, home seeds43/44 | 88/128 placements each; no contacts above 20 N | Privileged simulation teacher; not the physical camera policy |
| Camera grasp PPO200, home seed43 | 122/128 grasps;117/128 lifts;0 placements | Pickup improved; complete placement remains unsolved |
| Symmetry-aware active-view expert, home seed43 | 72/128 placements; full-rack framing in 92/100 lifted episodes | Training-only controller; no deployment claim |
| Camera release-clone100, release-only seed43 | 123/128 placements (96.1%); no contacts above 20 N | Validated release stage; not a complete-task score |
| Camera release-clone100, home seed43 | 0/128 grasps or placements | Release-only fine-tuning lost pickup; mixed-stage training is required |
| Trained release100 LEAPP export | 5/5 native examples and60/60 saved-image checks pass | Trained release-stage weights; not a complete-task qualification |
| Real robot deployment | Joint map remains unverified; no autonomous motion | Physical alignment and supervised trials remain pending |

## Goal and milestones

Use the Wowrobo SO-101 with this machine for sim-to-real work. First milestone: reliable leader-to-follower teleoperation.

- [x] Confirm host computer and inventory: both leader and follower are assembled.
- [x] Install and verify the initial LeRobot software environment.
- [x] Identify leader and follower USB ports; follower access verified through calibration.
- [x] Verify motor communication and resolve configuration if necessary.
- [x] Calibrate both arms and save their calibration files.
- [x] Verify basic teleoperation — user reports everything works.
- [x] Connect the wrist camera, capture its image stream, and verify adjusted optical focus.
- [ ] Match camera placement/projection with simulation and verify control-loop timing.
- [ ] Define the task and match the simulated and physical robot's joint order, offsets, units, limits, gripper convention, and control timing.
- [ ] Validate the policy in simulation, then perform supervised physical trials and record results.

The camera-based vial-placement task and LEAPP deployment are now the selected workflow; physical alignment and trials remain pending.
The current shared environment is the tutorial's `.venv`, described in section 6. Earlier sections
retain the original setup and troubleshooting history.

## References

Instructions were checked against Hugging Face's source-version documentation:

- [LeRobot installation](https://huggingface.co/docs/lerobot/main/installation)
- [SO-101 hardware setup](https://huggingface.co/docs/lerobot/main/so101)
- [Teleoperation tutorial](https://huggingface.co/docs/lerobot/main/il_robots#teleoperate)

The installed checkout predates the live documentation. We retain its exact revision and verify commands against its local implementation before use.

## 1. Host and software — completed

| Item | Recorded value |
| --- | --- |
| Host OS | Ubuntu 24.04.5 LTS, x86_64 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| NVIDIA driver | 595.91.07 |
| uv | 0.12.18 |
| Python | 3.12.13 |
| Existing LeRobot source | `/home/mhaiderbhai/code/lerobot` |
| Source revision | `b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf` |
| Package version reported by checkout | 0.5.2 |
| Dedicated environment | `/home/mhaiderbhai/.venvs/so101-lerobot` |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| TorchCodec | 0.11.1, CPU decoder build |
| Feetech SDK | 1.0.0 |
| System FFmpeg | 6.1.1 |

We reused the clean source checkout without updating it. A dedicated environment keeps robot dependencies separate from Isaac Lab. The installation guide supports uv and Python 3.12; `core_scripts` provides robot workflow dependencies and `feetech` adds motor support. We constrained PyTorch and TorchCodec for the guide's system-FFmpeg path. [Installation reference](https://huggingface.co/docs/lerobot/main/installation)

Commands executed successfully:

```bash
uv venv --python 3.12 /home/mhaiderbhai/.venvs/so101-lerobot
cd /home/mhaiderbhai/code/lerobot
uv pip install --python /home/mhaiderbhai/.venvs/so101-lerobot/bin/python \
  -e '.[core_scripts,feetech]' 'torch>=2.10' 'torchcodec>=0.10'
uv pip check --python /home/mhaiderbhai/.venvs/so101-lerobot/bin/python
```

Verification passed: dependency compatibility, `lerobot-info`, SO101 follower/leader imports, Feetech and TorchCodec imports, a CUDA tensor operation, and `--help` for calibration and teleoperation. Video decoding, cameras, motor communication, calibration, and physical motion have not been tested.

### Tutorial sim2real extra — completed

Current project: `/home/mhaiderbhai/code/IsaacLabTutorial`, branch `feat/sim2real-extra`.
The earlier standalone environment above remains available; use the tutorial's shared `.venv` for
subsequent work. The user wants routine software changes performed by the assistant, with a clear
explanation; the user will take over for hardware interactions and observing physical behavior.

Changes made:

- Added `[project.optional-dependencies].sim2real` with `lerobot[core-scripts,feetech]==0.6.1`
  and `torchcodec>=0.10,<0.12`.
- Updated `uv.lock` and installed the extra into `IsaacLabTutorial/.venv`.
- Added installation instructions and the compatibility explanation to the tutorial README.
- Added the project-wide override `transformers==5.10.4`, matching the current Isaac Lab checkout.
  The tutorial's pinned Isaac Lab requires Transformers 4.57.6, which requires Hub <1; LeRobot
  requires Hub 1.x. The original Git-pinned LeRobot proposal could not resolve together with Isaac Lab.
  LeRobot 0.6.1 from PyPI avoids an additional conflict between source-defined CUDA indexes.

The Transformers override affects the base tutorial environment too. Its own policy encoders pass
the tutorial tests; transformer-based image feature models from the older Isaac Lab revision have
not been validated against Transformers 5. Do not interpret the override as universal compatibility
with all optional Isaac Lab features.

Installed versions: LeRobot 0.6.1, Transformers 5.10.4, Hub 1.33.0, NumPy 2.2.6,
TorchCodec 0.11.1, and Feetech SDK 1.0.0. PyTorch remains 2.11.0+cu128.

Successful configuration and installation command:

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
# The Transformers override was added to pyproject.toml before resolving.
uv add --optional sim2real --no-sync \
  'lerobot[core_scripts,feetech]==0.6.1' 'torchcodec>=0.10,<0.12'
uv sync --locked --extra sim2real
```

Verification: all 61 existing tutorial tests passed; Ruff and whitespace checks passed; calibration
and teleoperation CLI help loaded; SO-101, Feetech, TorchCodec, and Transformers imports succeeded;
the tutorial task entry point was discovered; a CUDA tensor operation passed; and a temporary MP4
was successfully encoded with system FFmpeg and decoded with TorchCodec. `uv lock --check` passed.

For each new terminal, activate the tutorial environment before the direct hardware commands below:

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
source .venv/bin/activate
lerobot-info
```

Alternatively, run hardware commands as `uv run --locked --extra sim2real lerobot-find-port`, etc.
Keep `--extra sim2real` on `uv run` commands; synchronization without it can remove optional packages.

## 2. USB identification — completed

At the initial check, there were no `/dev/ttyACM*`, `/dev/ttyUSB*`, or `/dev/serial/by-id` entries,
and the user was not in `dialout`. These conditions have since been resolved.

The user subsequently identified the follower as `/dev/ttyACM0` and the leader as `/dev/ttyACM1`.
Both devices now exist with mode `0660`, owned by `root:dialout`. Their stable aliases are:

- Follower: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00`.
- Leader: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0179854-if00`.

An unattended attempt to add `mhaiderbhai` to `dialout` failed because sudo required a password.
The user then reported completion of the permissions and follower-calibration steps. A subsequent
check confirmed `dialout` membership and a saved follower calibration containing all six motors.
Leader communication and calibration subsequently passed the read-only preflight recorded below.

Secure both arms, clear the follower's workspace, and use the power supplies specified for the supplied Wowrobo kit. Keep access to motor power and keep hands clear during powered motion. Confirm the leader and follower supplies individually from their labels/manual.

Connect the follower adapter's USB and power, then run:

```bash
lerobot-find-port
```

Follow its unplug/reconnect prompt. Repeat for the leader and label the cables. This tool is interactive: unplug the adapter being identified when prompted. [Port discovery instructions](https://huggingface.co/docs/lerobot/main/so101#1-find-the-usb-ports-associated-with-each-arm)

Inspect device names and permissions:

```bash
ls -l /dev/serial/by-id/
id -nG
```

Prefer distinct `/dev/serial/by-id/...` paths when available. Otherwise record the discovered `/dev/ttyACM...` or `/dev/ttyUSB...` paths and recheck after reconnecting.

If the devices belong to `dialout` and access is denied, add the current user:

```bash
sudo usermod -aG dialout "$USER"
```

Then run `newgrp dialout` to start a shell with the group enabled, or log out of the desktop and back in.
Run subsequent hardware commands inside that new shell. Group membership is now confirmed;
there is no need to repeat this permissions change.

Set these variables in the terminal using the confirmed stable paths:

```bash
export SO101_FOLLOWER_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00'
export SO101_LEADER_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0179854-if00'
```

| Setting | Actual value |
| --- | --- |
| Follower port | `/dev/ttyACM0`; stable alias ending `5AE6079843-if00` |
| Leader port | `/dev/ttyACM1`; stable alias ending `5AB0179854-if00` |
| Follower calibration ID | `wowrobo_follower` (proposed) |
| Leader calibration ID | `wowrobo_leader` (proposed) |
| Vendor motor configuration confirmed | Unknown |

## 3. Motor communication and calibration — both verified

Assembly does not establish whether motor IDs and baud rates were configured. Start with the calibration commands below; if motor discovery fails, capture the error and verify ports, power, and vendor setup. Stop at that error before continuing.

`lerobot-setup-motors` writes motor configuration and requires one motor connected at a time. Use it only if needed, following the official SO-101 procedure; do not run it blindly on the assembled daisy chain. [Motor configuration instructions](https://huggingface.co/docs/lerobot/main/so101#2-set-the-motors-ids-and-baudrates)

Calibrate separately so we can inspect each result before starting motion:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$SO101_FOLLOWER_PORT" \
  --robot.id=wowrobo_follower

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="$SO101_LEADER_PORT" \
  --teleop.id=wowrobo_leader
```

Follow the installed script's prompts for the middle pose and manual range sweep. Support the arm when torque is disabled, move gently, and never force a mechanical stop. The installed follower implementation excludes `wrist_roll` from the manual sweep; follow its prompt rather than treating every joint identically.

Record each printed calibration file path here. Keep the same IDs during teleoperation and subsequent recording/evaluation because they identify stored calibrations. [Calibration and ID guidance](https://huggingface.co/docs/lerobot/main/il_robots#teleoperate)

- Follower calibration file: `/home/mhaiderbhai/.cache/huggingface/lerobot/calibration/robots/so_follower/wowrobo_follower.json`.
- Leader calibration file: `/home/mhaiderbhai/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/wowrobo_leader.json`.

Follower calibration was confirmed by reading the saved file after the user reported completion.
All six expected joint entries exist with IDs 1–6 and increasing raw position ranges. This checks
the saved data's structure; correct physical joint behavior still needs the teleoperation check.

| Follower joint | Motor ID | Homing offset | Raw minimum | Raw maximum |
| --- | --- | --- | --- | --- |
| shoulder_pan | 1 | 420 | 757 | 3421 |
| shoulder_lift | 2 | -118 | 1531 | 3875 |
| elbow_flex | 3 | 662 | 381 | 2485 |
| wrist_flex | 4 | 1368 | 86 | 2380 |
| wrist_roll | 5 | -1908 | 0 | 4095 |
| gripper | 6 | -627 | 2047 | 3496 |

These are native motor calibration values, not simulation joint angles.

Leader calibration was attempted but failed during the motor handshake on `/dev/ttyACM1`:
`FeetechMotorsBus motor check failed`, with IDs 1–6 missing (expected model 777) and
`Full found motor list: {}`. The serial port opened, but no expected motor responded.
The user identified the cause: the leader's motor power cable was not plugged in.
The USB connection alone made the adapter visible. No motor ID or baud-rate changes were made.
After connecting power, the user retried. The handshake now found `{5: 777}` but still missed
IDs 1, 2, 3, 4, and 6. This confirms a response from at least one expected motor during that attempt;
it does not establish the physical joint corresponding to ID 5 or the remaining motors' configuration.
Successful leader calibration remains pending.

### Leader communication diagnostics — historical; resolved below

The following notes preserve the failed attempts and proposed checks. Current status is recorded
under "Resolution and calibration preflight" below; troubleshooting retries are no longer required.

The assistant inspected the installed LeRobot implementation: its leader handshake pings expected
IDs 1–6 at the default 1,000,000 baud and expects model number 777 (STS3215).
The stable leader USB alias still resolves to `/dev/ttyACM1`.

Two read-only diagnostic runs were made using a temporary script, with `sg dialout` to give the
assistant's existing process serial access after the user's group-membership change:

```bash
sg dialout -c '/home/mhaiderbhai/.local/bin/uv run --no-project --python /home/mhaiderbhai/code/IsaacLabTutorial/.venv/bin/python python /tmp/so101_read_only_scan.py'
```

The script created `FeetechMotorsBus` with no configured motors, opened the adapter with
`connect(handshake=False)`, and sent ping queries. It scanned IDs 0–253 at 1,000,000 baud and repeated
IDs 1–6 three times. The extended second run also queried IDs 1–6 at 4,800, 9,600, 14,400, 19,200,
38,400, 57,600, 115,200, 128,000, 250,000, and 500,000 baud. Both runs found no responses, including
from ID 5. No torque, position, motor ID, calibration, or motor baud-rate registers were written;
baud-rate changes applied only to the host adapter. The script closed with
`disconnect(disable_torque=False)` to avoid the default torque-register writes.

The scan did not reproduce the user's ID-5 response. The cause remains unresolved; motor power,
bus wiring, intermittent communication, and motor ID configuration remain possibilities.
The scan does not rule out nonstandard IDs at other baud rates or duplicate IDs.

Next physical checks: with leader power off, inspect and reseat the controller-to-motor 3-pin cable
and each motor-to-motor link, then restore the specified supply and confirm its power indication.
For a Waveshare board, confirm both routing jumpers select B/USB, as described in the
[official troubleshooting instructions](https://huggingface.co/docs/lerobot/v0.6.1/so101#2-set-the-motors-ids-and-baudrates).
Confirm whether Wowrobo factory-configured all leader IDs before considering motor setup.

After the user unplugged and reseated the connections, another calibration attempt found `{}`:
all six expected IDs were missing again. The assistant rechecked the USB aliases and confirmed that
the leader adapter `5AB0179854` still maps to `/dev/ttyACM1`; the follower is still `/dev/ttyACM0`.
The disappearance of ID 5 does not identify a specific failed motor or prove an ID configuration error.

Single-motor diagnostic originally proposed, not performed: power off the leader, connect its controller directly
to one accessible leader motor (for example, the gripper), and disconnect that motor from the rest
of the motor chain. Use the kit's specified motor cable and supply; do not guess connector orientation
or substitute the follower's supply. Restore power and let the assistant run read-only discovery on
the isolated motor. This follows the official guide's single-motor connection arrangement while
omitting the configuration writes. If connector routing is unclear, inspect a photo of the controller
and wiring before changing connections. If one motor responds in isolation, check the other motors
individually before deciding whether ID setup or cable repair is necessary.

The user reported that the motor connectors are difficult to remove. Single-motor isolation is
deferred; do not force connectors or proceed with dismantling the motor chain. The next step is
visual inspection of the leader controller with its existing wiring intact: obtain a clear photo
showing power and motor connections, indicator LEDs, and any routing jumpers, plus a photo of the
leader power-supply label. Identify the actual controller and supply before recommending routing
changes or further electrical checks. Current leader motor IDs remain unverified.

The user subsequently reported that all motors appear to flash red. It is not yet confirmed whether
this means the leader only or both arms; blink timing, exact servo variants, and the connected supply
label are pending. Do not assign a specific fault code from the LED colour alone. Feetech documents
voltage, current, temperature, and overload protections and sells both 7.4 V and 12 V STS3215 variants:
[manufacturer product listing](https://www.feetechrc.com/products.html?keyword=STS3215) and
[manufacturer protection description](https://www.feetechrc.com/20210430-56680.html).
The next priority is to pause calibration and motor commands, disconnect the leader's external motor
supply while checking its label, and compare the supply with the kit's actual leader specifications.
Do not try the follower's supply or change motor protection limits to clear the indication.

Interpretation correction for the earlier diagnostics: installed LeRobot's `ping()` returns `None`
both for failed communication and for a nonzero servo error flag. Therefore the reported empty
scan results mean no successful, error-free model replies; they do not establish that no packets
arrived. A protection alarm could contribute to the missing-ID result, but this is unconfirmed until
the correct supply is verified and raw communication/error flags or status registers can be read.

Calibration command used for the leader (no further retry needed after the successful preflight):

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
uv run --locked --extra sim2real lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=wowrobo_leader
```

### Resolution and calibration preflight — completed

The user identified that the wrong power supply had been used, reported correcting it, and confirmed
that the calibration step had been performed for both arms. The exact supply-label voltage/current
has not been recorded; do not infer it from the generic servo model.

The assistant confirmed both saved calibration files and ran a read-only check on both arms using
their stable USB aliases. This instantiated the corresponding LeRobot robot/teleoperator config
with the saved IDs, called only `device.bus.connect()` and `device.bus.is_calibrated`, and closed
each bus with `disconnect(disable_torque=False)`. The motor handshake and calibration comparison
read registers; robot/teleoperator `connect()` methods that configure motors were not called.
No torque commands, motion targets, or calibration writes were sent.

Command executed:

```bash
sg dialout -c '/home/mhaiderbhai/.local/bin/uv run --no-project --python /home/mhaiderbhai/code/IsaacLabTutorial/.venv/bin/python python /tmp/so101_calibration_preflight.py'
```

Results:

```text
wowrobo_follower: motor handshake PASS; saved calibration matches hardware: True
wowrobo_leader: motor handshake PASS; saved calibration matches hardware: True
```

All six expected motors on each arm passed the handshake. Saved ranges and homing offsets match
the motor registers. No recalibration or motor ID setup is needed for the first teleoperation check.

| Leader joint | Motor ID | Homing offset | Raw minimum | Raw maximum |
| --- | --- | --- | --- | --- |
| shoulder_pan | 1 | -921 | 1054 | 3148 |
| shoulder_lift | 2 | -370 | 1318 | 3682 |
| elbow_flex | 3 | -380 | 676 | 2889 |
| wrist_flex | 4 | -1898 | 326 | 2619 |
| wrist_roll | 5 | -521 | 0 | 4095 |
| gripper | 6 | 363 | 1908 | 3129 |

## 4. First teleoperation — completed

After both calibrations succeed, place the arms in similar comfortable poses with clearance around the follower. Start with no cameras:

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
uv run --locked --extra sim2real lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00 \
  --robot.id=wowrobo_follower \
  --robot.max_relative_target=5 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0179854-if00 \
  --teleop.id=wowrobo_leader \
  --fps=30
```

The base command follows the [Hugging Face teleoperation tutorial](https://huggingface.co/docs/lerobot/main/il_robots#teleoperate). We add a 30 Hz loop and the locally verified `max_relative_target` option for the initial trial. That option caps each command's change from measured follower position: arm joints use degrees in this checkout; the gripper uses its normalized scale. It is not a collision detector or a speed guarantee.

Move the leader slowly, checking one joint at a time: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, and gripper. Confirm the expected direction, smooth tracking, and absence of communication errors. Stop with Ctrl+C; the installed follower defaults to disabling torque on disconnect, so support it if it could fall. Cut motor power immediately if motion is unexpected.

Pass criteria: all six joints respond correctly, modest combined motions track smoothly, and the session exits cleanly. This validates basic teleoperation only; sim-to-real mapping and autonomous policy control still require their own checks.

## Session record

| Date | Action | Result / remaining work |
| --- | --- | --- |
| 2026-10-06 | Confirm computer and arms | This host; assembled leader and follower |
| 2026-10-06 | Install isolated LeRobot environment | Completed; 91 packages; dependency/import/CUDA/CLI checks passed |
| 2026-10-06 | Inspect USB devices | No serial adapters visible; connect hardware next |
| 2026-10-06 | Add and install tutorial `sim2real` extra | LeRobot 0.6.1 installed in tutorial `.venv`; Transformers compatibility override recorded; 61 tests and software checks passed |
| 2026-10-06 | Identify ports and access | Follower ACM0; leader ACM1; stable aliases recorded; user dialout membership subsequently confirmed |
| 2026-10-06 | Calibrate follower | User reported completion; saved JSON verified for all six joints; calibration values recorded above |
| 2026-10-06 | Attempt leader calibration | No motors responded; user identified unplugged leader power cable; powered retry pending |
| 2026-10-06 | Retry powered leader calibration | Only ID 5/model 777 responded; IDs 1–4 and 6 missing; calibration remains pending |
| 2026-10-06 | Run read-only leader scans | No motor responses during either scan; host baud rates and query coverage recorded above; physical checks next |
| 2026-10-06 | Retry after reseating leader connections | User reported all six IDs missing again; adapter still ACM1; single-motor isolation proposed, not yet performed |
| 2026-10-06 | Adjust diagnostic approach | User reports tight motor connectors; isolation deferred; controller and supply photos requested with motor wiring intact |
| 2026-10-06 | Observe flashing red motor LEDs | User observation recorded; fault meaning and affected arm(s) unconfirmed; verify supply/servo variants before further motor commands; ping error-filtering limitation noted |
| 2026-10-06 | Correct supply and confirm both calibrations | User identified wrong supply, reported correction and calibration of both arms; both JSON files confirmed |
| 2026-10-06 | Read-only calibration preflight | Both motor handshakes passed; saved calibration matches hardware for both; no writes or motion commands |
| 2026-10-06 | First teleoperation | User reports everything works; removed unnecessary pose-holding guidance at their request |

For later sim-to-real work, retain calibration files, software revision, camera settings, control units, timing, and trial results. We will extend this log with the exact simulation and deployment commands as those decisions are made.


## 5. Camera bring-up — stream and adjusted focus verified

Camera: Sonix USB2.0 CAM1. Image-capture endpoint:
`/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0`
(currently `/dev/video0`). The `video-index1` endpoint is metadata, not the image stream.

Saved a real 640 × 480 frame to:
`/home/mhaiderbhai/code/IsaacLabTutorial/outputs/camera/so101_camera_check.png`.
The initial frame showed the rack, vial and gripper, but was severely out of focus.
After the user manually adjusted the lens on 2026-10-06, captured and visually inspected
`outputs/camera/so101_camera_focused.png` at 640 × 480. The vial's measurement markings,
cap edges and desk grain are now clearly resolved. The nearer gripper tips remain slightly soft;
focus is substantially improved at the vial's working distance. Camera framing and projection
still need alignment with simulation before physical policy trials.

Repeat capture without opening a motor port:

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
uv run --locked --extra sim2real python scripts/so101_policy.py \
  --capture-image outputs/camera/so101_camera_check.png
```

The script warms the camera for 15 frames, saves the original BGR frame as PNG, and converts BGR to RGB
explicitly for policy inference. The training camera is 64 × 48; deployment resizes the 4:3 capture
with area interpolation. LEAPP owns the tensor normalization and observation/action preprocessing.

The user's setup photo is `/home/mhaiderbhai/Downloads/IMG_5967.jpg`: orange printed robot parts,
yellow four-opening rack, clear blue-cap vial and a wooden desk. Colors can be inferred from this photo;
metric geometry, calibrated camera projection and robot-base-relative object poses cannot.

### Manual lens focus — user adjustment completed

Read-only `v4l2-ctl --device /dev/video0 --list-ctrls` lists exposure, white balance, sharpness
and other image controls, but no focus/autofocus control. The uniform optical blur is consistent
with incorrect lens focus. The [SO-101 wrist-camera installation guide](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Optional/Wrist_Cam_Mount_32x32_UVC_Module/README.md)
describes manual focus by twisting the lens.

Keep the robot stationary, place the vial at the expected grasping distance, and preview:

```bash
ffplay -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0
```

Turn the lens barrel gently in small increments (roughly one eighth of a turn), watching the
label and cap edges. Reverse direction if the image becomes blurrier; make smaller adjustments
near best focus. Do not force a locked lens or unscrew it out of its holder. Focus on the task's
working distance, not the far background. Software sharpness cannot correct optical defocus.
Close the preview before running the capture/inference script, which also opens the camera.
The user performed the lens adjustment; the assistant verified it with a fresh capture.
To repeat that check, use the capture command above with
`--capture-image outputs/camera/so101_camera_focused.png`.

## 6. Isaac Lab develop update — completed

Updated the tutorial's exact Git pin from `f754f2965af9d6aa4f2e18d7d6d3ddf8f8471c69`
to upstream develop `f5383e7feb4c433372caf0d4d537e1e6c03ef0f2`, the latest upstream revision when checked.
This is a fixed revision for reproducibility, not a floating branch pin. It includes the camera export
fix `953d0425d` (LEAPP camera inputs and frame history). OpenUSD 26.08 from usd-exchange 3.0.0
also removes the need for the old `PXR_WORK_THREAD_LIMIT=1` workaround; the README was updated.

Current shared environment: PyTorch 2.12.0+cu130, torchvision 0.27.0+cu130,
torchaudio 2.11.0+cu130, Newton 1.6.1, usd-exchange 3.0.0, RSL-RL 5.5.1,
LEAPP 0.6.1 and LeRobot 0.6.1. The prior Transformers override was removed because develop already
requires Transformers 5.10.4. LeRobot caps Torch/torchvision below develop's requirements;
project-wide overrides use develop's versions. CUDA, SO-101 imports, and a real read-only policy loop
were checked; these overrides do not establish compatibility with every optional LeRobot training model.

```bash
uv sync --locked --extra sim2real
```

Update-related fixes in the tutorial:

- Replaced removed private `NewtonManager._builder` access with the simulation backend registry.
- Made the placement-history reset accept slice indices passed by current Isaac Lab.
- Added the optional `action_name` argument required by LEAPP's last-action feedback wrapper.
- Increased MJWarp line-search iterations from 15 to 50 after the randomized 1,024-world smoke run
  emitted convergence warnings. The fresh smoke run completed without those warnings.
- Used native OpenUSD material authoring/binding for kitless tabletop/rack colors; the generic file
  spawner otherwise skips creation of new visual materials without Kit.

## 7. Scene defaults and sim2real preset — implemented and smoke-tested

Defaults now show orange robot plastic (black servo materials preserved), a yellow rack and a brown
surface. Existing collision geometry and task coordinates were retained; they are not measured from
the photograph. Baseline vial mass/friction are deterministic. Camera/proprioception corruption is
disabled in the default camera task and enabled by the `sim2real` preset.

Isaac Lab's native selector is plural, `presets=sim2real`. The tutorial adds the `so101` launcher,
which forwards to Isaac Lab and accepts the requested singular spelling:

```bash
uv run --locked --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera preset=sim2real
```

`sim2real` works with the task's default Newton MJWarp physics and Newton Warp camera renderer;
no additional backend selector is required. The same selector is used for training, evaluation and export.

| Quantity | Range/model | Timing |
| --- | --- | --- |
| Vial mass | 12–30 g | Startup, independently per environment |
| Robot body mass | 0.85–1.15 × nominal | Startup |
| Robot/vial inertia | 0.8–1.2 × nominal | Startup |
| Contact friction | Vial 0.5–1.4; robot 0.6–1.5; rack 0.4–1.2 | Startup |
| Restitution | 0–0.03 | Startup |
| Motor stiffness/damping | 0.7–1.3 × nominal, per joint | Startup |
| Joint friction / armature | 0.5–1.5 / 0.8–1.2 × nominal | Startup |
| Gravity / slight mounting tilt | x,y ±0.003 m/s²; z −9.91 to −9.71 | Startup |
| Rack / starting vial placement | 0.16–0.38 m radius, −100°…+100° sector, independent full yaw; initial vial filtered to wrist view | Reset; close-approach/held/insertion replay preserved |
| Vial body / cap diameter | 28.4–29.4 / 34.9–35.9 mm | Per world, before Newton finalization |
| Printed-part, rack and tabletop colors | Orange, yellow and brown RGB ranges in `sim2real_cfg.py`/events | Reset, per world |
| Exposure / contrast / white balance | 0.6–1.4 / 0.7–1.3 / 0.8–1.2 | Reset |
| Brightness | ±0.08 normalized intensity | Reset |
| Image projection variation | Zoom 0.9–1.1; rotation ±0.05 rad; translation ±0.05 normalized coordinates | Reset |
| Focus variation | 0–0.6 blend with a 3 × 3 blurred image | Reset |
| Pixel noise | Gaussian σ 0.01 plus uniform ±0.025, normalized RGB | Each observation |
| Camera / joint-position sensing delay | 0–2 control steps, up to about 67 ms | Reset |
| Encoder/calibration position bias | ±0.015 rad, plus per-step uniform ±0.01 | Reset / observation |
| Velocity / target observation noise | ±0.02 rad/s / ±0.005 rad | Observation |

Newton uses one friction coefficient, so the dynamic-friction argument is not an independently randomized
parameter on this backend. Startup sampling produces a fixed population of physical variants across
1,024 worlds; it does not resample physical parameters on every episode.

Limits of this first preset: affine image warping approximates small camera projection/alignment errors,
not full extrinsic parallax or calibrated lens distortion. Lighting changes are represented by image
appearance augmentation; wood grain, background clutter and moving shadows are not physically rendered
randomizations. Geometry variation currently covers vial diameters only. Mass-center changes, explicit command-transport delay,
servo saturation/backlash and detailed electrical/current models are not covered. CoM and collider-offset
experiments were removed from the final configuration during smoke-test diagnosis. We must measure
geometry, camera calibration, joint directions/zeros and the real controller response before claiming
transfer. No finite randomization set guarantees seamless sim2real.

Smoke tests completed: 16 worlds × 3 PPO iterations; then 1,024 worlds × 10 iterations and
1,024 worlds × 3 iterations after increasing the solver allowance. The last run had no
line-search warnings and no unstable-robot terminations.

## 8. Training and LEAPP export — in progress

Task: `IsaacTutorial-Place-Vial-SO101-Camera`; RSL-RL PPO; seed 42; 30 Hz control.
Warm start: existing camera checkpoint
`logs/rsl_rl/so101_vial_camera/2026-09-05_09-02-05_repro_20260905_vision_seed42_retry/model_4999.pt`.
This trains a new randomized policy; it is not training from scratch.

Full run launched:

```bash
uv run --locked --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 \
  --max_iterations 1200 --seed 42 --run_name sim2real \
  --checkpoint logs/rsl_rl/so101_vial_camera/2026-09-05_09-02-05_repro_20260905_vision_seed42_retry/model_4999.pt \
  preset=sim2real
```

Run directory: `logs/rsl_rl/so101_vial_camera/2026-10-06_18-09-22_sim2real`.
Console log: `/tmp/so101_sim2real_training.log` (will be copied into the run artifacts).
The first exploratory run was stopped at iteration 5573 after a failed home-pose audit;
its saved checkpoints are retained for diagnosis. Training episode success uses mixed reset phases;
it is not the canonical-home evaluation.
`Metrics/success_rate` is an instantaneous environment fraction, not completed-episode success.

### Randomization ablation and training correction

The interim camera checkpoint `model_5200.pt` completed 0/128 canonical home-pose episodes,
despite roughly 60% success in mixed training resets. It must not be deployed.
To separate physics changes from camera changes, evaluated the existing state teacher with
the current dependencies and deterministic actor mean, using one first episode per world:

| Evaluation condition | Successful placements / 128 |
| --- | --- |
| Updated simulation, nominal physics | 128 |
| Full initial sim2real physics and placement randomization | 0 |
| Placement offsets alone | 122 |
| Mass and inertia variation alone | 127 |
| Contact-property variation alone | 115 |
| Servo parameters and gravity variation alone | 6 |
| Actuator gains alone | 105 |
| Joint friction and armature alone | 124 |
| Original gravity variation alone (x,y ±0.15 m/s²) | 5 |
| Full randomization with zero horizontal gravity | 62 |
| Full randomization with x,y ±0.003 m/s² | 47 |

The original horizontal gravity range was the dominant failure. Reduced it to ±0.003 m/s²:
large effective tabletop tilt can move a free horizontal vial before the grasp. This is a
level-table operating assumption that still requires checking on the physical setup.
The combined corrected randomization still lowers the old teacher to 47/128; randomization families
interact, so their individual success rates cannot be combined as a transfer prediction.
Temporary ablation scripts live in `/tmp`, not in the tutorial's deployment code.
Distillation now also selects randomized camera/proprioception observations with `preset=sim2real`;
the privileged teacher inputs stay clean. Both preset variants passed the focused configuration tests.
Logs have been copied into `outputs/sim2real_audit`.

The next training stage adapts the state teacher to the same corrected physical preset before
using it to train the camera student. The state task now supports the same `preset=sim2real` selector.

```bash
uv run --locked --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 1024 \
  --max_iterations 400 --seed 42 --run_name sim2real_teacher \
  --checkpoint logs/rsl_rl/so101_vial_state/2026-09-05_08-08-28_repro_20260905_state_seed42/model_799.pt \
  preset=sim2real
```

Run: `logs/rsl_rl/so101_vial_state/2026-10-06_18-32-51_sim2real_teacher`.
The old distilled camera student also failed 0/128 on the corrected preset and changed scene defaults;
it needs visual retraining rather than direct deployment.

Early teacher audits: `model_850.pt` achieved 80/128 home-pose successes; `model_950.pt`
achieved 95/128 within 20 seconds and 101/128 with a 30-second horizon. These are intermediate
checks, not the final camera-policy acceptance result.

`scripts/distillation_checkpoint.py warm-start` preserves an existing camera student's weights,
replaces the privileged teacher with the newly trained state actor, clears optimizer moments,
and resets the training iteration counter. A 16-world, two-iteration retraining smoke test passed.

The pinned upstream LEAPP RSL-RL exporter selects `obs_groups["actor"]`, even for a distillation
runner whose deployable groups are named `student`. To export safely through its maintained camera
actor path, `scripts/distillation_checkpoint.py export` copies the student's weights unchanged into
an architecture-matching camera PPO checkpoint. The template critic is required for loading but
is excluded from LEAPP and inference. This is an export package, not a PPO-trained policy.
The converted smoke checkpoint passed LEAPP validation on all five recorded examples.

The first adapted teacher finished its 400 additional iterations in 597.71 seconds.
Final `model_1198.pt`: 108/128 home-pose successes (84.4%), 92.2% grasp,
89.1% lift, 87.5% insertion, 11.7% timeout, 3.9% vial loss and zero unsafe rack impacts.
Maximum recorded rack force was 10.60 N. Evaluation used seed 42, full corrected physics
randomization, and a 20-second horizon.

Started camera retraining with that teacher and the old camera student's weights:

```bash
uv run --locked --extra sim2real python scripts/distillation_checkpoint.py warm-start \
  --student logs/rsl_rl/so101_vial_camera_distillation/2026-09-05_08-34-40_repro_20260905_distillation_seed42/model_1599.pt \
  --teacher logs/rsl_rl/so101_vial_state/2026-10-06_18-32-51_sim2real_teacher/model_1198.pt \
  --output outputs/sim2real_audit/student_start.pt

uv run --locked --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation --num_envs 1024 \
  --max_iterations 600 --seed 42 --run_name sim2real_student \
  --checkpoint outputs/sim2real_audit/student_start.pt preset=sim2real \
  'env.events.reset_from_dataset.params.phase_weights=[4,2,1,1,1,1,1,1]'
```

Run: `logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_18-43-26_sim2real_student`.
Starting/pre-grasp phases receive extra training weight; evaluation still starts every episode at home.
In parallel, launched a further 300-iteration teacher pass from `model_1198.pt` with the same
home-weighted sampling and unchanged physics ranges, run name `sim2real_teacher_home`.

Reference export debugging followed Isaac Lab's maintained LEAPP exporter. The default ONNX Dynamo
export wrote artifacts but failed validation: an exported `GatherND` used int32 indices. These failed
artifacts live under `outputs/leapp_reference` and must not be deployed. The supported `jit-trace`
backend passed LEAPP validation on five recorded examples; that reference package is under
`outputs/leapp_reference_jit`. It belongs to the OLD checkpoint and is not the final trained policy.

The final export will use:

```bash
uv run --locked --extra sim2real so101 leapp export --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --checkpoint <NEW_CHECKPOINT> \
  --export_save_path outputs/leapp_sim2real --export_method jit-trace \
  --disable_graph_visualization preset=sim2real
```

Keep the whole output folder: YAML, TorchScript model, initial-value safetensors and validation log.
The YAML contains the camera/state inputs, named joint targets, 30 Hz frequency and last-action feedback.
Export/play disables training sensor augmentation; preprocessing, actor normalization and action
postprocessing remain packaged in LEAPP.

## 9. Physical inference script — read-only verified; motion pending

Script: `scripts/so101_policy.py`. It uses LEAPP `InferenceManager`, the follower's saved calibration,
its stable USB port and the real wrist camera. There is no leader dependency for policy inference.
The script binds the LEAPP-declared tensor shapes, dtypes, semantic connections and named joint order.
Unknown/privileged inputs, missing or duplicate joint outputs, and nonfinite outputs are rejected.

```bash
uv run --locked --extra sim2real python scripts/so101_policy.py \
  --pipeline <EXPORTED_LEAPP_YAML> --duration 10
```

Default mode only opens the motor bus for reads. It does not configure motors, enable/disable torque,
write calibration, or send goals. Read-only tests on the actual follower/camera completed 80 steps in
three seconds. For the assistant's stale group session, the same command was wrapped with `sg dialout -c`.
The stricter command checker reports that the current elbow position is outside the saved calibration
sweep's usable envelope. It continues prediction in dry mode but will refuse live commands there.

Joint map example: `config/so101_joint_map.example.json`; deliberately `verified: false`.
Each joint uses `simulation_radians = scale * lerobot_value + offset`. Arm LeRobot values are degrees;
gripper values are 0–100. The gripper uses this follower's 1449-tick span (127.3846°) and the workshop fitted zero
−12.50575°. Every axis sign, reference pose, endpoint and mapping must be verified against this arm.
Do not mark the example verified merely to bypass the check.

For a later supervised trial, use a separate measured mapping file and add `--execute --joint-map <FILE>`.
Live mode checks the complete policy before motor writes, seeds goals from measured positions before
configuration enables torque, and sends only finite position targets intersecting simulation limits,
the saved calibration envelope, and a five-degree/five-percentage-point relative cap. A control step over
200 ms stops command delivery. Support the arm during startup/shutdown; live cleanup disables torque.
This script does not implement collision avoidance or task-success perception.

The export also contains Newton stiffness/damping outputs in SI units. Those are checked for finiteness
but not copied into Feetech's unitless PID registers. Hardware uses the same LeRobot PID configuration
as the successful teleoperation session. Joint velocity is estimated from successive measured positions
and timestamps, bounded consistently with the trained observation contract.

Validation so far: 65 tests pass; Ruff and whitespace checks pass; camera capture succeeds;
reference TorchScript export validates; real-camera/follower dry inference succeeds. Final training,
canonical-home evaluation and export validation are still pending.


### Additional verification notes

A 128-home-pose interim evaluation of `model_5100.pt` with `--deterministic` failed before an outcome
could be recorded: Warp deterministic collision recording overflowed its counter buffer. The follow-up
uses `model_5200.pt`, seed 42 and normal Newton execution. RSL-RL already uses the actor mean in
playback; omitting this flag retains deterministic policy actions but does not claim bitwise-deterministic
physics. Current README evaluation commands were corrected accordingly.

The policy bridge now checks the export's 30 Hz frequency and presence of the wrist RGB input,
intersects the simulation range with the saved calibration sweep and relative motor cap, and validates
the first policy output before any live configuration writes. Dry inference reports unreachable commands;
live inference rejects them. The example map and current physical setup remain unverified for autonomy.

## 10. Broad workspace and workshop modeling corrections (2026-10-06)

The user requested broad independent placement, retained the wrist camera only, and then authorized
restricting initial vial samples to the wrist-camera view or adjusting the home pose. The broad candidate
workspace is an area-uniform annular sector 0.16–0.38 m from the robot base, −100° to +100° about world +X.
Rack and horizontal vial headings span 360°. Reset rejection checks desk support, vial/rack separation,
base clearance and conservative link envelopes. Held replay phases preserve the vial/robot grasp;
insertion/release replay remains aligned to its opening. These bounds are candidates, not an exhaustive
IK reachability certificate. The desk support was enlarged to approximately 0.914 × 1.006 m.

Before visibility filtering, only 41/128 vial grasp points and 49/128 rack openings projected into the
initial wrist frame. The earlier narrow-trained teacher scored 0/128 on these broad starts.
A separate 60-step zero-action support check kept all 128 vials on the desk, with root heights
0.0508206–0.0508264 m. Thus the old teacher's object losses were not explained by missing table support.
The narrow teacher/student continuation runs were stopped; their results do not validate broad placement.
Visibility filtering and a measured overview home pose are being implemented before broad retraining.

### Source of measured corrections

The local reference checkout is `../sim-to-real-so-101-workshop`, inspected at commit `97a706b`.
The private GitLab MR URLs could not be fetched, but the local source contains the corrections:

- `source/sim_to_real_so101/assets/usd/so101_workshop_overrides.usda`: elbow frame quaternion,
  +6.4° coordinate shift, corrected limits and gripper limits.
- `source/sim_to_real_so101/assets/usd/Vial_three_primitive_collision.usda`: 28.9 mm body,
  35.4 mm cap, 12.77 mm cap height, cylindrical body plus spherical bottom and cylindrical cap.
- `source/sim_to_real_so101/utils/gripper_calibration.py`: shared 1485-tick average across ten arms
  and fitted jaw zero −12.505751884817162°.
- `source/sim_to_real_so101/tasks/vials_to_rack_env_cfg.py`: μ=0.7, vial contact dimension 6,
  rolling/torsional friction 0.0005 and 0.06 m free-vial spawn height.

The tutorial applies the elbow frame, limits and target together. Bundled reset joint positions and
joint targets were shifted +6.4° to preserve FK; the original artifact is saved at
`outputs/sim2real_audit/reset_poses_before_elbow_fix.pt`. This is a coordinate conversion, **not** a new
physical validation of all held reset states after changing vial contacts. That validation remains pending.

Vial spawn overrides correct collider **and render** dimensions without shortening the body. The bottom
sphere and shorter cap use the reference geometry. Under `preset=sim2real`, body diameter spans
28.4–29.4 mm and cap diameter 34.9–35.9 mm, independently sampled per Newton world during model
construction. Geometry is finalized before solver/collision/render initialization; it is not changed live.
Inertia variation remains independent of the small diameter variation.

The first USD-event approach failed when physics replication was disabled: Newton could not locate the
robot articulation. It was replaced with changes to the cloned Newton builder before finalization,
preserving physics replication. No upstream environment files were patched for this workaround.

Our actual follower gripper calibration spans 3496−2047=1449 ticks, or **127.3846°** using LeRobot's
4095 tick denominator. Therefore neither the old 110° span nor the reported 134.6° span is appropriate
for this saved calibration. The example map uses this measured scale and the workshop fitted zero,
but remains `verified: false`: zero, direction and jaw width need checking on this physical arm.
Simulation gripper limits were expanded to the same provisional angular interval. Existing radian grasp
waypoints were not rescaled, since they describe jaw geometry rather than calibrated percentages.

Rolling/torsional friction defaults are 0.0005; the old tutorial values were 0.05/0.005.
Six-dimensional contact is explicitly enabled so the solver can use both resistances. Free broad-workspace
vials spawn at 0.06 m (about 7–8 mm clearance over the desk); held replay poses retain their measured height.
Small rolling friction is not assumed to suppress rolling: displacement must be measured under the
chosen collider geometry, gravity perturbation and drop height.

Long training is paused until geometry/reset, camera visibility and passive-contact checks pass.
The earlier 108/128 (84.375%) teacher result applies to the old narrow-placement geometry only.

### Verified geometry, projection and command timing

- Corrected-geometry passive check: 128 vials, 150 control steps (5 s), zero policy action.
  Median XY movement 0.7688 mm; maximum 9.1239 mm; no vial below desk. Newton's finalized model
  contained 128 unique body diameters (28.4026–29.3998 mm), matching bottom spheres, and 128 unique
  cap diameters (34.9029–35.8837 mm). Contact dimension was 6. Log:
  `/tmp/so101_geometry_support.log`. These numbers describe this test, not a guarantee that all real
  vials stop rolling; small rolling friction alone does not prevent rolling.
- Camera home-pose sweep: 126 candidates from one-joint sweeps and a shoulder-lift/wrist grid.
  Baseline optical-axis alignment with world down: 0.8687; camera height 0.2741 m.
  Best nearby overview candidate: height 0.2947 m, down alignment 0.8037; conservative point coverage
  increased only 26.5%→29.5%. Kept the established home pose. Results:
  `outputs/sim2real_audit/pose_sweep.json`.
- Initial vial sampling now rejects conservative bounding volumes outside the wrist image, with a
  two-pixel border. Independent rendered-camera projection audit found the vial grasp point in-frame
  for **128/128** starts; rack opening in-frame for 26/128 (20.3125%). This checks image bounds,
  not a segmentation-based occlusion guarantee. Contact sheet:
  `outputs/camera/workspace_start_views.png`.
- The USD visual frame's relative camera rotation disagreed with the actual Newton camera-site pose.
  An initial filter using that visual rotation yielded only 90/128 visible grasp points. The filter was
  corrected to the measured pinned Newton site transform: position
  (−0.00188633, 0.05226452, −0.05853188) m relative to the gripper and ROS XYZW quaternion
  (0.95371729, 0, 0, 0.30070585). Intrinsics at 64×48: fx=fy=41.5366287, cx=32, cy=24.
  Any future camera mount/asset/intrinsics change requires rerunning this measurement and audit.
- Visibility rejection applies to actual home/phase-0 placement samples. Pregrasp/grasp replay keeps
  the original vial pose, since moving the vial independently invalidates its close approach and
  can leave no visible collision-free start (35/1024 resets failed the first attempt).
  Rack placement remains broad for phases before insertion.
- `ControlRateRelativeJointPositionAction` computes arm/gripper absolute goals from measured joint
  positions **once per 30 Hz control cycle** and holds those goals over all four physics steps.
  This matches LEAPP/hardware command cadence. The old actions recomputed goals from evolving joint
  positions during each physics step, changing their effective actuation. Existing trained policies
  therefore require fresh evaluation/retraining. Observation/action dimensions remain unchanged.

### Visual memory and current runs

The wrist-only actor/student now has a CNN followed by a 128-unit GRU and the policy head.
It uses existing RSL-RL hidden-state/reset APIs and supports recurrent PPO trajectory batches.
Distillation uses 16-step truncated backpropagation. A behavioral test verifies reset isolation and
agreement between native recurrent inference and TorchScript over a sequence. LEAPP supports explicit
actor-state feedback; its exported graph must include this feedback before deployment.
Memory retains previously observed information; it cannot determine an unseen rack's position.
Broad rack placement still requires the policy to learn camera searching and to encounter the rack.

Recurrent distillation smoke: 16 environments, two iterations, teacher weights loaded from
`outputs/sim2real_audit/wide_teacher_start.pt`; completed successfully in 3.32 s.
Artifact: `logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_19-17-07_visual_memory_smoke/model_1.pt`.
This is an architecture smoke artifact, not a trained deployment policy.
The first export attempt used a feedforward PPO template, causing an optimizer parameter-group
mismatch. Export preparation must use a PPO template with the same recurrent architecture.

Teacher initialization preserves the old network weights, shifts observation-normalizer means for
elbow joint position/target by 6.4°, reduces the normalization prior count from 242,679,808 to 65,536
so the broader domain can adapt, clears optimizer history and starts the iteration counter at zero.
The earlier measured-geometry run was stopped after identifying command-cadence mismatch.
The active replacement command is:

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 1024 --max_iterations 1000 \
  --run_name sim2real_wide_measured_control30 \
  --checkpoint "$PWD/outputs/sim2real_audit/wide_teacher_start.pt" \
  preset=sim2real agent.resume=True \
  env.events.reset_from_dataset.params.phase_weights='[4,2,1,1,1,1,1,1]'
```

Current develop accepts `agent.resume=True`; it does not accept the old `--resume` CLI flag.
An explicit `--checkpoint` loads its weights even without `agent.resume=True`, as confirmed in the
pinned training entrypoint. All 70 tests at that stage and Ruff passed.
The broad-domain teacher acceptance audit, final visual training and final LEAPP export remain pending.

Visual PPO trajectory smoke also passed: 16 worlds, two iterations, 4.61 s. Architecture-matching
export template: `logs/rsl_rl/so101_vial_camera/2026-10-06_19-19-13_visual_memory_ppo_smoke/model_1.pt`.
The 30 Hz target-hold change is in the policy action terms; the dataset generator's existing action
remains unchanged. Future reset generation must be physically revalidated under the corrected geometry.
Material variation samples brightness and modest white balance around the default orange/yellow/brown
colors; independent channel extremes were producing unrealistic green/purple desks and were tightened.

LEAPP recurrent/30 Hz action smoke export passed all five numerical validation examples.
Folder: `outputs/leapp_memory_smoke_v2/IsaacTutorial-Place-Vial-SO101-Camera/`.
The YAML declares exactly the real wrist RGB and joint position/velocity/target inputs, plus internal
feedback for `actor_state` (1×1×128) and `last_action`. Camera RGBA's unused alias generates a warning;
RGB remains a declared, used input. This artifact verifies deployment wiring, not task performance.

The first current teacher checkpoint (`model_50.pt`) was evaluated on 128 phase-0 starts with unseen
seed 43: success 0/128, grasp 3/128, lift/insertion 0, all timeouts, no lost vials or unsafe rack impacts.
This is an early optimization checkpoint. Training curriculum averages include late-phase starts and
must not be interpreted as a home-start success rate. Continue training and audit later checkpoints.
Log: `/tmp/so101_wide_teacher_eval50.log`.

A fresh state-teacher ablation (400 iterations, same physics, reset curriculum, observations and seed)
was launched as `sim2real_wide_fresh_control30` to compare against the warm start. It changes only
initial weights/normalization history. Evaluate home-start success on the same unseen seed before
choosing a continuation. Training logs: `/tmp/so101_wide_teacher_control30.log` and
`/tmp/so101_wide_teacher_fresh.log`. No final task-success or physical rollout claim yet.

Corrected-geometry replay check: 1024 worlds, 128 rows per phase, placement randomization disabled
for this diagnostic, one second of zero policy action under randomized physics. Phases 3–6 retained
bilateral jaw contact in 128/128 worlds each. Phase 2 retained contact in 123/128; phase 1 in 105/128.
These approach phases are not claimed to be settled load-bearing grasps. Phase 0 settled approximately
1.4 mm as expected from the larger cap. Late release worlds can finish and reset during this diagnostic,
so their final displacement is not used as a retained-state validity claim.
Log: `/tmp/so101_reset_geometry_audit.log`. This contact check does not replace a complete generator
acceptance audit under the new geometry and timing.

A control-cadence runtime check confirmed **four physics target writes with exactly zero target change**
within one control interval, while measured joints moved 0.0007169 rad. The target-hold behavior is
therefore active in simulation rather than inferred only from code. Log: `/tmp/so101_control_hold_audit.log`.
Current placement plot: `outputs/sim2real_audit/workspace_visible_coverage.png` (also SVG).
This plot distinguishes the broad rack sector from the camera-conditioned initial vial region.

### Actuator authority correction and final training contract

The measured-relative-goal training runs were stopped. A finalized-model diagnostic found a randomized
elbow stiffness of 11.6109 Nm/rad and joint friction of 0.5338 Nm. A maximum 0.033 rad target error
produces only 0.3832 Nm spring torque, insufficient to overcome that friction before even considering
load torque. Maintaining the same small offset from measured position prevents accumulation of
servo error and makes some randomized worlds uncontrollable in one direction.

The policy therefore increments the **previous commanded goal**, rather than measured position,
by up to 0.033 rad (arm) / 0.02 rad (gripper) per control cycle. Goals remain inside joint limits and
within 5° of measured arm positions or 5% of this follower's calibrated gripper span (6.3692°).
The goal is held over the four physics steps. Hardware uses the same absolute goal and tracking cap;
its actually sent goal is supplied as the next `joint_target` observation/LEAPP input. This provides
bounded friction-overcoming authority and an observable command state.
The old measured-relative action artifacts are not final candidates.

`preset=sim2real` now allows 40 s per episode for broad transport and wrist searching;
the baseline remains 20 s. The visual sim2real task starts every training episode at home, preserving
uninterrupted observation history rather than asking memory to infer a rack from a blind late-phase reset.
The state teacher retains phase-balanced replay. Current bounded-goal runs:

- Warm teacher: 2048 worlds, 1500 iterations, `sim2real_bounded_goal_teacher`.
- Fresh-initialization diagnostic: 512 worlds, 300 iterations, `sim2real_bounded_goal_fresh`.

These runs use different batch sizes and are not a strict same-step-budget initialization ablation;
compare home success and total collected transitions, not iteration numbers alone.
Logs: `/tmp/so101_bounded_teacher.log`, `/tmp/so101_bounded_fresh.log`.
Actuator diagnostic: `/tmp/so101_actuator_audit.log`. The early zero-success evaluations belong to the
superseded action contract and must not be attributed to the bounded accumulated-goal runs.

### Rack symmetry and observable teacher target

With unrestricted rack rotation, a visually uniform four-hole rack cannot reliably label one simulator-designated
opening. An optional user preference was requested (any empty opening versus a marked specific opening).
After allowing time for a response, continued with **any empty opening**, matching ordinary vial-to-rack placement.
All four openings are assumed empty at the start; this task has one vial and does not model occupied holes.
The measured opening centers are (0,0), (0.06,0), (0,0.06), (0.06,0.06) m in the authored rack frame.
Reward shaping, insertion and stable-release success use the nearest opening. Generator rack-overflight
checks still use the full rack's bounds, with opening clearance checked against the nearest opening.
A behavioral test confirms acceptance in all four openings after rotating/translating the rack and rejects
the solid central area between them.

The teacher's three-component target error is now rotated back into **fixed world axes**. The old rack-local
vector changed direction with rack yaw, while the state actor had no rack orientation observation.
That was an unobserved variable once full rack rotation was enabled. Tensor dimensions remain unchanged;
existing baseline checkpoints are still architecture-compatible, but broad-domain results require retraining.

The bounded-goal/single-hole runs were stopped before final policy selection. Current corrected-task runs:
`sim2real_any_hole_teacher` (2048 worlds, warm start, 1500 iterations) and
`sim2real_any_hole_fresh` (2048 worlds, fresh initialization, 800 iterations).
Logs: `/tmp/so101_any_hole_teacher.log`, `/tmp/so101_any_hole_fresh.log`.
The earlier runs are diagnostic history, not final candidates.

Recurrent playback's TorchScript and ONNX exports also completed successfully.
The LEAPP feedback initial values for both GRU state and previous action were checked to be exactly zero.
A six-control-step actuator check with positive bounded increments moved the elbow 0.06887 rad in 0.2 s,
while goals accumulated and then respected the tracking cap. Log: `/tmp/so101_goal_accumulation_audit.log`.
Diagnostic scripts/logs have been archived under `outputs/sim2real_audit/diagnostics/`.

### Current-contract acceptance checks

The corrected-task warm run is `logs/rsl_rl/so101_vial_state/2026-10-06_19-44-04_sim2real_any_hole_teacher/`;
the fresh run is `logs/rsl_rl/so101_vial_state/2026-10-06_19-48-18_sim2real_any_hole_fresh/`.
Both use 2048 worlds, the same seed, 40 s episodes, full sim2real physics and broad placement,
with replay phase weights `[4,2,1,1,1,1,1,1]`. Warm/fresh initialization is now compared with equal batch sizes.
The first warm checkpoint audit (`model_150.pt`, 128 home starts, unseen seed 43) completed 0/128:
grasp 1/128, no lifts/insertions, one lost vial, no unsafe rack contacts. This is an early failed acceptance
check, not a deployable result. A trajectory diagnostic measures approach distance and clipped-action
saturation while optimization continues, to distinguish a controller problem from missing learned behavior.
Logs: `/tmp/so101_any_hole_teacher_eval150.log`, `/tmp/so101_any_hole_approach200.log`.

The full current-contract test suite passes: **72 tests** (`/tmp/so101_final_contract_tests.log`).
The provisional hardware joint-map example now uses the simulator's 0.98-factor soft limits for every joint,
not wider rounded hard limits. It remains `verified: false`; physical joint signs and zero offsets still
need measured confirmation before the first powered policy experiment.

The warm teacher was stopped after saving `model_400.pt`: a second audit at `model_350.pt`
again had no grasps or successes (128/128 timeouts). Fresh initialization was retained.
Fresh `model_150.pt`: 3/128 grasps, 1/128 lift, no placements, no unsafe rack contacts.
Fresh `model_250.pt`: **2/128 stable placements (1.5625%)**, 8/128 grasps, 3/128 lifts and insertions,
one lost vial, no unsafe rack contacts; successful episodes averaged 24.83 s.
Logs: `/tmp/so101_any_hole_fresh_eval150.log`, `/tmp/so101_any_hole_fresh_eval250.log`.
This is an early learning result and remains below acceptance for physical experiments.

The develop update moved Newton IK to `isaaclab_newton.controllers.ik` and clone-source lookup to
`cloner.path.get_asset_prototypes` / `get_asset_prototype_paths`. The reset generator now uses those
current APIs; a real 128-world IK launch verifies the repaired path.
A geometric reachability diagnostic found 78/128 nominal downward-grasp solutions, 93/128 when both
cylinder-symmetric wrist branches were tried. Increasing from 64 to 256 seeds did not change that result.
Allowing angled grasp orientations increased it to 126/128. This is a search diagnostic, not proof of
physical impossibility for the other starts. The broad placement sampler was kept unchanged.

A further simulated pregrasp/closing check used the measured cap resting height (~52.7 mm),
then 8 hold steps and 22 bounded closing steps under full randomized physics. It produced bilateral
jaw contact in 125/128 worlds. Of 126 IK-valid candidates, 109 cleared the mat using a conservative
7.5 mm contact-pad half-height approximation; the selected candidates are therefore **not all validated
for safe table clearance**, and bilateral contact alone does not establish a lifted grasp.
These diagnostics must not be reported as a policy success rate or a physical-robot validation.
Logs: `/tmp/so101_reachability_both.log`, `/tmp/so101_reachability_dense.log`,
`/tmp/so101_reachability_tilted.log`, `/tmp/so101_grasp_clearance.log`.

### Longer-horizon comparison and camera background variation

Successful broad-layout episodes already take about 25 s. At 30 Hz, `gamma=0.995` has an effective
reward horizon of about 6.7 s and discounts a 25 s completion by roughly 0.023. A controlled continuation
from fresh `model_300.pt` compares `agent.algorithm.gamma=0.999` (about 33.3 s horizon) with the unchanged
fresh run. Both use 2048 worlds and the same reset/physics contract. This is an optimization comparison;
the environment and deployment action contract are unchanged. Run: `sim2real_long_horizon`, 500
additional iterations; log `/tmp/so101_long_horizon_teacher.log`.

The simulated brown tabletop is visually plain, unlike the real wood grain. Sim2real camera training now
adds episode-sampled directional grain and smooth mottling (0–20% multiplicative intensity) to brown
background pixels. The mask uses the native RGB ratios before photometric changes and excludes the white
vial, blue cap, yellow rack, orange robot and dark servos. The image-plane augmentation approximates
background appearance; it is not a physically rendered world-space wood texture and does not simulate
new object occlusions. It is disabled for play/export along with the other sensor augmentations.
A behavioral palette test protects task-object colors. A real 16-world, two-iteration camera distillation
smoke passed (4.46 s); the full suite passes **73 tests**. Logs: `/tmp/so101_texture_smoke.log`,
`/tmp/so101_texture_tests.log`.

Refreshed post-color-correction wrist views: `outputs/camera/workspace_start_views.png`.
The 128-world seed-43 projection check again puts all vial grasp points in frame. This verifies projection,
not an occlusion guarantee; the reset sampler itself requires the conservative vial volume inside the image.
Log: `/tmp/so101_final_workspace_views.log`.

### Grasp alignment learning correction

Fresh `model_400.pt` home audit: 13/128 grasps, 7/128 lifts, 1/128 placement, 3 lost vials,
and one rack impact over 20 N (peak 22.94 N). Longer-horizon `model_400.pt`: 6/128 grasps,
4/128 lifts, 3/128 placements (2.34375%), no lost vials or unsafe rack contacts, peak 4.91 N.
Logs: `/tmp/so101_any_hole_fresh_eval400.log`, `/tmp/so101_long_horizon_eval400.log`.
These small audits do not establish a statistically reliable preference yet. The original fresh run
was stopped after `model_450.pt`; the longer-horizon run remains the comparison.

The main failure is now grasp acquisition under broad vial rotation. A jaw-midpoint-only approach
potential provides no guidance for pad/vial alignment and can reward approaching with closed jaws.
The approach potential now adds 30 mm-equivalent pad-axis error, 20 mm-equivalent closing-axis error,
and insufficient jaw clearance relative to the maximum cap diameter plus 8 mm. Clearance weighting
fades over a 15 mm distance scale as the midpoint reaches the cap, so closure at the cap is permitted.
The vial axis is unsigned: either pad direction is accepted. Reward remains a bounded difference of
potentials and stops once grasped; no continuous hovering bonus was introduced.
Physics, reset geometry, success criteria, action bounds and deployable observations are unchanged.

A continuation `sim2real_grasp_pose` starts from the longer-horizon `model_400.pt`, with the same
2048 worlds, `gamma=0.999`, reset weights and 600 additional iterations. The already-running longer-horizon
process retains its old distance-only reward, providing a comparison against the new pose potential.
Log: `/tmp/so101_grasp_pose_teacher.log`. Unit behavior protects axis symmetry, opening during approach,
closure at the cap and reset/grasp reward gating. Full suite: **74 passing tests**; focused post-rename
suite: 14 passing. A real 16-world, two-iteration teacher smoke passed (6.32 s).

At checkpoint 500, the old distance-only/long-horizon policy placed 4/128 (3.125%), with 17 grasps,
12 lifts, 6 insertions, 3 lost vials and no unsafe rack contacts (peak 9.80 N).
The grasp-pose continuation placed 5/128 (3.90625%), with 14 grasps, 9 lifts and 5 insertions,
one lost vial and no unsafe rack contacts (peak 1.64 N). Small differences are not yet a reliable
statistical improvement. Logs: `/tmp/so101_long_horizon_eval500.log`, `/tmp/so101_grasp_pose_eval500.log`.

The optional `--encoder-source` on `scripts/distillation_checkpoint.py warm-start` copies only an exactly
matching CNN, preserving the current student's GRU, controller and proprioceptive normalization.
Mismatched keys/shapes are rejected. Prepared `outputs/sim2real_audit/visual_encoder_bootstrap.pt`
from the recurrent smoke student, current long-horizon teacher400, and the earlier pretrained camera CNN
in `student_start.pt`. Optimizer momentum is cleared and provenance recorded. This is initialization,
not an accepted trained camera policy. Final teacher replacement will retain the prepared student.

A curriculum diagnostic is collecting broad-layout pregrasp/closed-contact states from the actual simulator.
Candidate IK states must satisfy joint margins, conservative pad/mat clearance and ≤12 mm cap-centering
error. Closing uses the ordinary bounded 30 Hz action; an extra 30-step zero-action hold checks persistent
bilateral contact. Only qualified rows are candidates for a separate trial dataset. Home evaluation remains
on the broad randomized workspace; this does not narrow its sampling or count expert resets as policy success.

Grasp-pose checkpoint600 home audit: **18/128 placements (14.0625%)**, 44/128 grasps,
30/128 lifts, 19/128 insertions, 7 lost vials, no unsafe rack contacts; peak rack force 11.28 N,
mean successful completion 13.82 s. This is a substantial improvement over checkpoint500 but remains
below useful acceptance. Log: `/tmp/so101_grasp_pose_eval600.log`.

### Broad pregrasp curriculum trial

Training candidate collection uses seeds **142/143**, keeping audit seeds 43/44 out of its training data.
Pools: 213 qualified pregrasps and 220 persistent closed-contact states. The trial selects 128 per phase,
replacing only phases1/2 in a copy of the corrected 1024-row dataset. Phase0 home rows and phases3–7 remain
unchanged. Schema validation and content digest pass. Trial artifact:
`outputs/sim2real_audit/reset_poses_wide_grasps.pt`, SHA-256 content digest
`f4c8233f2a47b2efc00ac3a7f75500d115bca10c8bfe82d13144f0f1920e343f`.
The repository's default reset asset is unchanged by this trial. Diagnostic 43/44 candidates were not used.
Reproducers: `/tmp/so101_collect_grasps.py`, `/tmp/so101_assemble_wide_resets.py`, archived under
`outputs/sim2real_audit/diagnostics/`. Training qualification includes ordinary bounded closing and
30 consecutive contact-retaining control steps, ≤12 mm centering error, 36 mm minimum conservative pad
lower height, finite joints and soft-limit checks. Both flags remain unseeded for these phases so the
policy must physically prove its grasp/lift after reset.

Independent 1024-world replays select phases1/2, preserve broad rack placement, randomize the physics
again, and run 30 zero-action control steps. At seed144, closed-grasp contact persisted in 512/512,
median vial motion 1.18 mm, maximum 110.61 mm; some large shifts require caution. At seed145,
closed-grasp contact again persisted in 512/512: median1.26 mm, p95=10.19 mm, p99=18.01 mm,
maximum19.15 mm, 94.73% within10 mm. Pregrasp median motion0.63 mm, p95=4.61 mm,
97.85% within10 mm. These results qualify a **training comparison**, not a guarantee that every
randomized reset remains stationary or a regenerated validation of the other six phases.
The original metadata's corrected-geometry revalidation-pending flag is retained.
Logs: `/tmp/so101_wide_reset_replay.log`, `/tmp/so101_wide_reset_replay145.log`;
summary: `outputs/sim2real_audit/wide_reset_replay.json`.

Long-horizon/distance-only checkpoint650 audit: 13/128 placements (10.15625%), 64 grasps,
43 lifts, 15 insertions, 5 lost vials and one unsafe rack contact (peak24.55 N).
That comparison was stopped after saving checkpoint700. The current pose-reward baseline continues
and the wide-grasp trial starts from pose600 with the same 2048 worlds, gamma0.999, phase weights,
and 600 additional iterations. Run: `sim2real_wide_grasp_curriculum`; log
`/tmp/so101_wide_grasp_curriculum.log`. Only the trial reset dataset changes between these recipes.
The prepared camera bootstrap's teacher was updated to pose600 in
`outputs/sim2real_audit/visual_teacher600_start.pt`; visual training acceptance is still pending.

### Robot self-collision audit

The pinned upstream SO101 asset disables self-collision in both Newton and PhysX. A paired seed43
128-home-start audit of pose700 gave 25/128 placements (19.53125%) with the old setting, versus
26/128 (20.3125%) with Newton self-collision enabled. The one-episode difference is not evidence of
superior performance. Enabled case: 57 grasps,45 lifts,29 insertions,6 lost vials, no unsafe rack contacts,
peak17.50 N. Disabled case:63 grasps,52 lifts,29 insertions,9 lost vials,one unsafe rack contact,
peak20.65 N. Logs: `/tmp/so101_grasp_pose_eval700.log`, `/tmp/so101_self_collision_eval700.log`.

The tutorial's copied robot spawn configuration now enables self-contact in both backend schema fragments,
retaining upstream solver settings and leaving the upstream global asset configuration intact. This is a
standard physical setting, not domain randomization. Already-running teachers retain their constructed
old collision model; subsequent jobs, play and export use the new default. Final candidates must be
accepted under enabled self-collision, and the selected teacher should be continued with that setting.
The replay under enabled self-collision retained contact in 512/512 closed starts: median1.22 mm
motion, p95=10.20 mm, maximum19.20 mm, 94.73% within10 mm. This is a reset check, not policy success.


### Current teacher and recurrent visual bootstrap

Pose-reward checkpoint800, current self-collision enabled, completed **30/128 placements
(23.4375%)** at seed43: 68 grasps,61 lifts,38 insertions,7 lost vials,3 unsafe rack contacts,
peak25.10 N and mean successful completion14.20 s. Improving, but not accepted for deployment.
Log: `/tmp/so101_grasp_pose_eval800.log`.

State PPO now defaults to gamma **0.999**, matching current experiments: effective discounted
horizon about33 seconds at30 Hz, versus about6.7 seconds with0.995 in the40-second episode.
This changes training, not policy architecture or deployment inputs.

Started512-world,300-iteration recurrent wrist-camera bootstrap, seed42, current self-collision,
full preset, home-only starts and camera/background augmentation. Initialization preserves the
matching CNN and current GRU architecture, replacing the teacher with pose800. The teacher
still has limited success; the student must be updated and independently evaluated. Loading
weights or decreasing imitation loss is not evidence of placement success.

```bash
uv run --no-sync --extra sim2real python scripts/distillation_checkpoint.py warm-start \
  --student outputs/sim2real_audit/visual_encoder_bootstrap.pt \
  --teacher logs/rsl_rl/so101_vial_state/2026-10-06_20-19-32_sim2real_grasp_pose/model_800.pt \
  --output outputs/sim2real_audit/visual_teacher800_start.pt

uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation --num_envs 512 \
  --max_iterations 300 --seed 42 --run_name sim2real_visual_bootstrap \
  --checkpoint outputs/sim2real_audit/visual_teacher800_start.pt preset=sim2real
```

Log: `/tmp/so101_visual_bootstrap.log`. Ongoing teachers were instantiated before self-collision
was enabled; final continuation and acceptance use the enabled default.


The broad-grasp curriculum trial home audits were26/128 at checkpoint700 and25/128
at800, versus30/128 for the unchanged curriculum at800. Both used enabled self-collision
and seed43. Trial800:77 grasps,71 lifts,37 insertions,8 lost vials,zero unsafe contacts,
peak13.96 N. This small comparison does not establish superiority. The trial was stopped
after preserving checkpoint800 so compute can focus on the stronger baseline and visual
learning. The trial reset dataset remains separate; it was not promoted to the default.
Logs: `/tmp/so101_wide_grasp_eval700.log`, `/tmp/so101_wide_grasp_eval800.log`.


Pose checkpoint900 home audit, seed43/current self-collision: **58/128 placements (45.3125%)**,
87 grasps,75 lifts,61 insertions,3 lost vials and1 unsafe rack contact; maximum22.59 N,
mean successful completion13.07 s. Log: `/tmp/so101_grasp_pose_eval900.log`. This is
a stronger teacher, but still not the final camera-policy acceptance result.


Hard vial/rack impacts were tracked but did not contribute any negative training reward.
Recent audits measured1–3 unsafe episodes per128 starts. Added the existing tested
`unsafe_rack_contact` predicate as a reward at weight−200, using its20 N threshold so normal
insertion guidance is allowed. At30 Hz one flagged step costs6.67 reward units, matching
the one-step200-weight success payout. Multiple flagged steps cost more. This is a reward
change for subsequent training, not a claimed hardware-safe force limit or a new termination.
Already-running jobs retain their original reward configuration. The continuation will also
use enabled self-collision; its results cannot isolate either change causally.


Visual bootstrap checkpoint100 audit:0/128 placements,38 grasps,19 lifts,1 insertion,
2 lost vials and1 unsafe rack contact (peak23.11 N). This is an initial recurrent
visual learning diagnostic, not an accepted deployment result. Log:
`/tmp/so101_visual_bootstrap_eval100.log`. Reward/config verification after adding
the impact penalty:14 relevant tests passed,35 warnings,3.76 s. The reference
pregrasp/grasp/release constants now explicitly state their unchanged angles
(14.64°,−8.9°,36.97°), avoiding the obsolete110° normalized-span comment.


Visual bootstrap finished300 iterations in590.99 seconds, saving
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_21-01-17_sim2real_visual_bootstrap/model_299.pt`.
Checkpoint200 audit:0/128 placements,45 grasps,29 lifts,0 insertions,6 lost vials,
no unsafe rack contacts, peak7.62 N. Final bootstrap evaluation is pending.
Logs and standalone audit/collection scripts are copied to
`outputs/sim2real_audit/diagnostics/`; the current evaluation/contract manifest is
`outputs/sim2real_audit/current_contract_results.json`. This records failures as well as successes.


Pose950:64/128 home placements (50%),100 grasps,89 lifts,68 insertions,4 lost
vials,2 unsafe rack-contact episodes, peak39.26 N, successful completion mean11.66 s.
Final visual bootstrap299:0/128 placements,42 grasps,30 lifts,1 insertion,3 lost
vials,no unsafe contacts,peak14.79 N. Logs: `/tmp/so101_grasp_pose_eval950.log`,
`/tmp/so101_visual_bootstrap_eval299.log`.

Started a1024-world,600-iteration distillation continuation (`sim2real_visual_teacher950`)
using the preserved bootstrap299 student and stronger pose950 teacher, seed42/full preset.
Prepared `outputs/sim2real_audit/visual_teacher950_start.pt` with the warm-start helper;
optimizer state is cleared and iteration reset, while CNN/GRU/MLP/normalizer weights remain.
Log: `/tmp/so101_visual_teacher950.log`. This continuation is learning, not acceptance.


The pose baseline finished its600 additional iterations in3269.42 seconds, final
`2026-10-06_20-19-32_sim2real_grasp_pose/model_999.pt`. Started the selected continuation
with4096 worlds,1500 additional iterations,seed42,full preset,enabled self-collision,
impact penalty and the unchanged home-weighted curriculum. This retains checkpoint
optimizer/actor/critic state and uses the current gamma0.999 default.

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --max_iterations 1500 \
  --seed 42 --run_name sim2real_self_collision_safe \
  --checkpoint logs/rsl_rl/so101_vial_state/2026-10-06_20-19-32_sim2real_grasp_pose/model_999.pt \
  preset=sim2real agent.resume=True \
  'env.events.reset_from_dataset.params.phase_weights=[4,2,1,1,1,1,1,1]'
```

Log: `/tmp/so101_self_collision_safe_teacher.log`. This name describes the training
configuration; it does not certify real-world safety. Final independent home-start
audits and camera-policy acceptance are still required.


Final baseline999 home audit:67/128 placements (52.34375%),101 grasps,89 lifts,
75 insertions,5 lost vials,5 unsafe rack-contact episodes, peak43.71 N and successful
completion mean13.43 s. Log: `/tmp/so101_grasp_pose_eval999.log`.

The4096-world safety continuation collected rollouts in8.7–8.8 seconds alongside camera
training, projecting several hours. Preserved `2026-10-06_21-14-20_sim2real_self_collision_safe/model_1000.pt`
and stopped it after seven updates. Resumed that checkpoint with1024 worlds,1500 additional
iterations, otherwise the same recipe, named `sim2real_safe_1024`; log
`/tmp/so101_safe_1024_teacher.log`. This reduces rollout batch size while preserving
the broad layout and physical randomization; it does not claim equal sample efficiency.


GPU scheduling diagnostic: with the1024-camera job active, the1024-state teacher
collected rollouts in4.83–4.90 seconds (5.00–5.07 seconds total iteration). Temporarily
paused the camera process withSIGSTOP after preserving checkpoint100, leaving its
training state intact. Teacher-only rollouts then took2.33–2.55 seconds (2.43–2.65
total), approximately halving update time. Teacher receives compute priority; the
camera job can resume withSIGCONT, or its saved student can be warmed with the next
strong teacher. No physical device is connected by either job. This is scheduling,
not a domain/randomization change.


Stronger-teacher visual checkpoint100 audit:0/128 placements,76 grasps (59.375%),
63 lifts (49.21875%),3 insertions,5 lost vials,zero unsafe contacts,peak16.73 N.
Log: `/tmp/so101_visual_teacher950_eval100.log`. This separates improved pickup
from the still-unresolved visual placement/search problem.

Prepared `outputs/sim2real_audit/visual_ppo_candidate100.pt` for reward-based visual
training. The student becomes the camera actor unchanged; the trained state999
critic replaces the smoke-template critic. Exact parameter names and shapes are
checked for both networks. Optimizer state is cleared and iteration reset. Actor
exploration std is0.15 for training, within the bounded0.05–0.3 range; deterministic
mean inference is unchanged. Reproducer:
`outputs/sim2real_audit/diagnostics/so101_prepare_visual_ppo.py`. A16-world/two-iteration
PPO compatibility check is running (`visual_ppo_warm_smoke`), not a policy acceptance audit.


### Visual reward training for the placement gap

The16-world visual PPO warm-start compatibility check passed both iterations in5.0 seconds.
The saved camera distillation100 student provides the actor; the newer enabled-collision
teacher1100 supplies the critic in `outputs/sim2real_audit/visual_ppo_teacher1100_start.pt`.
No privileged state enters the camera actor.

Started512-world visual PPO,1200 iterations,seed42,full preset/home-only starts,
32 control steps per rollout,3 learning epochs and4 minibatches. These overrides reduce
image-training overhead and are part of this recipe, not promoted defaults.
Paused the state teacher after checkpoint1100 and the camera distillation job after100
(they preserve their live state) to avoid the measured concurrent-job slowdown.

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 512 --max_iterations 1200 \
  --seed 42 --run_name sim2real_visual_ppo \
  --checkpoint outputs/sim2real_audit/visual_ppo_teacher1100_start.pt preset=sim2real \
  agent.num_steps_per_env=32 agent.algorithm.num_learning_epochs=3 \
  agent.algorithm.num_mini_batches=4
```

Log: `/tmp/so101_visual_ppo.log`. Actor exploration std starts0.15; deterministic
evaluation uses its mean. This run must demonstrate completed broad-workspace placements
from home, not just grasping, low imitation loss or successful expert resets.


Safety continuation teacher1100 home audit:66/128 placements (51.5625%),100
grasps,92 lifts,75 insertions,5 lost vials,**zero unsafe rack contacts**,peak14.90 N
and mean successful completion11.94 s. Log: `/tmp/so101_safe_1024_eval1100.log`.
Compared with baseline999 at the same audit seed, success is effectively unchanged
while observed impacts are lower; this one batch is not a safety guarantee or
an isolated causal test of the penalty. The teacher is paused for visual PPO.

Started a fresh LEAPP `jit-trace` compatibility export of the current recurrent
actor/control architecture into `outputs/leapp_current_control_smoke`. This checks
current goal accumulation, soft/tracking limits, feedback memory and image export
together. It is an unaccepted bootstrap policy, separate from the eventual
trained-policy export. Log: `/tmp/so101_current_control_export.log`.


### Export tracking-limit regression found and fixed

The current-control smoke export passed LEAPP's five numerical examples, but a
20-step bridge test on the saved focused camera image held measured joints fixed
and fed previous targets back. At step2 the shoulder-lift goal exceeded the5°
tracking bound (5.1726°). Inspection of the exported JIT graph showed goal
accumulation without its limit clamp. **Do not deploy `leapp_current_control_smoke`.**

Cause: the custom action ignored the return of an in-place `clamp_`; LEAPP traced
tensors did not retain that mutation. Changed to explicit assignment of
`self._control_target.clamp(lower, upper)`. Ordinary Torch physics/control values
are equivalent, so current training remains compatible. A fresh export goes to
`outputs/leapp_current_control_smoke_v2`; log `/tmp/so101_current_control_export_v2.log`.
The independent multi-step bridge contract is required in addition to native
five-example validation. Reproducer `/tmp/so101_export_contract.py` uses the saved
real camera image and synthetic joint inputs; it opens neither serial nor camera.
The hardware bridge also has its independent native-unit command bounds.


Visual PPO checkpoint100 home audit: **1/128 completed placements (0.78125%)**,
52 grasps,40 lifts,3 insertions,3 lost vials andzero unsafe contacts; peak11.95 N,
the successful episode completed in37.20 s. Log: `/tmp/so101_visual_ppo_eval100.log`.
This is the first completion by the current broad-workspace recurrent camera
policy, but its success rate remains far below useful acceptance.


Corrected current-control smoke_v2 passed native LEAPP validation on5/5 examples
and the expanded bridge contract on**60 steps across three input cases**: central
joints, near lower soft limits with an initially stale high goal, and near upper
limits with an initially stale low goal. Every absolute target was finite and
within both authored soft limits and the measured5°/5%-travel tracking envelope.
The check uses the saved focused camera image and synthetic joints, with no
serial/camera opened. Log: `/tmp/so101_current_export_contract_v2.log`.
The full74 tests passed after the clamp fix (39 warnings,4.11 s); Ruff check and
format checks pass. Smoke_v2 proves architecture/control compatibility, not
placement performance or physical joint alignment.


Visual PPO300:0/128 placements,65 grasps,54 lifts,0 insertions,1 lost vial,
no unsafe contacts,peak9.68 N. Checkpoint400:0/128 placements,71 grasps,52 lifts,
1 insertion,1 lost vial,no unsafe contacts,peak13.18 N. From the home rollout,
51 episodes turned a grasped vial above0.9 vertical alignment at least once.
Logs: `/tmp/so101_visual_ppo_eval300.log`, `/tmp/so101_visual_ppo_eval400_views.log`.

Captured16 native wrist views at steps0/150/300/600/900 in
`outputs/camera/visual_ppo400_step*.png`. Inspection shows grasped vials/caps taking
a large image area, with rack visibility varying. The first yellow-pixel
visibility diagnostic incorrectly classified brown desk pixels and reported
128/128 rack-seen; **those visibility counts are invalid** (the independently
computed physics success/uprightness counts are unaffected). Tightened the
diagnostic RGB ratio thresholds and started a new checkpoint500 view audit.
This classifier is only a diagnostic, never a policy input or production reward.


Checkpoint500 view audit:0/128 placements,101 grasps,84 lifts,0 insertions,
2 lost vials,no unsafe contacts,peak12.66 N. The corrected yellow-color diagnostic
estimates70/128 episodes ever see the rack and50/101 grasp episodes see it while
holding;77 episodes reach>0.9 vertical vial alignment while grasped. These
visibility counts are color-based estimates, not occlusion ground truth.
Log: `/tmp/so101_visual_ppo_eval500_views.log`; snapshots:
`outputs/camera/visual_ppo500_step*.png`.

Replayed every default reset phase under enabled self-collision and current
geometry/physical DR,1024 worlds (128 per phase),new seed146,30 zero-action
control steps. Automatic episode resets were disabled for this diagnostic so
release success cannot replace a sampled row during measurement. Phases2–6
retained bilateral jaw contact in128/128 for all30 steps. Median vial motions:
phase2=2.71 mm,3=4.49 mm,4=3.16 mm,5=5.12 mm,6=3.39 mm; maxima
5.87/19.21/8.50/15.82/9.86 mm. Released phase7 has16/128 residual contact,
median2.30 mm,max10.08 mm; this does not prove every release row is fully
disengaged. Phase0 settles from its spawn clearance (median10.57 mm).
Full metrics: `outputs/sim2real_audit/all_reset_replay.json`; log:
`/tmp/so101_all_reset_replay.log`. This qualifies a later-stage training
comparison; it is not complete revalidation of original generator metadata.


Visual PPO800 home audit:0/128 placements,**116 grasps (90.625%) and111 lifts
(86.71875%)**,3 insertions,6 lost vials,no unsafe contacts,peak10.85 N. Log:
`/tmp/so101_visual_ppo_eval800.log`. Strong pickup did not solve placement.
Preserved checkpoint1000 and stopped the home-only PPO run after1004 iterations
to focus on the unlearned later stages. No workspace or physical randomization
range was narrowed.

Started the placement-curriculum continuation from visual PPO1000,512 worlds,
1600 additional iterations,seed42,same32-step/3-epoch/4-minibatch recipe. Reset
weights `[4,0,0,1,1,1,2,0]` mix home,held/reorient/transport,and insertion starts;
release-only rows are excluded because16/128 still had residual contact in the
replay. The held/insertion phases all retained contact throughout that replay.
The saved CNN/GRU and optimizer are retained. Reward/physics/control stay fixed.
**Curriculum training success is not home-start policy success.** Independent
play audits always use the home-pose reset configuration without these overrides.

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 512 --max_iterations 1600 \
  --seed 42 --run_name sim2real_visual_placement_curriculum \
  --checkpoint logs/rsl_rl/so101_vial_camera/2026-10-06_21-21-55_sim2real_visual_ppo/model_1000.pt \
  preset=sim2real agent.num_steps_per_env=32 agent.algorithm.num_learning_epochs=3 \
  agent.algorithm.num_mini_batches=4 \
  'env.events.reset_from_dataset.params.phase_weights=[4,0,0,1,1,1,2,0]'
```

Log: `/tmp/so101_visual_placement_curriculum.log`. The defaults remain home-only
for camera training; these curriculum weights are an explicit experimental recipe.


Placement-curriculum PPO1300 home audit:0/128 placements,112 grasps,109 lifts,
0 insertions,11 lost vials,no unsafe contacts,peak14.24 N. The dedicated
phase6 action comparison found the student initially opens in128/128 insertion
starts, with mean normalized gripper command0.27–0.31 (teacher opens in128/128).
By30 control steps, insertion falls from97/128 to33/128 as the arm withdraws;
by60 steps only2/128 remain inserted and the student largely recloses. This
rules out the proposed initial all-close/saturated-head explanation in this
batch. The missing behavior is strong, sustained release while holding the
arm at insertion. Log: `/tmp/so101_release_action_audit.log`; report:
`outputs/sim2real_audit/release_action_audit.json`.

Preserved placement-PPO1500 and stopped that run to give the same camera actor
direct teacher labels in later-stage states. Prepared
`outputs/sim2real_audit/placement_student1500_teacher1100.pt` from PPO1500 actor,
state1100 teacher and the matching distillation template; exact keys/shapes
checked,optimizer cleared,iteration reset. Started512-world/800-iteration
DAgger with the same `[4,0,0,1,1,1,2,0]` curriculum,seed42,full preset.
Run `sim2real_placement_dagger`; log `/tmp/so101_placement_dagger.log`.
The actor retains its learned CNN/GRU/proprioception weights; the preparation
script is `/tmp/so101_prepare_placement_student.py`.


Rechecked the126-pose camera sweep with explicit scene-cache refresh after
kinematic writes; its results match the earlier sweep (26.5% sampled workspace
centers visible at the original home,29.5% at the best nearby grid pose).
A wider4096-pose kinematic search reached35.75% with a0.621-rad maximum joint
change; the random candidates within0.5 rad did not improve on baseline.
These are point-center/frustum estimates with a five-pixel margin, excluding
occlusion, not camera acceptance rates or a globally optimal pose proof.
There is no measured small home change that removes the need to search across
the requested broad workspace, so the original default remains. Artifacts:
`outputs/sim2real_audit/pose_sweep_refreshed.json`,
`outputs/sim2real_audit/global_pose_sweep.json`. No physical robot was moved.


Placement DAgger200 home audit:2/128 placements (1.5625%),83 grasps,66 lifts,
4 insertions,4 lost vials,no unsafe contacts,peak15.82 N andmean successful
completion31.03 s. Log: `/tmp/so101_placement_dagger_eval200.log`. The mixed
training reset success statistic is much higher, but is not reported as home
policy success. Later checkpoint audits remain necessary.


Placement DAgger completed800 iterations in766.19 seconds, final checkpoint
`2026-10-06_21-47-53_sim2real_placement_dagger/model_799.pt`. Checkpoint600 home
audit regressed to0/128 placements,65 grasps,55 lifts,3 insertions,8 lost vials,
no unsafe contacts,peak19.11 N. The low home scores remain explicit; no deployment
acceptance is inferred from mixed-phase training success.

A temporary training-only search expert (`/tmp/so101_search_expert.py`) holds
the arm/jaw goal and sweeps shoulder pan between±1.05 rad only after a current
confirmed lift clears the rack height, while the yellow rack has not been seen.
It then resumes the state teacher. It changes expert labels, not exported actor
inputs or hardware code. Teacher-controller home trial:50/128 placements
(39.0625%),104 grasps,90 lifts,54 insertions,6 lost vials,no unsafe contacts,
peak16.82 N.61 episodes scanned,62 observed the rack after lift; mean scan206
steps,max1121. This regresses versus ordinary teacher1100 (66/128), so it is
**not promoted or used to train a student**. Camera/actuator checks are pending
to explain scans that failed to discover the rack. Log:
`/tmp/so101_search_expert_teacher_eval.log`.


Motor-authority audit used128 full randomized worlds (exact-audit mode avoids
play's16-world cap). Every sampled motor's5°/5%-travel PD torque envelope
exceeded its dry-friction limit: minimum torque/friction ratios pan6.37,
shoulder6.01,elbow1.66,wrist7.47,roll11.19,gripper46.36. This only checks
dry friction, excluding gravity/contact loads; it does not certify dynamic
tracking. No actuator or DR range change follows from this check. Report:
`outputs/sim2real_audit/motor_authority.json`; log `/tmp/so101_authority_audit.log`.

Measured wrist axes via native forward kinematics instead of guessing from
the image: wrist roll is gripper-frame+Z; wrist flex is approximately
(−0.7204,+0.6935,0). Log: `/tmp/so101_wrist_axis_audit.log`.

A second temporary search-expert prototype first approaches the measured
nearby camera pose (shoulder−1.2 rad,wrist1.6 rad,other non-pan arm joints at
home),holding the jaw goal,then sweeps pan±1.75 rad. It requires16 native
yellow pixels and accepts darker pixels (R>50) so rack shadows do not block
discovery. It still requires current bilateral grasp and whole-vial rack
clearance. This is simulation-only validation; task defaults and hardware
code remain unchanged.

The second search expert completed **69/128 home-start placements (53.90625%)**,
with 96 grasps, 86 lifts, 76 insertions, 6 lost vials, no contacts above the
20 N audit threshold, and a 16.74 N peak. Every lifted episode saw the rack:
86/86. Of all 128 episodes, 62 required a scan; mean scan duration across all
episodes was 41.2 control steps (1.37 s), maximum 389 steps (12.97 s).
Successful completion averaged 14.21 s. This small-batch result establishes
that the overview-and-pan strategy can expose the rack without sacrificing
the state teacher's measured placement rate; it is not a visual-student result
or a physical safety certification. Log: `/tmp/so101_search_expert_v2_teacher_eval.log`.

Final ordinary placement-DAgger799 audit: 1/128 placement (0.78125%), 83 grasps,
61 lifts, 4 insertions, 9 lost vials, no unsafe contacts, peak 15.04 N, mean
successful completion 35.5 s. This run is not accepted. It motivates making
the expert's transport labels visually observable before more cloning.

Started `sim2real_search_memory64_fixed`: 512 worlds, 800 planned iterations,
128 rollout steps, 64-step backpropagation through the GRU (2.13 s instead
of the previous 0.53 s), and learning rate 0.0002. The saved placement-PPO1500
student and state1100 teacher are reused with cleared optimizer state. The
training-only callback replaces teacher labels with the validated active-view
strategy until the rack is seen after lift. The student still receives only
wrist RGB and proprioception and executes its own actions. Exported policy
inputs, default home pose, and broad rack sampling remain unchanged. This
is an experimental recipe, to be accepted only by independent home-start audits.

```bash
PYTHONPATH=/tmp SO101_SEARCH_MODE=train uv run --no-sync --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 512 --max_iterations 800 --seed 42 --run_name sim2real_search_memory64_fixed \
  --checkpoint outputs/sim2real_audit/placement_student1500_teacher1100.pt \
  --external_callback so101_search_expert_v2.install preset=sim2real \
  agent.num_steps_per_env=128 agent.algorithm.gradient_length=64 \
  agent.algorithm.learning_rate=0.0002 \
  'env.events.reset_from_dataset.params.phase_weights=[4,0,0,1,1,1,2,0]'
```

Log: `/tmp/so101_search_memory64_fixed.log`. The callback is a temporary experiment;
its exact source is retained under `outputs/sim2real_audit/diagnostics/` before
promotion. Teacher visibility and home-start student placement are separate
acceptance measurements.

The initial `sim2real_search_memory64` attempt was stopped: its training
callback returned `[]`, causing the native launcher's intersection to discard
the preset and all Hydra overrides. That run actually used the default
32-step rollout, 16-step gradient, 0.0005 learning rate, all-phase resets,
and no diameter randomization. Its weights are not promoted. Returning `None`
preserves the unconsumed arguments; the restarted run's saved YAML confirms
128/64 steps, 0.0002 learning rate, the requested curriculum, randomized vial
diameters, and 40-second episodes. Earlier exact-audit callbacks return the
original argument list and preserved the preset. Earlier placement-curriculum
training used no external callback and retained the requested weights.
Saved configuration audit: `outputs/sim2real_audit/run_config_audit.json`.

Search expert v2, independent seed44: 75/128 placements (58.59375%), 109 grasps,
104 lifts, 82 insertions, 2 lost vials, no unsafe rack contacts, peak 15.68 N,
mean successful completion 14.86 s. Rack seen after lift in 103/104 lifted
episodes; 72 episodes scanned, maximum 390 steps. Across seeds43/44 the
expert completed 144/256 placements (56.25%) and saw the rack in 189/190
lifted episodes. These remain expert-controller measurements, not student
acceptance. Log: `/tmp/so101_search_expert_v2_teacher_seed44.log`.

The first long-memory student checkpoint100 completed **0/128 placements**,
with 62 grasps, 42 lifts, no insertions, 4 lost vials, no unsafe contacts, and
peak 8.32 N. Stopped this run rather than completing 800 unchanged iterations.
Log: `/tmp/so101_search_memory64_eval100.log`. A second configuration check
found that the loaded optimizer retained learning rate **0.0005**, despite
the requested/saved YAML value 0.0002: native checkpoint loading restores
optimizer parameter groups as well as moment estimates. Clearing only its
`state` does not reset the learning rate. This distinction is retained in
the experiment record.

Started `sim2real_search_rollin`, with the same CNN/GRU actor mean and teacher,
but explicit optimizer learning rate 0.0002 and student exploration standard
deviation 0.05. Initial expert control is selected for 80% of whole episodes,
decreasing linearly to zero over 600 updates; all later episodes use the student.
The teacher still labels every visited state. This supplies complete observable
transport/release trajectories before relying on student-only rollouts. The
exported actor receives no expert state or search controller. Training now uses
only home starts, preserving continuous perception history; full domain
randomization, 40-second episodes, 128-step rollout and 64-step backpropagation
remain active. Rack detection ignores the first two reset frames. Preparation:
`/tmp/so101_prepare_search_rollin.py`; callback `/tmp/so101_search_expert_v3.py`;
checkpoint `outputs/sim2real_audit/search_rollin_start.pt`;
log `/tmp/so101_search_rollin.log`. Independent home audits use student-only
playback regardless of the training expert fraction.

The exact callback source is archived, so the launch can also be reproduced
without relying on the temporary directory:

```bash
PYTHONPATH=outputs/sim2real_audit/diagnostics SO101_SEARCH_MODE=train \
  uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation --num_envs 512 \
  --max_iterations 800 --seed 42 --run_name sim2real_search_rollin \
  --checkpoint outputs/sim2real_audit/search_rollin_start.pt \
  --external_callback so101_search_expert_v3.install preset=sim2real \
  agent.num_steps_per_env=128 agent.algorithm.gradient_length=64 \
  agent.algorithm.learning_rate=0.0002
```

Resumed the paused 1024-world state-teacher run to checkpoint1600 to improve
the teacher's placement ceiling while the camera student acquires complete
trajectories. It will pause at that checkpoint for an independent audit.

Added reproducible saved-image export verification:

```bash
uv run --no-sync --extra sim2real python scripts/check_so101_export.py \
  --pipeline outputs/leapp_current_control_smoke_v2/IsaacTutorial-Place-Vial-SO101-Camera/IsaacTutorial-Place-Vial-SO101-Camera.yaml
```

All 60 steps passed central/near-limit feedback checks on the compatibility
export; no camera or serial port was opened. Log: `/tmp/so101_export_checker.log`.
Run the same checker on the final trained export before physical trials.

Native forward-kinematics reach audit sampled 65,536 soft-limit joint poses
with the gripper at 0.08 rad. For pad-axis vertical alignment at least 0.9,
measured maximum horizontal radii were 47.05 cm at 18–21 cm grasp height,
46.56 cm at 21–24 cm, 45.32 cm at 24–27 cm, and 44.78 cm at 27–30 cm.
These reachable examples do not include collision or trajectory feasibility
and do not certify every rack pose, but give no evidence for shrinking the
current 16–38 cm sampling annulus. No workspace change was made. Report:
`outputs/sim2real_audit/reachability_sweep.json`;
log `/tmp/so101_reachability_sweep.log`.

Rejected a hybrid-expert trial (`/tmp/so101_search_expert_v4.py`) that used
the audited visual-PPO800 actor for pickup, then switched to the active-view
state expert after a confirmed lift. It grasped in 116/128 episodes and lifted
in 108/128, but completed only 16/128 placements (12.5%), with 21 insertions,
11 lost vials and 2 unsafe rack contacts (peak 24.83 N). The successful
pickup prefix does not preserve the grasp/arm states expected by the learned
placement controller; this is the working explanation, not a measured root
cause. It is **not used for student training**. The current roll-in recipe
retains the state expert's complete pickup/search/place behavior.
Log: `/tmp/so101_search_expert_v4_hybrid_eval.log`.

Resumed state-teacher1200, home seed43: 69/128 placements (53.90625%), 92 grasps,
83 lifts, 71 insertions, 6 lost vials, no unsafe contacts, peak 13.38 N,
mean successful completion 12.53 s. This small improvement over teacher1100
is not treated as statistically established progress. Log:
`/tmp/so101_safe_1024_eval1200.log`.

Search-roll-in student100, home seed43 with **no expert assistance**: 0/128
placements, 12 grasps, 5 lifts, 1 insertion, 3 lost vials, no unsafe contacts,
peak 12.79 N. Expert-assisted training success is much higher and is explicitly
not acceptance evidence. The next checkpoint is needed to distinguish early
cloning regression from a persistently unsuitable recipe. Log:
`/tmp/so101_search_rollin_eval100.log`.

The v2 visibility count means at least 16 yellow pixels, not an entire rack or
visible openings. A stricter v5 expert requires all eight rack bounding-box
corners within the image (2-pixel margin), plus at least 48 yellow pixels.
Home seed43: 48/128 placements (37.5%), 99 grasps, 92 lifts, 57 insertions,
5 lost vials, 2 unsafe contacts, peak 31.62 N, mean successful completion
19.34 s. Only 73/92 lifted episodes obtained a full-rack view; some spent
up to 1121 steps scanning. This proves that the earlier patch-detection metric
was insufficient evidence for full visual localization. The v5 controller
is not promoted. Log: `/tmp/so101_search_expert_v5_full_view_eval.log`.

The current roll-in student will pause at checkpoint200 for an independent
home audit. A native-FK pose-bank search is testing whether several high
overview poses, each with a generic pan sweep, can expose the full rack
throughout the requested workspace. Its geometric coverage is an intermediate
check; occlusion, dynamic transitions and contact must be audited separately.

Roll-in student200 home audit: 0/128 placements, 33 grasps, 11 lifts,
no insertions or lost vials, one unsafe rack contact, peak 27.35 N.
Stopped this student run after preserving checkpoint200; its modest pickup
recovery does not establish task learning. Log:
`/tmp/so101_search_rollin_eval200.log`.

The first geometric overview bank covered 248/256 sampled rack poses with
three pose/pan combinations; a wider candidate bank covered 250/256.
The best single pose is shoulder−1.3, elbow0.3, wrist-flex1.6, wrist-roll−1.25
rad, swept through pan±1.8 rad. These are FK/frustum results, with no occlusion
or trajectory validation. Reports: `outputs/sim2real_audit/overview_bank.json`
and `overview_bank_v2.json`. Eight-corner full-frame visibility is deliberately
stricter than seeing a yellow patch.

Dynamic v6 test of that single best pose: 54/128 placements (42.1875%),
99 grasps, 90 lifts, 56 insertions, 6 lost vials, no unsafe contacts,
peak 17.67 N, mean successful completion 18.24 s. Only 71/90 lifted episodes
obtained a full-rack view, despite the better geometric coverage. It is not
promoted. A follow-up diagnostic records pose tracking, pan extent, and
frustum versus pixel visibility in the missed episodes to distinguish an
unreachable camera view from dynamics or occlusion. Log:
`/tmp/so101_search_expert_v6_full_view_eval.log`.

V7 instrumentation of the same single-pose controller found 15 scanned
episodes without an accepted rack view: four never tracked the overview pose
within 0.1 rad; five obtained full bounding-box frustum coverage but not the
48 yellow pixels; the remainder did not obtain full coverage during the
available motion. The repeated batch completed 53/128 placements with one
unsafe contact (peak 28.92 N). Small native-GPU repeat differences are retained,
not treated as deterministic equality. Report:
`outputs/sim2real_audit/search_full_view_diagnostic.json`;
log `/tmp/so101_search_expert_v7_diagnostic.log`.

State-teacher1600 home seed43: **83/128 placements (64.84375%)**, 109 grasps,
108 lifts, 94 insertions, 5 lost vials, one unsafe contact, peak 28.94 N,
mean successful completion 10.54 s. The state run is paused at this checkpoint.
This is a stronger privileged teacher, not a deployable camera policy. Log:
`/tmp/so101_safe_1024_eval1600.log`.

Prepared `outputs/sim2real_audit/search_teacher1600_start.pt` using the existing
warm-start helper: original placement-PPO1500 student mean, explicit 0.05
exploration standard deviation, cleared optimizer at learning rate 0.0002,
and the stronger state1600 teacher. V8 is being audited before student training.
It requires the full eight-corner rack bbox in frame, accepts 16 native yellow
pixels, and cycles three overview wrist-roll variants after complete pan
sweeps. It uses whole expert episodes at a fixed 80% fraction for the planned
learning-capacity ablation; that fraction is not a final acceptance recipe.
Broad placement and all physical/material DR remain unchanged. A temporary
observation-noise ablation, if needed, will be explicit and must subsequently
be re-enabled before accepting a sim2real policy.

V8, stronger teacher1600 with three full-frame overview scans: 68/128
placements (53.125%), 109 grasps, 107 lifts, 78 insertions, 3 lost vials,
one contact slightly above the 20 N threshold (peak 20.03 N), mean completion
17.61 s, full-rack framing plus visible pixels in 95/107 lifted episodes.
Log: `/tmp/so101_search_expert_v8_teacher1600_eval.log`.

V9 removes axial-spin/sign ambiguity from the teacher's vial quaternion by
querying the same actor with the canonical shortest rotation from +Z to the
vial axis. The physical object, axis, velocities, rewards and collisions are
unchanged. This is a training-only teacher-input transformation: the student
still receives only RGB and proprioception. Trial: 72/128 placements (56.25%),
107 grasps, 100 lifts, 81 insertions, 6 lost vials, one unsafe contact,
peak 23.15 N, mean completion 19.36 s, full-rack framing plus visible pixels
in 92/100 lifted episodes. This small score difference is not claimed as a
statistically established improvement. Log:
`/tmp/so101_search_expert_v9_symmetric_eval.log`.

Started controlled cloning ablation `sim2real_clean_clone`: 512 worlds, 500
planned iterations, home-only resets, 64-step rollouts, 32-step backpropagation,
8 learning epochs, explicit optimizer learning rate 0.0002, exploration 0.05,
fixed 80% expert-controlled whole episodes. The V9 expert labels every state.
All physical and material/appearance DR remain enabled; **only image and
proprioceptive observation corruption are temporarily disabled**. This tests
learning capacity before restoring the full sensory randomization. Its
weights cannot be accepted as sim2real-ready merely for succeeding here.

```bash
PYTHONPATH=outputs/sim2real_audit/diagnostics SO101_SEARCH_MODE=train \
  uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation --num_envs 512 \
  --max_iterations 500 --seed 42 --run_name sim2real_clean_clone \
  --checkpoint outputs/sim2real_audit/search_teacher1600_start.pt \
  --external_callback so101_search_expert_v9.install preset=sim2real \
  agent.num_steps_per_env=64 agent.algorithm.gradient_length=32 \
  agent.algorithm.num_learning_epochs=8 agent.algorithm.learning_rate=0.0002 \
  env.observations.wrist_rgb.enable_corruption=False \
  env.observations.proprioception.enable_corruption=False
```

Run directory: `logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_23-17-32_sim2real_clean_clone`;
log `/tmp/so101_clean_clone.log`. Resolved YAML confirms the overrides and
randomized vial diameters. Independent audits remain student-only home starts.

Clean-clone home audits: checkpoint100 had 55 grasps and 40 lifts but 0/128
placements; checkpoint200 had 31 grasps, 15 lifts, 1 insertion, 6 lost vials,
no unsafe contacts and peak 13.16 N; checkpoint300 had 38 grasps, 15 lifts,
no insertions, 2 lost vials, no unsafe contacts and peak 18.92 N, again 0/128
placements. Lower imitation loss did not produce task success. The experiment
also changed the teacher, symmetry handling, view strategy and optimization
settings, so it does not isolate observation corruption as the sole cause.
Logs: `/tmp/so101_clean_clone_eval100.log`, `eval200.log`, `eval300.log`
(all carry the `so101_clean_clone_` prefix).

Added optional `localization_features=True` to the existing visual-memory
model. It appends twelve measurements derived **only from the RGB image**:
blue-cap and yellow-rack centers, spatial variance/covariance and pixel area
fractions. The CNN and recurrent memory remain active. Missing colors yield
finite zero features; no object poses or simulator flags are new actor inputs.
These coarse color measurements complement image features; they are not a
validated object detector or a guarantee against occlusion/lighting changes.
The default remains false until the experiment is validated.

The GRU input expands from 792 to 804. Warm migration copies the previous
CNN/GRU/MLP/normalizer weights and initializes the twelve new GRU columns to
zero. Eight independent image/state steps numerically preserved the initial
actor output. The checkpoint helper now supports an explicit localization
template and learning-rate reset rather than accidentally restoring a stale
optimizer rate. Preparation:

```bash
uv run --no-sync --extra sim2real python scripts/distillation_checkpoint.py warm-start \
  --student outputs/sim2real_audit/search_teacher1600_start.pt \
  --teacher logs/rsl_rl/so101_vial_state/2026-10-06_21-16-13_sim2real_safe_1024/model_1600.pt \
  --localization-template logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_23-29-45_sim2real_localization_template/model_1.pt \
  --learning-rate 0.0002 --output outputs/sim2real_audit/localization_teacher1600_start.pt
```

Validation: 76 tests pass (including both export/reset branches and image
location/missing-object measurements), Ruff passes, all 44 Python files are
formatted. A two-update native simulation load/training smoke passed with
full sensory randomization. Logs: `/tmp/so101_localization_full_tests.log`,
`/tmp/so101_localization_warm_parity.log`, `/tmp/so101_localization_loaded_smoke.log`.

Started `sim2real_localized_clone`: 512 worlds, 800 planned iterations, same
symmetry-aware view expert and fixed 80% expert roll-in, 64/32 rollout/gradient
steps, 8 epochs, explicit learning rate 0.0002, student exploration 0.05,
`agent.student.localization_features=True`. **Full image and proprioceptive
randomization are enabled again**, alongside all physical/material DR and
broad placement. Log: `/tmp/so101_localized_clone.log`. Student-only home
audits and a matching LEAPP export are still required; expanded checkpoints
must be loaded with the matching localization flag.

### Localization experiment: export checks and early independent audit

The clean-image clone was stopped after iteration 400 was saved. Its final
student-only home audit completed 0/128 placements, 58 grasps, 46 lifts, no
insertions, 4 lost vials, no contacts above 20 N, and peak rack contact 17.04 N.
It is not an accepted camera policy (`/tmp/so101_clean_clone_eval400.log`).

The new localization branch passed LEAPP's 5/5 native numerical checks after
a short native training run gave its additional GRU input columns nonzero
weights (norm 0.204118). Its compatibility pipeline is:
`outputs/leapp_localization_nonzero_smoke/IsaacTutorial-Place-Vial-SO101-Camera/IsaacTutorial-Place-Vial-SO101-Camera.yaml`.
The saved focused camera image also passed all 60 feedback/soft-limit checks
through `scripts/check_so101_export.py`; no camera or serial device was opened
by this check. This proves software compatibility, not learned task quality.
Logs: `/tmp/so101_localization_nonzero_leapp_export.log` and
`/tmp/so101_localization_export_checker.log`.

Current learning run:
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-06_23-33-50_sim2real_localized_clone`.
Exact command (training callback is experimental and archived with diagnostics):

```bash
PYTHONPATH=/tmp SO101_SEARCH_MODE=train uv run --no-sync --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 512 --max_iterations 800 --seed 42 --run_name sim2real_localized_clone \
  --checkpoint outputs/sim2real_audit/localization_teacher1600_start.pt \
  --external_callback so101_search_expert_v9.install preset=sim2real \
  agent.student.localization_features=True agent.num_steps_per_env=64 \
  agent.algorithm.gradient_length=32 agent.algorithm.num_learning_epochs=8 \
  agent.algorithm.learning_rate=0.0002
```

At checkpoint 100, independent student-only play from home (128 worlds,
seed43, full broad placement and physical/material DR) achieved **0/128
placements**, 19 grasps, 17 lifts, no insertions, 1 lost vial and 1 contact
above 20 N (peak 24.42 N). The expert callback is absent from this audit.
Play disables synthetic observation corruption; thus this result does not
establish robustness to all training image/noise perturbations. Loss reduction
is not an acceptance criterion. Log: `/tmp/so101_localized_clone_eval100.log`.

At localized-clone checkpoint 200, student-only seed43 play again completed
0/128 placements: 25 grasps, 5 lifts, no insertions, 6 lost vials, no rack
contacts above 20 N (peak 13.36 N). Log:
`/tmp/so101_localized_clone_eval200.log`. The clone is paused after saving
checkpoint 300 for a final diagnostic audit; declining behavior does not
justify blindly finishing the planned 800 updates.

Started a separate placement-focused reward-learning experiment from the
strongest prior camera pickup policy (PPO800: 116 grasps, 111 lifts/128).
`/tmp/so101_prepare_localized_ppo.py` copied its entire actor, extended the
GRU by twelve zero-weight image-location inputs, reused the state1600 critic,
cleared optimizer moments, reset iteration to zero and learning rate to
0.0001, and set training exploration to 0.15. The deterministic actor is
preserved initially; no privileged teacher/controller runs during collection.
Start artifact: `outputs/sim2real_audit/localized_pickup800_ppo_start.pt`.
New run: `logs/rsl_rl/so101_vial_camera/2026-10-06_23-48-52_sim2real_localized_placement_ppo`.

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 512 \
  --max_iterations 1500 --seed 42 --run_name sim2real_localized_placement_ppo \
  --checkpoint outputs/sim2real_audit/localized_pickup800_ppo_start.pt \
  preset=sim2real agent.actor.localization_features=True \
  agent.algorithm.learning_rate=0.0001 \
  'env.events.reset_from_dataset.params.phase_weights=[1,0,0,1,2,2,4,0]'
```

This curriculum gives held/insertion starts more training weight to focus on
the missing transport/release behavior. Phase0 home starts remain present;
all phase-specific geometry/contact filtering and broad rack poses remain
active. Final acceptance still requires independent home-start camera-only
play, not curriculum metrics. Full sim2real image/proprioception and physical
DR remain enabled during training.

Localized clone300 completed 0/128 placements, 51 grasps, 46 lifts,
no insertions, 6 lost vials and no contacts above 20 N (peak 18.70 N).
Log: `/tmp/so101_localized_clone_eval300.log`. The run was stopped with all
checkpoints preserved. These results do not establish that the color features
are ineffective; they establish that this imitation recipe remains inadequate.
The independent placement-focused PPO experiment continues.

### Corrected cap direction in the transport reward

Found a reward/task mismatch in `symmetric_axial_keypoint_error`: it selected
the smaller error after swapping the vial's two axial endpoints. An upside-down
vial shifted 83 mm to preserve the same physical center therefore scored zero
pose error, despite cap-up placement being required by insertion and success.
Rotation around the cylinder's axis is a symmetry; exchanging cap and bottom
is not. The reward now compares corresponding center/bottom/cap keypoints,
preserving the authored root offset and ignoring axial spin while penalizing
cap-down orientation. Actor inputs, action semantics, physical randomization
and success criteria are unchanged. Existing yaw/translation/tilt checks remain;
the corrected center-offset regression rejects the previously equivalent
cap-down pose. Focused geometry/reward checks: 14 pass; Ruff passes.

The pre-fix localized placement PPO was stopped after iteration139 (checkpoint
100 preserved). Its insertion-stage-only audit had 104/128 inserted episodes,
no placements, 2 lost vials, no impacts above 20 N and peak contact 19.93 N.
This audit starts with the vial already held near insertion and is **not** a
home-start score. Log: `/tmp/so101_localized_placement_eval100_phase6.log`.

Restarted from the same pickup800 warm start and the same placement curriculum,
with only the cap-direction reward correction:
`sim2real_cap_up_placement_ppo`, log `/tmp/so101_cap_up_placement_ppo.log`.
The previous paused ordinary state-teacher process was stopped as well;
checkpoint1600 is preserved, but its in-memory reward code would have been stale
if resumed. Further training should start a fresh process with current source.
Tests: `/tmp/so101_cap_direction_tests.log`.

### Placement progress reward experiment (not promoted to task defaults)

Cap-up PPO100's insertion-stage-only audit still had 0/128 placements,
102 inserted episodes, 2 lost vials and 3 contacts above 20 N (peak 20.23 N).
Log: `/tmp/so101_cap_up_placement_eval100_phase6.log`. The orientation correction
is justified by the task contract, but has not alone solved release learning.

A separate experimental callback replaces the held-goal Gaussian with a
progress signal during held-object manipulation:
`potential = -cap_up_pose_error / 0.01 + 10 * inserted * clamp(openness / 0.25, 0, 1)`.
Its per-step difference is clipped to [-1,1], with reward weight1, disabled
before the grasp and on the first reset step. Opening the jaws contributes
only while the vial is physically upright and inserted. Withdrawing or closing
reduces this potential. There is no constant positive reward merely for holding
still. Clipping means the accumulated reward is not an exact telescoping
potential. The unchanged physical success check still requires released,
upright, seated, slow motion for ten consecutive steps.

Prototype `/tmp/so101_task_progress_experiment.py` is training-only; it has
not changed production reward defaults or added actor observations. Native
16-world, two-update smoke passed with full sim2real DR and near-insertion
starts (6.28 s training; `/tmp/so101_task_progress_smoke.log`). The resolved
configuration confirms that `held_goal.func` uses the experimental term.

Started 512-world `sim2real_task_progress_ppo` from the same pickup800 warm
start, with the same learning rate, exploration, placement curriculum and
full DR. Run:
`logs/rsl_rl/so101_vial_camera/2026-10-06_23-58-53_sim2real_task_progress_ppo`.
Use the previous cap-up training command with
`--run_name sim2real_task_progress_ppo --external_callback so101_task_progress_experiment.install`
and `PYTHONPATH=/tmp`. Log: `/tmp/so101_task_progress_ppo.log`.
No expert actions run in this experiment. Independent release-stage and
home-start evaluations must justify promoting the reward or accepting weights.

Cap-up PPO200's release-only audit: 0/128 placements, 110 insertion histories,
no lost vials, 4 contacts above 20 N and peak rack contact 24.28 N.
Log: `/tmp/so101_cap_up_placement_eval200_phase6.log`. The process was stopped
with checkpoint200 preserved; training continues only on the progress-reward
experiment. Stochastic curriculum training successes are not independent
camera-only success evidence.

### Isolated release feasibility and release-only cloning

The state1600 teacher completed **128/128** insertion-stage-only episodes
(seed43, current cap-up geometry/reward source, full physical/material DR),
averaging 1.604 s. No vials were lost and no contacts exceeded 20 N; peak
rack force was 8.42 N. Thus the near-insertion reset and simulated release
mechanics are feasible for this privileged controller. This is not a
camera-policy score and does not establish home-start success.
Log: `/tmp/so101_teacher1600_eval_phase6.log`.

Progress-reward camera PPO100 still completed 0/128 release-stage trials,
110 insertion histories, 1 lost vial and 1 contact above 20 N (peak 20.43 N).
Log: `/tmp/so101_task_progress_eval100_phase6.log`. PPO200 is undergoing a
separate complete home-start audit.

To isolate imitation of the known-feasible release behavior, started
`sim2real_release_clone`: the same pickup800 camera actor with optional image
locations, teacher1600, 256 worlds, 400 planned updates, release-stage-only
resets, 80% whole-episode teacher roll-in, 64/32 rollout/gradient steps,
4 epochs, explicit learning rate0.0001 and student exploration0.05.
Start artifact: `outputs/sim2real_audit/release_clone_start.pt`.
Preparation: `/tmp/so101_prepare_release_clone.py`. Experimental collection
callback: `/tmp/so101_release_clone.py` (returns None to preserve Hydra args).
All image/proprioceptive and physical/material training DR remain enabled.
No privileged state enters the student. The actual teacher actions are clipped
before being used as labels and roll-in actions. This experiment may affect
pickup, so both release-only and home-start audits are needed before combining
it with further full-task learning.

```bash
PYTHONPATH=/tmp uv run --no-sync --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 256 --max_iterations 400 --seed 42 --run_name sim2real_release_clone \
  --checkpoint outputs/sim2real_audit/release_clone_start.pt \
  --external_callback so101_release_clone.install preset=sim2real \
  agent.student.localization_features=True agent.num_steps_per_env=64 \
  agent.algorithm.gradient_length=32 agent.algorithm.num_learning_epochs=4 \
  agent.algorithm.learning_rate=0.0001 \
  'env.events.reset_from_dataset.params.phase_weights=[0,0,0,0,0,0,1,0]'
```

After the cap-direction change, all76 tests pass (42 dependency warnings,
4.20 s); Ruff passes; all46 Python files are formatted; guide/README spelling
passes. Logs use the `so101_cap_direction_full_` prefix. Prepared
`/tmp/so101_augmented_home_eval.py` to explicitly re-enable both camera and
proprioception corruption after play-mode configuration; this audit has not
run yet. It is separate from nominal-image home qualification.

Progress-reward PPO200 complete home audit: 0/128 placements, 92 grasps,
82 lifts, no insertions, 5 lost vials, no contacts above20 N, peak17.88 N.
Log: `/tmp/so101_task_progress_eval200_home.log`. The run is paused after
saving300 for a final release-stage audit. The pickup800 baseline remains
preserved; changed training rewards have not earned promotion to defaults.
The release-only cloning run directory is:
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-07_00-06-36_sim2real_release_clone`.

Release-clone100 independent camera-only release audit achieved **123/128
placements (96.09375%)**, average1.839 s, all128 insertion histories, no lost
vials and no contacts above20 N; peak11.39 N. The expert callback is absent
from play. This validates the release stage under nominal images and physical/
material DR, **not** the complete home-start task or real robot transfer.
Log: `/tmp/so101_release_clone_eval100_phase6.log`. Its home-start pickup
retention audit is running. This is the first independently strong learned
camera-stage result; release behavior can now seed broader training.

Release-clone100 home-start retention audit: **0/128 grasps, lifts, insertions
or placements**, all timeouts, no rack contacts. It learned release but forgot
pickup after training only on release starts. Log:
`/tmp/so101_release_clone_eval100_home.log`. The stage score is not a full-task
qualification.

Prepared `outputs/sim2real_audit/mixed_stage_release100_start.pt` using the
checkpoint helper: unchanged release100 student, teacher1600, optimizer cleared,
iteration0 and learning rate0.0002. Mixed-stage cloning uses 512 worlds,
1000 planned updates, 64/32 rollout/gradient steps, 8 epochs, image locations,
80% whole-episode symmetry-aware view-expert roll-in and phase weights
`[4,0,0,1,2,2,4,0]`. This deliberately includes pickup, grasped, lifted and
release data together to address forgetting. The first mixed run was stopped
at iteration23 to incorporate the appearance fix below; its logs/checkpoints
remain preserved.

Measured the actual focused camera's rack colors: upper-region median RGB
[0.733,0.714,0.459], wall median [0.537,0.565,0.365]. The existing strict yellow
feature detected only6.2% of upper-region pixels and none of the sampled wall
region (1396 yellow pixels across the full640x480 image). These are manually
chosen regions, not segmentation ground truth. The physical camera sees much
paler colors than the saturated renderer. Added episode-fixed saturation
variation **0.35–1.2** to `Sim2RealCameraImage`, using luminance-preserving
chroma scaling. It is active only under the sim2real observation augmentation;
play/export disable it. The RGB-derived color features remain optional aids,
not a validated detector. The CNN still receives the complete image.

Restarted mixed training from the same warm start with saturation enabled:
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-07_00-20-29_sim2real_mixed_stage_saturation`.
Log: `/tmp/so101_mixed_stage_saturation.log`. Command matches the prior view
expert recipe, with `--run_name sim2real_mixed_stage_saturation`, the mixed warm
checkpoint and phase weights above. Lint/format checks passed after this change;
native startup and training with the new augmentation passed. Broader camera
quality audits remain pending.

Exported the **actual trained release100 student** through the matching camera
PPO template (critic required by the loader but absent from inference):
`outputs/sim2real_audit/release_stage100_export.pt`. LEAPP pipeline:
`outputs/leapp_release_stage100/IsaacTutorial-Place-Vial-SO101-Camera/IsaacTutorial-Place-Vial-SO101-Camera.yaml`.
Native validation passed **5/5 examples**; the focused-image bridge checker
passed **60/60** central/lower/upper-bound feedback tests without opening serial
or camera devices. Logs: `/tmp/so101_release_stage100_leapp_export.log` and
`/tmp/so101_release_stage100_export_checker.log`. This is a release-stage artifact;
it is not approved for complete-task real rollouts. The joint map stays unverified.

Resumed the state teacher in a fresh process with current cap-up reward source,
from1600, 1024 worlds, 1000 additional iterations, seed42, full sim2real physical/
material DR and phase weights `[16,1,1,1,1,1,1,1]` (about70% home starts).
Run name `sim2real_cap_up_teacher`; log `/tmp/so101_cap_up_teacher.log`.
This can improve the teacher's64.8% complete-task ceiling before subsequent
student refinement. It retains the existing held-goal reward; the experimental
progress reward has not been promoted.

Release-only training finished at checkpoint399 (400 updates, 948.95 s native
training). Independent **seed44** release-stage evaluation achieved125/128
placements (97.65625%), average1.842 s, no lost vials or contacts above20 N,
peak8.79 N. Log: `/tmp/so101_release_clone_eval399_phase6_seed44.log`.
This supports repeatable release-stage performance; it is a different checkpoint
and seed, not evidence that the complete task is solved. The mixed run starts
from the earlier validated100 checkpoint.

The trained release100 export now includes a model card and SHA256 manifest
in `outputs/leapp_release_stage100/`. They explicitly record its stage-only
scope and failed home-start qualification. After adding saturation variation,
all76 tests pass (4.51 s,42 dependency warnings), Ruff and all46-file formatting
checks pass, and guide/README spelling passes. Logs use the
`so101_saturation_full_` prefix.


## Preferred grasp direction and longer visual memory — October 7, 2026

Complete wrist-camera placement remains unqualified. Strong release-stage scores
are not complete-task scores. The physical joint map remains unverified; no
policy-driven hardware motion has been performed.

The mixed-stage saturation run failed to retain useful pickup: checkpoint100
had10 grasps/2 lifts and checkpoint200 had11 grasps/9 lifts, with **0/128
placements** in each independent home audit. It was stopped after checkpoint500;
checkpoints and logs remain preserved. Adding stage starts alone did not solve
forgetting. Logs: `/tmp/so101_mixed_stage_saturation.log` and matching evaluation
logs. The temporary task-progress and cap-up camera PPO experiments were also
stopped after their unsuccessful audits; their reward modifications were not
promoted into the task.

A native wrist-view diagnostic at the old camera PPO800 checkpoint measured110
first qualified lifts:52 preferred pad directions (cosine above0.7),45 opposite
directions (below-0.7),13 other. The camera optical axis has dot product-0.5736
with the pad axis. An opposite grasp can point the camera away from the table
when the vial is upright; this is a simulation observation, not a measured
physical-camera extrinsic calibration. Artifact:
`outputs/sim2real_audit/grasp_view_direction.json`.

Changed the existing grasp-approach error to prefer the pad direction toward
the vial cap, rather than treating both axial directions equally. Added an
optional6-point alignment bonus to the existing first-grasp milestone when
cosine exceeds0.85. It is paid once, ignores already-grasped reset states, and
uses the existing milestone tracking. Neither the physical grasp definition nor
placement success criterion changed. Added meaningful direction/reset tests;
**77 tests pass**,42 dependency warnings,4.44 s. Ruff and46-file formatting pass.
Logs: `/tmp/so101_camera_grasp_full_tests.log` and corresponding lint/format logs.

Resumed the state teacher from cap-up checkpoint2200 with512 worlds,800
additional updates, seed42, full sim2real physics/material randomization and
phase weights `[8,4,2,1,1,1,1,0]`:

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 512 \
  --max_iterations 800 --seed 42 --run_name sim2real_camera_grasp_teacher \
  --checkpoint logs/rsl_rl/so101_vial_state/2026-10-07_00-21-59_sim2real_cap_up_teacher/model_2200.pt \
  preset=sim2real 'env.events.reset_from_dataset.params.phase_weights=[8,4,2,1,1,1,1,0]'
```

Run: `logs/rsl_rl/so101_vial_state/2026-10-07_01-14-38_sim2real_camera_grasp_teacher`.
Independent checkpoint2350 home audit: **88/128 placements (68.75%)**,114
grasps,106 lifts,95 insertions,4 lost vials, no contacts above20 N, peak17.95 N,
mean successful episode12.15 s. Of106 measured lifts,105 had cosine above0.7
and97 above0.85; none had opposite cosine below-0.7. Earlier cap-up1650 also
scored88/128, so this is not evidence of a statistically significant success
improvement. Logs: `/tmp/so101_camera_grasp_teacher_eval2350_home.log`;
artifact: `outputs/sim2real_audit/camera_grasp_teacher2350.json`.

The camera PPO warm start preserves pickup800, enables image-derived location
features, clears its optimizer and uses the teacher critic. The camera policy
receives RGB, joint positions/velocities, previous target and internal action/
recurrent state; privileged teacher state is absent from deployment.

```bash
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 512 \
  --max_iterations 1200 --seed 42 --run_name sim2real_camera_grasp_ppo \
  --checkpoint outputs/sim2real_audit/localized_pickup800_ppo_start.pt \
  preset=sim2real agent.actor.localization_features=True \
  agent.num_steps_per_env=128 agent.algorithm.lam=0.99 \
  agent.algorithm.learning_rate=0.0001
```

Run: `logs/rsl_rl/so101_vial_camera/2026-10-07_01-17-28_sim2real_camera_grasp_ppo`.
Checkpoint100 independent home audit:115/128 grasps (89.84%),112/128 lifts
(87.5%), **0 placements or insertions**,3 lost vials, no contacts above20 N,
peak18.48 N. Of112 measured lifts,61 preferred and48 opposite; direction is
still inconsistent. Artifact: `outputs/sim2real_audit/camera_grasp_ppo100.json`.
This remains a pickup policy, not a complete placement policy.

The temporary V10 hybrid expert uses frozen pickup800 camera control before
lifting, then the state/view expert. With teacher1800 it completed32/128 full
home trials (25%),112 grasps,107 lifts,38 insertions,9 lost vials, one contact
above20 N, peak23.92 N. Its handoff and opposite grasps limit the ceiling; it
has not been promoted into production or deployment.

Prepared a longer-sequence cloning warm start:
`outputs/sim2real_audit/long_memory_teacher2350_start.pt`, preserving the camera
pickup actor and replacing only the teacher with2350, clearing optimizer,
iteration0, learning rate0.0003. Rollout and gradient windows are512 steps
(17.07 s at30 Hz), rather than the prior32-step backpropagation window.
Temporary callbacks use native trajectory splitting/padding for episode resets
and an executed-command loss: equal saturated commands incur zero error;
wrong saturation has a straight-through corrective gradient. This is an
experimental optimization, not an upstream algorithm change.

CPU parity checks with staggered resets and nonzero initial memory match the
per-step recurrent actions and final hidden states. Native16-world two-update
smokes passed with the initial batching implementation. The128-world run
`2026-10-07_01-32-20_sim2real_long_memory_clone` was stopped after a few updates:
45.5 s/update and about80.8 GB GPU memory were too expensive. It produced no
qualified checkpoint. The next version encodes each real image once before
padding latent trajectories, and explicitly enables BF16 training in its
temporary callback. This runtime override is not represented by the serialized
runner config and must accompany any reproduction. The rollout policy and
export weights remain float32. Smoke verification is in progress; this does
not establish full-task performance.

Experimental helpers/logs use `/tmp/so101_batched_hybrid.py`,
`/tmp/so101_batched_search.py`, `/tmp/so101_batched_hybrid_parity.py`,
`/tmp/so101_batched_hybrid_parity.log`, `/tmp/so101_encoded_memory_smoke.log`.
They will be copied into the diagnostic archive with hashes. Broad independent
rack/vial placement, full yaw and non-overlap constraints remain unchanged.


Camera-grasp PPO200 independent seed43 home audit: **122/128 grasps (95.31%),
117/128 lifts (91.41%), zero insertions and placements**,3 lost vials, no
contacts above20 N, peak19.74 N. Of116 first qualified lifts captured by the
alignment diagnostic,60 preferred and53 opposite; the improvement in pickup
does not establish a consistent grasp direction. Log:
`/tmp/so101_camera_grasp_ppo_eval200_home.log`; artifact:
`outputs/sim2real_audit/camera_grasp_ppo200.json`. The capture counter can differ
from the history-based final lift counter because it additionally requires
bilateral contact at the sampled frame.

The encoded-sequence smoke exposed two mixed-precision boundary issues: GRU
input/initial-state dtype must agree, and the final training hidden state must
be converted back to float32 before normal rollout. Both corrections are in
the temporary helper. A complete two-update smoke is being repeated to verify
update-to-rollout continuity. The optimized CPU parity check also passes.
The current-contract metric manifest and run-config audit have been refreshed;
helper scripts/logs are copied to `outputs/sim2real_audit/diagnostics/` with a
SHA256 manifest. Failed runs remain recorded rather than treated as accepted.


The corrected encoded/BF16 smoke completed both updates and resumed normal
float32 rollout between them:53.05 s,16 worlds. This verifies the training
path, not policy quality. Started the128-world long-memory experiment with600
planned updates and unchanged broad/full-DR task:

```bash
SO101_SEARCH_MODE=train PYTHONPATH=/tmp \
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 128 --max_iterations 600 --seed 42 \
  --run_name sim2real_long_memory_encoded \
  --checkpoint outputs/sim2real_audit/long_memory_teacher2350_start.pt \
  --external_callback so101_batched_search.install preset=sim2real \
  agent.student.localization_features=True agent.num_steps_per_env=512 \
  agent.algorithm.gradient_length=512 agent.algorithm.num_learning_epochs=4 \
  agent.algorithm.learning_rate=0.0003 \
  'env.events.reset_from_dataset.params.phase_weights=[4,0,0,1,2,2,4,0]'
```

Log: `/tmp/so101_long_memory_encoded.log`. Temporary callback details and
mixed-precision runtime override are documented above; archived copies are
required when reproducing outside `/tmp`. Independent camera-only home audits
will determine whether the long memory helps placement. The80% expert roll-in
training metric is not an independent student success rate.


Independent teacher2600 home audit with fresh seed44 also scored **88/128
placements (68.75%)**,107 grasps,106 lifts,94 insertions,3 lost vials, no
contacts above20 N, peak15.57 N, mean successful episode10.21 s.
Grasp-direction artifact: `outputs/sim2real_audit/camera_grasp_teacher2600.json`
(106/106 measured lifts with preferred pad direction).
Log: `/tmp/so101_camera_grasp_teacher_eval2600_home_seed44.log`.
This replicates the teacher score on another seed; camera placement is still
unqualified.


The128-world encoded/BF16 run settled at about26.2 s/update versus45.5 s for
the earlier padded-image version, with total concurrent GPU memory about34 GB
versus about80.8 GB previously. Other concurrent jobs affect these timings;
this is a practical throughput observation rather than a controlled benchmark.
The current600-update run takes several hours. Its exact run directory is
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-07_01-51-33_sim2real_long_memory_encoded`
(check the run log's experiment directory if reproducing timestamps).

Started a separate camera PPO200 home audit at seed44 with training camera and
proprioception corruption explicitly re-enabled, using
`so101_augmented_home_eval.install`. Nominal play disables observation
augmentation, so that previous clean-image score is not an augmented score.
Log: `/tmp/so101_camera_grasp_ppo_eval200_augmented_seed44.log`.


Camera PPO200 augmented home audit, seed44:111/128 grasps (86.72%),106/128
lifts (82.81%), **zero insertions or placements**,9 lost vials, no contacts
above20 N, peak13.75 N. Both camera and proprioception corruption were explicitly
re-enabled, including image latency, photometric changes and calibrated-state
noise. This is a different seed from the clean audit, so the difference cannot
be attributed only to augmentation. Log:
`/tmp/so101_camera_grasp_ppo_eval200_augmented_seed44.log`.


Restarted the encoded-memory experiment after its initial throughput check,
from its saved checkpoint0, to save every20 updates instead of100. This
preserves the first completed update and permits earlier independent failure
checks rather than waiting roughly45 minutes for the first audit. Several
unsaved updates from the throughput check were discarded; no qualified policy
was lost. New run name `sim2real_long_memory_audited`, log
`/tmp/so101_long_memory_audited.log`. Command is identical to the encoded run
above except the saved model0 input, run name and `agent.save_interval=20`.
The original throughput run is stopped; the audited run supersedes it.


Camera PPO300 home audit, seed43:113/128 grasps,112/128 lifts, **zero insertions
or placements**,8 lost vials, one contact above20 N (peak22.68 N). The first
qualified-lift alignment sample was61 preferred/49 opposite/2 other. The
better-audited pickup checkpoint remains200; later weights are not automatically
better. Log: `/tmp/so101_camera_grasp_ppo_eval300_home.log`.

Started a read-only checkpoint watcher `/tmp/so101_watch_long_memory.py`, log
`/tmp/so101_watch_long_memory.log`, for long-memory checkpoints20,60,100,200,
400 and599. Each evaluation uses128 first home-start episodes, seed43, the
camera-distillation student with image locations enabled, and only the exact
home episode-counter callback. It does not install the teacher/view controller
and does not connect to hardware. Results/commands are written under
`outputs/sim2real_audit/long_memory_evaluations/` and appended to this guide.
An assisted training score is never used as the student qualification score.


Checked camera PPO200 raw action means during native home playback, using a
measurement-only callback `so101_action_saturation_eval.install`. Across1200
sampled frames (153600 samples/joint, including reset episodes), fractions
above magnitude1 ranged from0.003% to8.49%; no means exceeded3. Per-joint
maxima ranged1.42–2.38. Thus severe saturation is not supported as the main
explanation for this camera PPO's lack of placement; no policy-head change was
made. Artifact: `outputs/sim2real_audit/camera_ppo200_action_saturation.json`.
The repeated seed43 evaluation had119 grasps/lifts, zero placements and one
contact above20 N (peak23.07 N), illustrating rollout variability despite a
fixed seed. Log: `/tmp/so101_action_saturation_eval.log`.

Started a separate native audit of the exact teacher2350 plus symmetry-aware
active-view controller used to label long-memory training. Its score will
establish the current label-controller ceiling. This audit deliberately runs
the privileged expert and is not a camera-only score. Log:
`/tmp/so101_camera_grasp_teacher2350_view_expert_eval.log`.


Automated long-memory camera-only home audit, checkpoint20, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.421875, "insertion_rate": 0.0, "lift_rate": 0.125, "max_rack_contact_force_n": 14.023468971252441, "mean_peak_rack_contact_force_n": 0.43332988023757935, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.953125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.046875}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/long_memory_evaluations/checkpoint_20_home_seed43.log`.


Teacher2350 with the exact symmetry-aware active-view controller used by the
long-memory labels completed **79/128 home trials (61.71875%)**,106 grasps,
104 lifts,84 insertions,2 lost vials, one contact above20 N, peak39.61 N,
mean successful episode18.16 s. It scanned103 episodes and framed the full rack
in92 episodes after lifting. These are privileged expert results, not student
results. Log: `/tmp/so101_camera_grasp_teacher2350_view_expert_eval.log`.
The view-controller ceiling is lower than the unmodified teacher's68.75% and
still has occasional excessive contact; it is not a perfect demonstration set.
The long-memory checkpoint20 audit above retained54 grasps/16 lifts but no
placements; continued learning requires evidence of recovery at subsequent
checkpoints. No complete visual policy has been accepted.


Camera-grasp PPO400 home audit, seed43:114/128 grasps and lifts, zero
insertions/placements,9 lost vials, one contact above20 N (peak34.69 N).
Measured first qualified lifts:55 preferred/52 opposite/6 other. This did not
improve the intended grasp-direction preference or placement. Stopped this PPO
run after logged iteration447; all checkpoints remain preserved. Checkpoint200
remains the better audited pickup checkpoint. Log:
`/tmp/so101_camera_grasp_ppo_eval400_home.log`. GPU resources now prioritize
long-memory imitation rather than continuing a failing PPO recipe blindly.


Automated long-memory camera-only home audit, checkpoint60, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.1328125, "insertion_rate": 0.0, "lift_rate": 0.0390625, "max_rack_contact_force_n": 7.951842784881592, "mean_peak_rack_contact_force_n": 0.06989654153585434, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.984375, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.015625}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/long_memory_evaluations/checkpoint_60_home_seed43.log`.


The camera-grasp state teacher completed800 additional updates at checkpoint2999
in4275.77 s. Its earlier2600 checkpoint remains the independently audited
candidate; a final checkpoint is not automatically better.

Long-memory checkpoint60 failed to recover pickup:17/128 grasps,5 lifts, zero
placements,2 lost vials, no contacts above20 N, peak7.95 N. Stopped the long-
memory run after checkpoint60 and its watcher; the600-update plan was not
completed. The increasing imitation fit did not translate into student task
performance. Both20/60 independent audit artifacts remain archived.

Code inspection identified an observability mismatch worth testing: the state
teacher receives the true rack-relative target even before the wrist camera
frames the rack. The existing view controller only replaces actions once the
held vial is above rack clearance; earlier actions can still depend on the
unseen rack. This could produce conflicting imitation targets for the camera
student. It is a hypothesis for the learning failure, not a demonstrated sole
cause.

Prepared a temporary observability-aware expert:
`/tmp/so101_observable_expert.py`. It substitutes a fixed nominal rack target
and matching XY-distance feature until the rack is fully framed, latches
visibility from the beginning of the episode, and uses the bank lift with pan/
gripper held before beginning the search. It never applies this lift after
insertion. It otherwise retains the symmetry-aware state teacher and bounded
labels. The camera observation contract and broad workspace are unchanged.
A native teacher-only home audit is running before training another student:
`/tmp/so101_observable_expert_eval2350_home.log`. A startup run was interrupted
to add the explicit inserted-state guard; no result from it is qualified.


The observability-aware teacher2350 native home audit completed70/128 placements
(54.6875%),107 grasps,104 lifts,77 insertions,6 lost vials, one contact above20 N
(peak37.84 N), mean successful episode19.18 s. It scanned102 episodes and framed
the rack in94. This preserves the grasp/lift level but reduces teacher success
relative to the previous view expert. Its value as a more learnable label source
still needs an independent student audit; it is not an improved task score.

A workspace failure diagnostic initially captured only87 endings because the
outer exact counter exited before the diagnostic observed the final timeout
batch. That partial file is explicitly archived as
`outputs/sim2real_audit/state2600_workspace_failures_incomplete87.json` and is
not a complete failure analysis. Corrected hook order and terminal-state
snapshot use, then repeated128 first episodes at teacher2600/seed44:
**90/128 placements**,106 grasps,104 lifts,95 insertions,4 lost vials, one contact
above20 N (peak38.77 N). Log:
`/tmp/so101_workspace_failure_eval_complete.log`. The first diagnostic run had
80/128 placements and two excessive contacts; nominally equal seeds are not
bitwise deterministic in this GPU contact simulation. Report this variability
rather than selecting only the strongest run.

The corrected128-row artifact is
`outputs/sim2real_audit/state2600_workspace_failures.json`. Successful nearest-
hole radii reached0.394 m;9 lifted-without-insertion endings had median0.261 m,
maximum0.378 m, and only1 within0.05 rad of an arm soft limit. These observations
do not support attributing the remaining failures to a narrow outer reach band.
Radius alone is not an IK proof. **No workspace restriction was added.**

Prepared `outputs/sim2real_audit/observable_frozen_pickup800_start.pt` using the
checkpoint helper with the original pickup800-derived student, teacher2350,
cleared optimizer and explicit learning rate0.0001. The temporary
`so101_observable_clone.install` combines the new expert with the verified
batched512-step update, freezes the CNN encoder and empirical normalization,
and retains the executed-command loss/BF16 training. The native16-world,
two-update smoke passed in18.72 s. Comparing checkpoint tensors confirmed all
10 encoder/normalization tensors were unchanged, while10 GRU/MLP tensors changed.
This checks the intended freeze boundary; it does not qualify task performance.

Started the128-world full-DR **home-only** clone with600 planned updates and
20-update checkpoints. Full episodes avoid ambiguous memory-free held-stage
starts; random rack/vial poses, full yaw and non-overlap are unchanged.

```bash
SO101_SEARCH_MODE=train PYTHONPATH=/tmp \
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 128 --max_iterations 600 --seed 42 \
  --run_name sim2real_observable_frozen_home \
  --checkpoint outputs/sim2real_audit/observable_frozen_pickup800_start.pt \
  --external_callback so101_observable_clone.install preset=sim2real \
  agent.student.localization_features=True agent.num_steps_per_env=512 \
  agent.algorithm.gradient_length=512 agent.algorithm.num_learning_epochs=4 \
  agent.algorithm.learning_rate=0.0001 agent.save_interval=20 \
  'env.events.reset_from_dataset.params.phase_weights=[1,0,0,0,0,0,0,0]'
```

Log: `/tmp/so101_observable_frozen_home.log`. The frozen encoder/normalization
and BF16 flags are runtime callback changes and are not fully represented by
runner YAML. Archived callback code is required for reproduction. A read-only
watcher `/tmp/so101_watch_observable.py` evaluates selected checkpoints without
the teacher controller and appends results here; artifacts are under
`outputs/sim2real_audit/observable_frozen_evaluations/`.


The observability-aware frozen-encoder run is
`logs/rsl_rl/so101_vial_camera_distillation/2026-10-07_02-41-29_sim2real_observable_frozen_home`.
With the competing jobs stopped, it runs at roughly10–11 s/update and about9 GB
GPU memory; comparisons with the prior concurrent timings are not controlled.
Started a512-world two-update throughput smoke with the same frozen recipe,
not a new task contract, to test whether the available GPU can process more
independent randomized episodes efficiently. Log:
`/tmp/so101_observable_512_smoke.log`. No substantive run is superseded yet;
its first20-update independent audit remains pending.


Automated observability-aware frozen-encoder camera-only home audit, checkpoint20, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.4375, "insertion_rate": 0.0, "lift_rate": 0.3046875, "max_rack_contact_force_n": 13.752460479736328, "mean_peak_rack_contact_force_n": 0.7026230787741952, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.953125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.046875}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/observable_frozen_evaluations/checkpoint_20_home_seed43.log`.


Automated observability-aware frozen-encoder camera-only home audit, checkpoint60, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.3046875, "insertion_rate": 0.0, "lift_rate": 0.1953125, "max_rack_contact_force_n": 15.354192733764648, "mean_peak_rack_contact_force_n": 0.3531935634673573, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.984375, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.015625}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/observable_frozen_evaluations/checkpoint_60_home_seed43.log`.


Automated observability-aware frozen-encoder camera-only home audit, checkpoint100, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.359375, "insertion_rate": 0.0, "lift_rate": 0.203125, "max_rack_contact_force_n": 33.1163330078125, "mean_peak_rack_contact_force_n": 0.7267398071126081, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.9765625, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.0234375}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/observable_frozen_evaluations/checkpoint_100_home_seed43.log`.


The512-world frozen-encoder throughput smoke passed two updates in79.7 s,
settling around32.9 s/update while the128-world run was concurrent. The128-world
run without competitors was about10.7 s/update. Larger batches provide only a
modest throughput improvement and substantially fewer optimizer updates per
minute, so the substantive experiment stayed at128 worlds.

Frozen observability-aware checkpoint20:56 grasps/39 lifts per128 home starts,
zero placements. Checkpoint60:39 grasps/25 lifts, zero placements.
Checkpoint100:46 grasps/26 lifts, zero placements,3 lost vials, one contact
above20 N (peak33.12 N). Stopped the substantive run and its watcher after the
100 audit; the600-update plan was not completed. Freezing the encoder and
normalization plus masking the unseen rack goal did not resolve the failure.
All exact checkpoint logs are under
`outputs/sim2real_audit/observable_frozen_evaluations/`.

Verified the actual untrained warm-start student in the native distillation
play path, with no teacher callback:114/128 grasps,110/128 lifts, zero placements,
no contacts above20 N (peak14.91 N), one lost vial. Thus the warm start retained
pickup in this task/loader before training. Comparing tensors with the original
PPO800 actor found all shared weights and normalization identical; only the
exploration log-standard-deviation differs and12 zero-input GRU columns/color
coordinates are added. This rules out a gross checkpoint-conversion loss of
pickup. Log: `/tmp/so101_observable_initial_student_eval.log`.

Prepared a geometry-supervision diagnostic in
`/tmp/so101_geometry_aux_clone.py`. A training-only12-output readout predicts
vial position/axis, goal error only after rack visibility, and latched progress
flags from the existing128-unit recurrent state. It adds **no actor inputs or
inference outputs**. The separate readout optimizer/state is saved outside the
student weights so the normal inference loader and final actor export remain
unchanged. The readout trains on detached memory features for20 updates before
its loss also trains the recurrent representation. Frozen encoder/normalization,
observable teacher,512-step gradients and broad full-physics randomization remain.
The auxiliary geometry coefficient is0.2; position/goal targets are scaled by2.5,
progress BCE coefficient0.25, head learning rate0.001. These are diagnostic
runtime settings, not a promoted production API.

For this first diagnostic, camera/proprioception observation corruption is
explicitly disabled to test whether the student can learn the geometric control
mapping before adding sensor uncertainty. Object placement ranges, full yaw,
non-overlap, physical/material/appearance randomization remain broad. This
intermediate recipe is **not full-domain-randomization acceptance**; restoring
observation corruption and auditing it is required before any final candidate.
Native two-update smoke log: `/tmp/so101_geometry_aux_smoke.log`.


The auxiliary native two-update smoke passed in18.17 s. Started a200-update,
128-world geometric learning diagnostic with the same full/broad physics recipe
and observation corruption disabled. Runtime callback settings are documented
above. The first20 updates train the readout on detached memory features;
subsequent updates also backpropagate its weighted loss into the recurrent state.
Logs report vial-position and visible-goal RMS error in metres, plus the fraction
of frames with a known visible goal. These are training estimates; independent
student task scores remain the qualification evidence.

```bash
SO101_SEARCH_MODE=train PYTHONPATH=/tmp \
uv run --no-sync --extra sim2real so101 train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 128 --max_iterations 200 --seed 42 \
  --run_name sim2real_geometry_aux_clean \
  --checkpoint outputs/sim2real_audit/observable_frozen_pickup800_start.pt \
  --external_callback so101_geometry_aux_clone.install preset=sim2real \
  agent.student.localization_features=True agent.num_steps_per_env=512 \
  agent.algorithm.gradient_length=512 agent.algorithm.num_learning_epochs=4 \
  agent.algorithm.learning_rate=0.0001 agent.save_interval=20 \
  'env.events.reset_from_dataset.params.phase_weights=[1,0,0,0,0,0,0,0]' \
  env.observations.wrist_rgb.enable_corruption=False \
  env.observations.proprioception.enable_corruption=False
```

Log: `/tmp/so101_geometry_aux_clean.log`. A read-only watcher audits20,60,100
and199 with only the ordinary student loader and exact home counter; no teacher
controller or auxiliary readout controls evaluation actions. Artifacts:
`outputs/sim2real_audit/geometry_aux_evaluations/`; watcher:
`/tmp/so101_watch_geometry.py`. An additional16-world loader smoke checks that
auxiliary checkpoint metadata does not require the training callback for normal
student inference: `/tmp/so101_geometry_aux_default_loader_smoke.log`.

Corrected the acceptance-counter docstrings to describe independently randomized
home starts. The earlier description of128 fixed scenes repeated8 times and
"deterministic canonical" episodes no longer described the broad workspace/
GPU contact simulation. Counter behavior and the1024-episode acceptance size
are unchanged; this is a documentation correction, not a new test requirement.


Automated geometry-supervised CLEAN-observation camera-only home audit, checkpoint20, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.5390625, "insertion_rate": 0.0, "lift_rate": 0.40625, "max_rack_contact_force_n": 8.930834770202637, "mean_peak_rack_contact_force_n": 0.3976879119873047, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.96875, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.03125}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/geometry_aux_evaluations/checkpoint_20_home_seed43.log`.


Automated geometry-supervised CLEAN-observation camera-only home audit, checkpoint60, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.4609375, "insertion_rate": 0.0, "lift_rate": 0.3125, "max_rack_contact_force_n": 11.824623107910156, "mean_peak_rack_contact_force_n": 0.6037635626271367, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.9453125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0546875}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/geometry_aux_evaluations/checkpoint_60_home_seed43.log`.


Automated geometry-supervised CLEAN-observation camera-only home audit, checkpoint100, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.2265625, "insertion_rate": 0.0, "lift_rate": 0.1015625, "max_rack_contact_force_n": 10.49048137664795, "mean_peak_rack_contact_force_n": 0.22572210431098938, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.953125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.046875}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/geometry_aux_evaluations/checkpoint_100_home_seed43.log`.


Automated geometry-supervised CLEAN-observation camera-only home audit, checkpoint199, seed43,128 first episodes: {"episodes": 128, "grasp_rate": 0.515625, "insertion_rate": 0.0, "lift_rate": 0.3359375, "max_rack_contact_force_n": 34.69538116455078, "mean_peak_rack_contact_force_n": 1.2439567595720291, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.953125, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.046875}. No expert callback or physical devices were used. Artifact: `outputs/sim2real_audit/geometry_aux_evaluations/checkpoint_199_home_seed43.log`.

### Geometry-supervision diagnostic completed; no candidate accepted (2026-10-07)

The 200-update run finished at checkpoint199 in2183.7 s. Ordinary camera-only
home audits at20/60/100/199 all produced **zero placements in128 episodes**.
Checkpoint199 grasped66/128, lifted43/128, lost6/128 vials, and had one rack
contact above20 N (peak34.70 N). Earlier checkpoints also failed placement;
reduced auxiliary loss did not establish useful geometric control. No job from
this diagnostic is still training. The auxiliary readout remains a temporary
training experiment and is not part of deployment.

Next diagnostic: collect eight first-episode trajectories of the raw state
teacher2350 on one canonical home row, with a fixed rack/vial pose and nominal
physics/appearance. Camera/proprioception corruption and diameter randomization
are disabled. This deliberately simplified diagnostic does **not** replace the
broad `preset=sim2real` workspace or qualify a final policy. Its purpose is to
check whether recurrent behavioral cloning can overfit a clean successful
sequence before further broad randomized training. `/tmp/so101_single_scene.py`
saves the selected schema-validated reset row and actor observations, bounded
teacher labels, done flags, and first-episode active masks. Collection uses only
simulation; physical devices remain unopened.

```bash
PYTHONPATH=/tmp uv run --no-sync --extra sim2real so101 play \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 8 --seed 43 \
  --checkpoint outputs/sim2real_audit/observable_frozen_pickup800_start.pt \
  --external_callback so101_single_scene.install preset=sim2real \
  agent.student.localization_features=True
```

Collection log: `/tmp/so101_single_scene_collection.log`; planned artifact:
`outputs/sim2real_audit/single_scene_teacher_trajectories.pt`.

Fixed-scene collection completed:7/8 placements (87.5%),8/8 grasps/lifts/
insertions, zero contacts above20 N (peak5.55 N), mean successful completion
7.55 s. One episode timed out. The single-row diagnostic initially rejected the
original eight-phase sampling weights; setting `phase_weights=None` correctly
samples its sole row without weakening dataset validation.

`/tmp/so101_single_scene_fit.py` performs offline full-episode recurrent cloning
on only the seven non-timeout trajectories (1585 active frames), with frozen CNN
and normalization and cached encodings. It resets recurrent state to zero for
each optimization epoch. Learning rate0.001, executed-action MSE, gradient norm
clip1,1001 epochs. Training MSE fell from0.8445 to0.00170. Checkpoints100/300/1000
are in `outputs/sim2real_audit/single_scene_fit/`. This is action-fit evidence
only, not closed-loop or generalization evidence. The first ordinary-loader
attempt rejected an optimizer parameter-group mismatch: the offline optimizer
initially excluded frozen encoder tensors. It was corrected to include all
model parameters (frozen tensors still receive no gradients), the fit repeated,
and native inference restarted. No production inference loader was relaxed.

```bash
uv run --no-sync --extra sim2real python /tmp/so101_single_scene_fit.py
SO101_SINGLE_MODE=student PYTHONPATH=/tmp \
uv run --no-sync --extra sim2real so101 play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation --num_envs 8 --seed 43 \
  --checkpoint outputs/sim2real_audit/single_scene_fit/model_1000.pt \
  --external_callback so101_single_scene.install preset=sim2real \
  agent.student.localization_features=True
```

`SO101_SINGLE_MODE=student` applies only the simplified diagnostic scene and
exact episode counter; it leaves the ordinary student policy in control and
does not collect or execute teacher actions. Logs:
`/tmp/so101_single_scene_fit.log`, `/tmp/so101_single_scene_student1000.log`.

The clean demonstration fit failed its independent closed-loop fixed-scene
trial:0/8 placements,8/8 grasps,7/8 lifts,3/8 insertions,2 lost vials, zero rack
contacts above20 N (peak11.93 N). Thus even excellent recorded-action fit is
insufficient; closed-loop states diverge from demonstrations.

Collected teacher correction labels while the student controlled the same
fixed scene (seed44). This first DAgger collection produced0/8 placements,
8/8 grasps/lifts,1/8 insertion, no lost vials and no rack contacts above20 N
(peak12.59 N). `outputs/sim2real_audit/single_scene_dagger1.pt` contains all9600
first-episode correction frames. The teacher was queried only for labels; it
never replaced student commands in this collection. The artifact's original
source string describes the teacher/scene, while this guide distinguishes
teacher-executed demonstrations from student-executed correction trajectories.

`/tmp/so101_single_scene_dagger_fit.py` aggregates the seven successful original
sequences and all student correction sequences, preserving full observation
history from home and masking padding/post-terminal frames. First DAgger fit:
3001 full-sequence optimization epochs, learning rate0.0003, frozen CNN and
normalization, same executed-action loss/gradient clipping. Outputs:
`outputs/sim2real_audit/single_scene_dagger_fit1/`; logs:
`/tmp/so101_single_scene_dagger1_collection.log`,
`/tmp/so101_single_scene_dagger1_fit.log`. This remains a fixed-scene diagnosis,
not a broadly randomized policy or a real-robot experiment.


Fixed-scene student-executed DAgger collection2, seed45,8 episodes: {"episodes": 8, "grasp_rate": 1.0, "insertion_rate": 0.125, "lift_rate": 0.875, "max_rack_contact_force_n": 14.195369720458984, "mean_peak_rack_contact_force_n": 4.440188050270081, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.75, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.25}. This evaluates checkpoint `outputs/sim2real_audit/single_scene_dagger_fit1/model_3000.pt`; teacher queries supply labels only. Log: `/tmp/so101_single_scene_dagger2_collection.log`.


Fixed-scene student-executed DAgger collection3, seed46,8 episodes: {"episodes": 8, "grasp_rate": 1.0, "insertion_rate": 0.0, "lift_rate": 0.875, "max_rack_contact_force_n": 16.58174705505371, "mean_peak_rack_contact_force_n": 2.52700412273407, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.625, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.375}. This evaluates checkpoint `outputs/sim2real_audit/single_scene_dagger_fit2/model_3000.pt`; teacher queries supply labels only. Log: `/tmp/so101_single_scene_dagger3_collection.log`.


Fixed-scene student-executed DAgger collection4, seed47,8 episodes: {"episodes": 8, "grasp_rate": 0.75, "insertion_rate": 0.5, "lift_rate": 0.75, "max_rack_contact_force_n": 13.202764511108398, "mean_peak_rack_contact_force_n": 5.387391090393066, "mean_time_to_success_s": 6.833333492279053, "success_rate": 0.125, "successes": 1, "timeout_rate": 0.625, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.25}. This evaluates checkpoint `outputs/sim2real_audit/single_scene_dagger_fit3/model_3000.pt`; teacher queries supply labels only. Log: `/tmp/so101_single_scene_dagger4_collection.log`.

Independent final state-teacher audit: checkpoint2999, seed45,128 broadly
randomized home starts, **93/128 placements (72.66%)**,115 grasps,112 lifts,
102 insertions,5 lost vials, zero rack contacts above20 N (peak18.70 N), mean
successful completion10.81 s. Log:
`/tmp/so101_camera_grasp_teacher_eval2999_home_seed45.log`. This is a privileged
state teacher score, **not** wrist-camera deployment performance. Its higher
point estimate is not proof of an improvement beyond sampling variation.

Saved diagnostic image montage:
`outputs/sim2real_audit/single_scene_teacher_views.png`, generated from the
recorded simulated wrist observations at0–9 s. In the successful first episode,
the cap occupies much of the image and the rack disappears during approach;
frames after the first episode ends show the next reset and are not part of the
successful trajectory. This supports checking remembered geometry and
closed-loop corrections; it does not prove that rack occlusion is the sole
cause of failures. Full teacher state is added to subsequent diagnostic
recordings for geometry inspection but remains excluded from camera student
inputs and offline fitting.


Fixed-scene student-executed DAgger collection5, seed48,8 episodes: {"episodes": 8, "grasp_rate": 0.875, "insertion_rate": 0.125, "lift_rate": 0.75, "max_rack_contact_force_n": 18.264812469482422, "mean_peak_rack_contact_force_n": 4.549674034118652, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 1.0, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0}. This evaluates checkpoint `outputs/sim2real_audit/single_scene_dagger_fit4/model_3000.pt`; teacher queries supply labels only. Log: `/tmp/so101_single_scene_dagger5_collection.log`.


Fixed-scene student-executed DAgger collection6, seed49,8 episodes: {"episodes": 8, "grasp_rate": 1.0, "insertion_rate": 0.25, "lift_rate": 1.0, "max_rack_contact_force_n": 27.854310989379883, "mean_peak_rack_contact_force_n": 9.239755630493164, "mean_time_to_success_s": 10.666666984558105, "success_rate": 0.125, "successes": 1, "timeout_rate": 0.75, "unsafe_rack_contact_rate": 0.125, "vial_lost_rate": 0.125}. This evaluates checkpoint `outputs/sim2real_audit/single_scene_dagger_fit5/model_3000.pt`; teacher queries supply labels only. Log: `/tmp/so101_single_scene_dagger6_collection.log`.

### Fixed-scene correction and encoder diagnostics

The bounded six-collection DAgger loop completed. Camera student scores for
collections2–6 were0/8,0/8,1/8,0/8,1/8 placements. Collection6 had one contact
above20 N (peak27.85 N), so even its single success does not qualify it. Each
subsequent fit aggregated all first-episode correction trajectories with the
seven original successful demonstrations. Final frozen-encoder aggregate:
52076 active frames,56 trajectories,3001 epochs, MSE0.00418. No broad scene or
real robot was controlled. Logs/commands/metrics are recorded by
`/tmp/so101_single_scene_dagger_loop.py` in
`outputs/sim2real_audit/single_scene_dagger{2..6}_audit.json` and guide entries.

Unfroze the existing CNN for an additional offline diagnostic,
`/tmp/so101_single_scene_cnn_fit.py`, without changing actor inputs or
architecture. Each update samples one successful original trajectory and one
student correction trajectory; losses are normalized per trajectory so long
failed rollouts do not overwhelm successful full-task demonstrations. The
normalization statistics remain fixed. Learning rate0.0001,2001 updates,
full-sequence recurrent gradients, two trajectories per update, same bounded
label loss and gradient clip1. Checkpoints300/1000/2000 are in
`outputs/sim2real_audit/single_scene_cnn_fit/`. Its reported loss is for the
sampled minibatch, not a held-out prediction error.

Independent ordinary-student fixed-scene checks: checkpoint1000 seed50,
**0/8 placements**,7 grasps/lifts,1 insertion,3 lost, peak8.43 N;
checkpoint2000 seed51, **2/8 placements (25%)**,7 grasps/lifts,4 insertions,
no lost vials or contacts above20 N, peak13.28 N. These diagnostics improve on
the original zero-success overfit check but are too weak and too narrow for
sim2real acceptance. Logs: `/tmp/so101_single_scene_cnn{1000,2000}_eval.log`.

Started a200-update PPO diagnostic from the unchanged camera actor at
CNN-fit2000, with privileged critic2999, cleared optimizer state, learning
rate0.0001 and initial exploration standard deviation0.15. The checkpoint
preparation records these sources in `infos`; no teacher controls PPO actions.
128 worlds,128 control steps/update,lambda0.99, fixed canonical reset row,
nominal0.020 kg vial, physical/material/appearance/diameter randomization and
sensor corruption disabled. This is a deliberate single-scene learning test,
**not** the final `preset=sim2real` recipe. Standard task rewards, physical
geometry, self-collision, control cadence and success criteria are preserved.
Log: `/tmp/so101_single_scene_ppo.log`; warm start:
`outputs/sim2real_audit/single_scene_cnn2000_ppo_start.pt`;
preparation script: `/tmp/so101_prepare_fixed_ppo.py`. The next independent
checks must use the student alone; success on this scene would only justify
progressing to wider randomized curricula.


Fixed-scene PPO20 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 0.9921875, "insertion_rate": 0.0234375, "lift_rate": 0.984375, "max_rack_contact_force_n": 27.980497360229492, "mean_peak_rack_contact_force_n": 3.7537430632510222, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.921875, "unsafe_rack_contact_rate": 0.03125, "vial_lost_rate": 0.078125}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_ppo_evaluations/checkpoint_20.log`. This is not broad-workspace acceptance.


Fixed-scene PPO60 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 0.984375, "insertion_rate": 0.046875, "lift_rate": 0.9765625, "max_rack_contact_force_n": 35.65800857543945, "mean_peak_rack_contact_force_n": 3.700293206376955, "mean_time_to_success_s": 20.066667556762695, "success_rate": 0.0078125, "successes": 1, "timeout_rate": 0.90625, "unsafe_rack_contact_rate": 0.0390625, "vial_lost_rate": 0.0859375}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_ppo_evaluations/checkpoint_60.log`. This is not broad-workspace acceptance.


Fixed-scene PPO100 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 0.96875, "insertion_rate": 0.390625, "lift_rate": 0.9453125, "max_rack_contact_force_n": 19.373210906982422, "mean_peak_rack_contact_force_n": 6.391402014909545, "mean_time_to_success_s": 13.366667461395263, "success_rate": 0.078125, "successes": 10, "timeout_rate": 0.7578125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.1640625}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_ppo_evaluations/checkpoint_100.log`. This is not broad-workspace acceptance.

The larger128-episode fixed-scene baseline for CNN-fit2000 produced only
**1/128 placements**,111 grasps,102 lifts,28 insertions,26 lost vials and3
contacts above20 N (peak32.32 N). The earlier2/8 result was preliminary and
has not replicated at useful reliability. Log:
`/tmp/so101_single_scene_cnn2000_baseline128.log`.

Checked offline full-sequence versus ordinary step-wise inference on218
recorded frames of the actual trained model, not only a small synthetic GRU.
Default CUDA math yielded max absolute action difference0.00219. Disabling
TF32 for both convolution and matrix multiplication reduced this to
**4.22e-6**, passing the2e-5 check. This is consistent with reduced-precision
kernel differences rather than an observation-order/recurrent batching bug.
Logs: `/tmp/so101_sequence_parity.log`,
`/tmp/so101_sequence_parity_full_precision.log`; source:
`/tmp/so101_sequence_parity.py` and saved default variant
`/tmp/so101_sequence_parity_tf32.py`. Full-precision native baseline (same
fixed scene/seed52/128 episodes) produced5 placements,119 grasps,114 lifts,
33 insertions,18 lost vials and5 contacts above20 N (peak60.04 N). It still
fails qualification; this is not evidence that precision alone solves control.
Log: `/tmp/so101_single_scene_cnn2000_fp32_eval128.log`.

Read the installed LeRobot normalization implementation while checking encoder
units: its degrees conversion deliberately uses `model_resolution - 1`, i.e.
4095 for STS3215. Current software maps/calibration-derived spans match that
installed convention. No encoder denominator or physical calibration was
changed; actual zero/scale alignment remains a measured real-robot follow-up.


Fixed-scene PPO199 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 0.921875, "insertion_rate": 0.1015625, "lift_rate": 0.90625, "max_rack_contact_force_n": 31.00832176208496, "mean_peak_rack_contact_force_n": 4.0229655049188295, "mean_time_to_success_s": 21.116667985916138, "success_rate": 0.03125, "successes": 4, "timeout_rate": 0.8828125, "unsafe_rack_contact_rate": 0.015625, "vial_lost_rate": 0.0859375}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_ppo_evaluations/checkpoint_199.log`. This is not broad-workspace acceptance.

The200-update fixed-scene PPO run completed in481.01 s. Independent128-episode
scores at20/60/100/199 were0/1/10/4 placements. Checkpoint100 was the best point
estimate at7.81%, with no contacts above20 N; final199 had two contacts above
20 N (peak31.01 N). This diagnostic did not produce a qualified policy and is
no longer training. Its ordinary source/configuration and metrics remain in
`outputs/sim2real_audit/single_scene_ppo_evaluations/`.

### Public-joint geometry feature diagnostic

Extracted the exact corrected simulated robot's seven joint frames using the
same clone-source/standalone Newton model construction as the existing reset
IK generator. Saved `outputs/sim2real_audit/robot_kinematic_frames.json` with
joint types/parents/children, local frames, axes, and measured home body/camera
poses for independent comparison. No real motor positions were read or changed.
Extraction source/log: `/tmp/so101_extract_kinematics.py` and `.log`.

`/tmp/so101_torch_fk.py` implements this six-revolute-joint chain in pure Torch.
At the measured simulated home pose, all seven body positions agree with the
native robot to5.22e-8 m; sign-invariant quaternion error1.72e-7. Camera
position error2.98e-8 m, quaternion error3.38e-7. TorchScript output agrees
exactly. This validates the simulated frame convention at home; it does not
validate the real robot's joint offsets or camera extrinsics. Log:
`/tmp/so101_torch_fk.log`.

`/tmp/so101_geometry_memory.py` adds24 features computed **only from public
joint angles and image color moments**: gripper pose, collision-pad midpoint,
camera pose, world-space blue-cap/yellow-rack viewing rays, and jaw gap. It
adds no ground-truth object pose or progress inputs. The parent CNN/128-unit GRU
architecture remains; GRU input804 expands to828. New columns are zeroed for
warm start; existing actor weights come from CNN-fit2000. The features supply
known robot geometry rather than requiring the recurrent network to learn the
entire joint-to-camera transform. This is a temporary diagnostic model, not a
promoted standard/default or a qualified camera policy.

The custom TorchScript actor initially rejected a norm overload without an
explicit `p`; setting `p=2` fixed scripting. Native-model/JIT comparison on a
recorded initial observation passes with zero action difference. Sources:
`/tmp/so101_geometry_prepare.py`; log:
`/tmp/so101_geometry_prepare.log`; warm start:
`outputs/sim2real_audit/single_scene_geometry_start.pt`. Prototype supports
TorchScript; ONNX export explicitly rejects this unvalidated diagnostic.

Started5001 offline full-sequence updates with the same seven successful
fixed-scene demonstrations and six correction collections. CNN/GRU/MLP learn,
normalization remains fixed; learning rate0.0001; each update samples one
successful demonstration and one correction trajectory with equal trajectory
weight. Source: `/tmp/so101_single_scene_geometry_fit.py`; log:
`/tmp/so101_single_scene_geometry_fit.log`; outputs:
`outputs/sim2real_audit/single_scene_geometry_fit/`. Matching simulator-only
student audits must load `so101_geometry_memory.GeometryMemoryModel` explicitly.
Even a good score here would need broad randomized training/acceptance next.

The first geometry-model playback attempts stopped before simulation playback
because the standard player automatically exports ONNX in addition to
TorchScript. Added matching functional ONNX output (actions plus next hidden
state) to the temporary prototype and reran rather than relaxing the player.
Geometry5000 ordinary camera-only fixed-scene audit, seed54,128 episodes:
**6/128 placements (4.69%)**,117 grasps,108 lifts,32 insertions,16 lost vials,
3 contacts above20 N, peak56.54 N. Geometry features are not yet evidence of a
useful control improvement. Log:
`/tmp/so101_single_scene_geometry5000_eval128.log`.

Compared the Torch FK with native gripper positions over8333 active frames from
correction collection6, including moving/grasped/insertion states. RMS position
error1.03e-7 m; maximum3.47e-7 m. Thus the features use the correct actual joint
frame/order across those trajectories, not only at home. The proposed soft-limit
observation-clipping explanation does not explain these fixed-scene failures;
no observation semantics were changed. Source/log:
`/tmp/so101_fk_observation_error.py` and `.log`.

Inspected the fixed-scene PPO optimizer and action means before another
reinforcement-learning change. All audited checkpoints20/60/100/199 had
adaptive learning rate at the floor1e-5; standard deviations stayed near0.15.
On8333 recorded correction frames, CNN-fit2000 and PPO100 mean commands
exceeded physical action bounds on roughly13–43% of frames per joint; maxima
were2.10–3.90. Commands were still safely clipped by the environment, but PPO
operates on the unbounded Gaussian mean. This differs from the earlier native
pickup-PPO200 saturation audit (lower fractions, none above3); the earlier
negative saturation finding is not revised to describe this later clone.
Source/log: `/tmp/so101_clone_saturation.py` and `.log`.

Testing a bounded-mean variant of the temporary geometry model: append a
parameter-free `Tanh` to the existing action MLP, so the Gaussian mean itself
lies in[-1,1]. Physical action/target clipping remains unchanged. This changes
the policy intentionally; existing tensor keys are compatible but its behavior
is not claimed identical to the old mean. The offline demonstration/correction
fit retrains it for3001 balanced full-sequence updates at0.0001 before any
reinforcement-learning audit. Mean bounding is a testable hypothesis for PPO
conditioning, not a proven cause or guaranteed fix. Source:
`/tmp/so101_single_scene_bounded_fit.py`; model class:
`so101_geometry_memory.BoundedMeanGeometryMemoryModel`; outputs:
`outputs/sim2real_audit/single_scene_bounded_fit/`; log:
`/tmp/so101_single_scene_bounded_fit.log`.


Fixed-scene bounded-mean geometry PPO20 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 0.84375, "insertion_rate": 0.2109375, "lift_rate": 0.7890625, "max_rack_contact_force_n": 45.65801239013672, "mean_peak_rack_contact_force_n": 6.024905964055506, "mean_time_to_success_s": 16.25555642445882, "success_rate": 0.0234375, "successes": 3, "timeout_rate": 0.8359375, "unsafe_rack_contact_rate": 0.046875, "vial_lost_rate": 0.140625}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_bounded_ppo_evaluations/checkpoint_20.log`. This is not broad-workspace acceptance.


Fixed-scene bounded-mean geometry PPO100 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 1.0, "insertion_rate": 0.1953125, "lift_rate": 1.0, "max_rack_contact_force_n": 43.16782760620117, "mean_peak_rack_contact_force_n": 5.346674619300757, "mean_time_to_success_s": 7.575000524520874, "success_rate": 0.03125, "successes": 4, "timeout_rate": 0.859375, "unsafe_rack_contact_rate": 0.0390625, "vial_lost_rate": 0.109375}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_bounded_ppo_evaluations/checkpoint_100.log`. This is not broad-workspace acceptance.

Bounded-mean offline fit3000 ordinary camera-only fixed-scene audit, seed55,
128 episodes:7 placements (5.47%),116 grasps,103 lifts,34 insertions,17 lost
vials and6 contacts above20 N (peak64.33 N). It remains unqualified. Log:
`/tmp/so101_single_scene_bounded3000_eval128.log`.

Started a400-update combined PPO-conditioning probe from that actor with
128 worlds/128 control steps, standard PPO/rewards, lambda0.99, fixed learning
rate0.00005, exploration standard deviation0.15, and frozen actor normalization
(critic normalization remains ordinary). The critic is inherited from the
prior single-scene PPO100. Geometry features, bounded mean, normalization and
learning-rate schedule are changed together; this probe cannot attribute a
result to one component in isolation. Scene/physics/sensor randomization stays
disabled only for this fixed-scene diagnostic. Production `preset=sim2real`
continues to contain broad placement/full randomization.

Source/callback: `/tmp/so101_bounded_ppo.py`; preparation:
`/tmp/so101_prepare_bounded_ppo.py`; warm start:
`outputs/sim2real_audit/single_scene_bounded_ppo_start.pt`; log:
`/tmp/so101_single_scene_bounded_ppo.log`; run:
`logs/rsl_rl/so101_vial_camera/2026-10-07_05-09-44_sim2real_single_scene_bounded_ppo`.
Read-only watcher `/tmp/so101_watch_bounded_ppo.py` audits20/100/200/399,
ordinary student only,128 first episodes, seed52; results/commands:
`outputs/sim2real_audit/single_scene_bounded_ppo_evaluations/`.


Fixed-scene bounded-mean geometry PPO200 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 1.0, "insertion_rate": 0.6796875, "lift_rate": 0.9921875, "max_rack_contact_force_n": 32.846038818359375, "mean_peak_rack_contact_force_n": 13.544743715086952, "mean_time_to_success_s": 12.8666672706604, "success_rate": 0.0625, "successes": 8, "timeout_rate": 0.8125, "unsafe_rack_contact_rate": 0.0859375, "vial_lost_rate": 0.125}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_bounded_ppo_evaluations/checkpoint_200.log`. This is not broad-workspace acceptance.


Fixed-scene bounded-mean geometry PPO399 ordinary camera-only audit, seed52,128 first episodes: {"episodes": 128, "grasp_rate": 1.0, "insertion_rate": 0.4375, "lift_rate": 1.0, "max_rack_contact_force_n": 37.58595275878906, "mean_peak_rack_contact_force_n": 11.539776653051376, "mean_time_to_success_s": 36.400001525878906, "success_rate": 0.0078125, "successes": 1, "timeout_rate": 0.9453125, "unsafe_rack_contact_rate": 0.0859375, "vial_lost_rate": 0.046875}. No teacher controls actions; no physical devices opened. Log: `outputs/sim2real_audit/single_scene_bounded_ppo_evaluations/checkpoint_399.log`. This is not broad-workspace acceptance.

The400-update bounded-mean/frozen-normalization/fixed-rate probe completed.
Its camera-only fixed-scene scores at20/100/200/399 were3/4/8/1 successes in128
trials. The final checkpoint grasped/lifted every vial, but placed only one and
had11 contacts above20 N (peak37.59 N). This combined conditioning change did
not produce a reliable policy; no continuation was launched from it.

Virtual camera-pitch framing was measured without modifying the simulated or
physical camera mount. Raw state teacher2350, seed45,128 broad starts, scored
93 placements with no contacts above20 N (peak15.90 N). Its full-rack-bounding-box
frame coverage before grasp/lifted-before-insertion/after insertion was
0%,0.0457%,0% at the present mount. Extra downward pitch10/20/30/45 degrees
changed lifted-before-insertion coverage to0.374%/1.28%/10.42%/16.46%, still
poor. These are **control-frame fractions**, not episode visibility rates;
frustum geometry ignores occlusion and actual image readability. No tilt has
been proposed as a proven fix or applied to hardware. Artifact:
`outputs/sim2real_audit/virtual_camera_pitch_framing.json`; source/log:
`/tmp/so101_virtual_view_audit.py`, `/tmp/so101_virtual_view_teacher_audit.log`.
An earlier student-pickup trajectory audit is separately preserved in
`virtual_camera_pitch_framing_pickup.json`; it is not the teacher measurement.

Testing an observable target-selection teacher in
`/tmp/so101_visible_hole_expert.py`. A visible opening needs a projected center
and at least six of eight21 mm ring samples inside the image, a yellow-ish
surround, and a darker center in the actual raw rendering. Ground-truth scene
poses are used **only in this training teacher** to query the correct pixel
location and assign labels; the camera student still receives no object pose.
After seeing a qualifying opening, the teacher latches its identity for the
episode. The corrected choice is the opening nearest the image center, making
selection depend on what can be seen rather than an unobserved nearest-vial
calculation. All four holes remain valid for physical success, and the broad
independent rack/vial workspace/rotation/non-overlap constraints are preserved.
Unknown rack goals remain masked; the active-view search continues until an
opening qualifies. No full-rack bounding box is required.

An initial prototype inherited the parent full-rack visibility OR, which could
stop search without a qualifying hole, and chose the closest visible hole by
true vial position. Its65/128 successes (seed45) are preserved as an intermediate
teacher diagnostic, not camera performance or evidence for the corrected
observable rule. Replaced that parent search path with the same bounded scan/
clearance logic using only the selected-opening latch, and changed selection
to image-center distance. Corrected independent teacher audit is running:
`/tmp/so101_visible_hole_expert_corrected_eval128.log`.

The corrected visible-opening teacher completed **71/128 (55.47%)** broad
starts, with111 grasps,104 lifts,74 insertions,2 lost vials and one contact
above20 N (peak20.93 N).108 episodes saw a qualifying opening and90 scanned.
This is privileged teacher performance, not a camera-student acceptance score.

A separate seed46 collection produced **66/128 successful demonstrations**,
103 grasps,97 lifts,69 insertions,9 lost vials and no contacts above20 N
(peak15.87 N). Saved all first-episode RGB/proprioception/action trajectories,
active/done/success masks and training-only selected-target annotations to
`outputs/sim2real_audit/broad_visible_hole_demonstrations.pt` (5.3 GB).
Source/log: `/tmp/so101_collect_visible_teacher.py` and
`/tmp/so101_collect_visible_teacher.log`. Broad placement and physical/material
randomization are enabled; play-mode observation corruption is disabled.

Measured the three existing scan postures with the actual robot FK: the held
vial axis tilts33.1/44.4/50.7 degrees from vertical. A constrained upright-pose
optimization produced candidates in `candidate_upright_search_poses.json`,
but these approach a shoulder joint limit and change camera coverage. They
have not passed native loaded collision/visibility validation and are not
adopted. No physical camera mount or robot was changed.

Started a broad-workspace camera fit from the generic pickup800 actor, rather
than the unsuccessful fixed-scene actor. It uses the66 successful independent
visible-opening demonstrations, trainable CNN/GRU, public-joint FK features,
and a bounded action mean. A training-only GRU auxiliary head predicts selected
hole XY after it becomes visible; neither those labels nor the visibility mask
are actor inputs. This is a new hypothesis, not an accepted policy. Source/log:
`/tmp/so101_broad_visible_fit.py`, `/tmp/so101_broad_visible_fit.log`; checkpoints:
`outputs/sim2real_audit/broad_visible_hole_fit/`. Independent native camera-only
rollouts must establish closed-loop performance before export or deployment.

Broad camera fit1000 independent ordinary student rollout, seed48: **0/128
placements**,35 grasps,23 lifts,16 lost vials and one contact above20 N
(peak21.76 N). Training action MSE fell to0.052 and the training-only goal-head
XY RMSE was15.8 mm on its final minibatch; neither establishes generalization.
Log: `/tmp/so101_broad_visible1000_eval128.log`. Demonstration lengths across
66 successes are170–1183 frames (mean568.17). Full200-frame native sequential,
batched-sequence and TorchScript parity passed with maximum action differences
6.41e-7 and2.38e-7 after disabling TF32; test source/log:
`/tmp/so101_broad_geometry_parity.py`/`.log`. The first parity attempt omitted
moving the exported module's CPU hidden buffer to CUDA; corrected the test
with `.to('cuda')` before scripting. This was a test-device mismatch.

Extended the offline fit to20000 minibatch updates to check convergence before
another independent rollout. No production deployment or physical robot motion.
Source/log: `/tmp/so101_broad_visible_fit.py`,
`/tmp/so101_broad_visible_fit_extended.log`; original1000-update source preserved
as `/tmp/so101_broad_visible_fit1000.py`.

The20000-update broad fit completed in157.3 seconds; final minibatch action
MSE0.01744 and selected-goal XY RMSE17.57 mm. Intermediate sampled minibatches
reached lower errors; these are training errors only. The2000 checkpoint's
independent seed49 audit is running. Source/loss log:
`/tmp/so101_broad_visible_fit_extended.log`.

Started a larger independent512-world visible-target teacher collection,
seed50, broad full physical/material/appearance randomization. New collector
`/tmp/so101_collect_visible_teacher512.py` stores RGB as uint8 (rounded from
render RGB ×255) and retains successful first episodes; training must divide
by255. This matches the real camera byte representation and limits disk/RAM
cost. Original128-world float corpus remains intact. New output:
`outputs/sim2real_audit/broad_visible_hole_demonstrations512_uint8.pt`; log:
`/tmp/so101_collect_visible_teacher512.log`. Disk check found9.8 GB free, so
no larger float-image corpus is being written; no unrelated files deleted.

Broad fit2000 seed49: **0/128 placements**,44 grasps,25 lifts,10 lost vials,
no contacts above20 N (peak10.18 N). Broad fit20000 seed51: **0/128 placements**,
45 grasps,31 lifts,15 lost vials,one contact above20 N (peak20.52 N). More updates
on the66-example corpus did not solve closed-loop control. Logs:
`/tmp/so101_broad_visible2000_eval128.log`,
`/tmp/so101_broad_visible20000_eval128.log`.

Larger visible-target teacher collection completed: **309/512 (60.35%)**
placements,440 grasps,421 lifts,340 insertions,17 lost vials,one contact above
20 N (peak25.34 N). This is a privileged teacher score. The uint8 dataset contains
309 successful trajectories, not all512 trials. Started a separate40000-update
camera fit from generic pickup800 using this larger corpus. CNN learning rate
0.00001,remaining actor0.0001, action loss weighted3× for the first180 frames
and training-only selected-goal XY auxiliary weight0.2. These combined changes
are a convergence/coverage probe, not an isolated ablation. Source/log:
`/tmp/so101_broad_visible_fit512.py`/`.log`; checkpoints:
`outputs/sim2real_audit/broad_visible_hole_fit512/`. The auxiliary goal/visibility
annotations remain outside actor inputs. Independent native rollouts pending.


Larger309-trajectory camera fit2000, independent seed52,128 broad
home first episodes, camera/proprioception actor only: `{"episodes": 128, "grasp_rate": 0.3125, "insertion_rate": 0.0078125, "lift_rate": 0.171875, "max_rack_contact_force_n": 32.03921127319336, "mean_peak_rack_contact_force_n": 0.7062012739479542, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.9453125, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.0546875}`.
Log/command: `outputs/sim2real_audit/broad_visible512_evaluations/checkpoint_2000.log`
and `.json`. Return code0. This is simulation evidence only; physical alignment
and full observation-corruption acceptance remain pending.


Larger309-trajectory camera fit10000, independent seed53,128 broad
home first episodes, camera/proprioception actor only: `{"episodes": 128, "grasp_rate": 0.4765625, "insertion_rate": 0.015625, "lift_rate": 0.3828125, "max_rack_contact_force_n": 35.600013732910156, "mean_peak_rack_contact_force_n": 1.0582223245874047, "mean_time_to_success_s": 33.06666946411133, "success_rate": 0.0078125, "successes": 1, "timeout_rate": 0.9453125, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.046875}`.
Log/command: `outputs/sim2real_audit/broad_visible512_evaluations/checkpoint_10000.log`
and `.json`. Return code0. This is simulation evidence only; physical alignment
and full observation-corruption acceptance remain pending.


Larger309-trajectory camera fit20000, independent seed54,128 broad
home first episodes, camera/proprioception actor only: `{"episodes": 128, "grasp_rate": 0.5234375, "insertion_rate": 0.015625, "lift_rate": 0.390625, "max_rack_contact_force_n": 25.367794036865234, "mean_peak_rack_contact_force_n": 0.9212751928716898, "mean_time_to_success_s": 16.666667938232422, "success_rate": 0.0078125, "successes": 1, "timeout_rate": 0.921875, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.0703125}`.
Log/command: `outputs/sim2real_audit/broad_visible512_evaluations/checkpoint_20000.log`
and `.json`. Return code0. This is simulation evidence only; physical alignment
and full observation-corruption acceptance remain pending.


Larger309-trajectory camera fit40000, independent seed55,128 broad
home first episodes, camera/proprioception actor only: `{"episodes": 128, "grasp_rate": 0.5390625, "insertion_rate": 0.0234375, "lift_rate": 0.390625, "max_rack_contact_force_n": 16.61143684387207, "mean_peak_rack_contact_force_n": 0.9919170687207952, "mean_time_to_success_s": 28.200002670288086, "success_rate": 0.015625, "successes": 2, "timeout_rate": 0.8828125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.1015625}`.
Log/command: `outputs/sim2real_audit/broad_visible512_evaluations/checkpoint_40000.log`
and `.json`. Return code0. This is simulation evidence only; physical alignment
and full observation-corruption acceptance remain pending.

The larger-corpus40000-update fit finished in324.2 seconds, final minibatch
MSE0.00948,selected-goal XY RMSE8.71 mm. Independent camera-only seed55
rollout: **2/128 (1.56%) placements**,69 grasps,50 lifts,3 insertions,13 lost
vials,no contacts above20 N (peak16.61 N). Checkpoint10000/20000 each scored
1/128 on separate seeds53/54. None meets acceptance.

Started1000 additional ordinary PPO updates with256 worlds,128 control steps,
full `preset=sim2real` physical/appearance/sensor randomization and uninterrupted
home-start episodes. Warm actor is the309-demo fit20000 (selected before the
40000 audit finished), privileged critic from state2999; actor remains camera/
proprioception only. Exploration std0.15, fixed LR0.00005,lambda0.99,actor
normalization frozen. Saved checkpoint100/300/600/999 audits use128 independent
first home episodes, seed57; observation corruption is disabled for those
initial nominal-camera diagnostics. No export/deployment acceptance implied.
Run:
`logs/rsl_rl/so101_vial_camera/2026-10-07_06-31-33_sim2real_broad_visible_camera_ppo`.
Preparation/callback: `/tmp/so101_prepare_broad_camera_ppo.py`,
`/tmp/so101_bounded_ppo.py`; training log:
`/tmp/so101_broad_visible_camera_ppo.log`; watcher:
`/tmp/so101_watch_broad_camera_ppo.py`; audit results/commands:
`outputs/sim2real_audit/broad_camera_ppo_evaluations/`. Training has started,
loaded weights successfully and produced finite optimization updates.

Reproduce the current PPO launch from the tutorial directory (preparation
creates the warm-start file; keep the archived prototype modules on PYTHONPATH):

```bash
PYTHONPATH=/tmp uv run --no-sync --extra sim2real python /tmp/so101_prepare_broad_camera_ppo.py
PYTHONPATH=/tmp uv run --no-sync --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 256 --max_iterations 1000 --seed 56 \
  --checkpoint outputs/sim2real_audit/broad_camera_ppo_start.pt \
  --external_callback so101_bounded_ppo.install preset=sim2real \
  agent.actor.class_name=so101_geometry_memory.BoundedMeanGeometryMemoryModel \
  agent.actor.localization_features=True agent.num_steps_per_env=128 \
  agent.save_interval=100 agent.algorithm.lam=0.99 \
  agent.algorithm.learning_rate=0.00005 agent.algorithm.schedule=fixed \
  agent.run_name=sim2real_broad_visible_camera_ppo
```

These commands have already been run; do not start a duplicate job. Prototype
sources/logs are archived in `outputs/sim2real_audit/diagnostics/` with SHA256
manifest. For reproduction after `/tmp` cleanup, use that directory in
PYTHONPATH and the equivalent archived preparation/callback paths. Geometry
metadata remains `outputs/sim2real_audit/robot_kinematic_frames.json`.
Documentation spelling check (`codespell` guide and README) passed. No new
production behavior was changed during these offline fitting/PPO prototypes;
the prior77-test software validation remains applicable.


Broad camera PPO100, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.6484375, "insertion_rate": 0.0, "lift_rate": 0.484375, "max_rack_contact_force_n": 15.239940643310547, "mean_peak_rack_contact_force_n": 0.5103632733225822, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.9375, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0625}`. Log/command:
`outputs/sim2real_audit/broad_camera_ppo_evaluations/checkpoint_100.log`
and `.json`. Return code0; no physical devices opened.

Read-only opening-visibility audit of broad camera PPO100, separate seed58:
1/128 placements,73 grasps,56 lifts,no contacts above20 N (peak12.04 N).
Only38/128 episodes ever had a qualifying opening in view, and13 had one
visible after lifting. It was visible in1340/42588 lifted control frames (3.15%).
Actions and rewards were unchanged; ground-truth projection was diagnostic
only. This supports investigating active-view learning, rather than treating
poor performance solely as insufficient training duration. Source/log:
`/tmp/so101_camera_opening_audit.py`/`/tmp/so101_camera_ppo100_opening_audit.log`.
The visibility test uses raw RGB ring/center contrast and projection; it is
not a perfect segmentation-based occlusion proof.

Started a16-world/two-update smoke of a training-only information milestone:
first qualifying currently visible opening while physically holding a lifted
vial, before insertion, pays3 reward units once per episode (weight90 ×1/30 s).
`ManagerTermBase.reset` clears only the reset worlds' paid flags. It uses scene
poses only to query corresponding raw image pixels for the reward, never as
actor observations or control commands. Physical success/termination/contact
penalties are unchanged; no production config has been modified. Source/log:
`/tmp/so101_camera_view_reward.py`, `/tmp/so101_camera_view_reward_smoke.log`.
This is an unvalidated active-view reward prototype, not a proven fix. The
original1000-update baseline continues independently.

Information-reward smoke completed two finite PPO updates in16.77 seconds.
The short run did not exercise a positive opening reward, so it only establishes
launch/optimization compatibility. Started a separate16-episode causal-target
teacher rollout to exercise positive cases and assert at most one positive
view-reward event per first episode. The first audit failed because configclass
captures the derived distillation config's post-init method: patching only the
camera base post-init left the extra reward absent in the derived task. Fixed
the prototype installer to explicitly patch both camera and camera-distillation
config classes. Original failure log preserved:
`/tmp/so101_view_reward_teacher_audit.log`; corrected log:
`/tmp/so101_view_reward_teacher_audit_corrected.log`. Production classes unchanged.


Broad camera PPO300, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.859375, "insertion_rate": 0.0078125, "lift_rate": 0.7890625, "max_rack_contact_force_n": 23.84504508972168, "mean_peak_rack_contact_force_n": 0.6465760469436646, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.8984375, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.1015625}`. Log/command:
`outputs/sim2real_audit/broad_camera_ppo_evaluations/checkpoint_300.log`
and `.json`. Return code0; no physical devices opened.

Corrected16-episode teacher validation exercised13 positive view-reward
cases, with **at most one payment per first episode**; assertion passed. Teacher
placed14/16 (not a student score), with one contact above20 N (peak20.69 N).
This establishes positive reward delivery and reset compatibility; it does
not prove image detection precision or learning benefit.

Baseline camera PPO300 independent seed57: **0/128 placements**,110 grasps,
101 lifts,one insertion,13 lost vials,one contact above20 N (peak23.85 N).
Pickup has improved but the downstream behavior remains unresolved.

Started a separate500-update information-milestone PPO branch from baseline
PPO300. It retains actor/critic and optimizer state, resets run iteration to0,
and adds only the validated once-per-episode3-unit view reward. Full physical,
appearance and sensor DR remain enabled;256 worlds,128 steps,fixed LR0.00005,
lambda0.99,actor normalization frozen. Independent seed61 differs from the
baseline seed56, so this is not a controlled one-variable ablation. Baseline
1000-update run continues separately. New warm start:
`outputs/sim2real_audit/camera_view_reward_ppo_start.pt`; log:
`/tmp/so101_camera_information_ppo.log`; source callback:
`/tmp/so101_camera_view_reward.py`. Actor observations and physical success
criteria remain unchanged. No real robot motion or deployment qualification.

Information-milestone run:
`logs/rsl_rl/so101_vial_camera/2026-10-07_06-49-50_sim2real_camera_information_milestone`.
Read-only watcher `/tmp/so101_watch_camera_information.py` evaluates100/300/499
with ordinary camera-only control, broad first128 home episodes, seed57. It
also measures qualifying-opening visibility without changing actions or
rewards. Results/commands:
`outputs/sim2real_audit/camera_information_evaluations/`.
The reward detector uses uncorrupted rendering while the actor trains with
sensor corruption; this is a shaping proxy, not proof that every rewarded
opening is readable in the actor's delayed/blurred/affine-augmented input.
Full corrupted-camera acceptance remains required. The bonus is experimental
and has not been added to production defaults or the sim2real preset.


Information-milestone camera PPO100, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.8125, "insertion_rate": 0.0078125, "lift_rate": 0.7578125, "max_rack_contact_force_n": 28.066926956176758, "mean_peak_rack_contact_force_n": 0.7169432435184717, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.859375, "unsafe_rack_contact_rate": 0.015625, "vial_lost_rate": 0.140625}`. Log/command:
`outputs/sim2real_audit/camera_information_evaluations/checkpoint_100.log`
and `.json`. Opening-visibility diagnostic: `{"any_visible_opening_episodes": 50, "episodes": 128, "lifted_frames": 90045, "visible_opening_after_lift_episodes": 38, "visible_opening_lifted_frames": 1759}`. Return code0; no physical devices opened.

Saved and visually inspected eight raw rewarded-opening frames at
`outputs/sim2real_audit/view_reward_debug/rewarded_openings.png`; marked centers
lie inside visible dark rack openings in these samples. This does not establish
precision across all randomized scenes. Instrumented repeat teacher audit,
same seed60,scored12/16 (previous14/16),11 view-reward cases,at most one payment
per first episode,no contacts above20 N (peak18.76 N). GPU contact variation is
visible even at a fixed seed; these are not camera-policy scores. Source/log:
`/tmp/so101_inspect_view_reward.py`/`.log`.

Collected a sparse observer-training dataset with103890 active first-episode
frames,including27795 held/lifted frames. Inputs are63 **public** features:
proprioception, image color moments, exact joint-derived FK and an approximate
blue-cap ellipse/depth feature using known nominal17.7 mm cap radius. Targets
are root-local vial position/axis and contact/lift labels from simulation only.
No scene object pose enters the features. Collection teacher seed62 scored
65/128 (50.78%),104 grasps,98 lifts,74 insertions,7 lost vials,no contacts above
20 N (peak17.63 N). Source/log:
`/tmp/so101_collect_pose_features.py`/`.log`; feature code:
`/tmp/so101_vision_geometry.py`; dataset:
`outputs/sim2real_audit/broad_held_pose_features.pt`.

Started a small geometric observer fit using96 training episodes and32
**episode-disjoint** validation episodes. Root/axis loss is restricted to
held cases with a sufficiently visible blue cap; contact/lift classification
uses all frames. This investigates whether explicit camera geometry can
estimate the held vial accurately enough for a controller; it is not a
closed-loop policy and has not been integrated or deployed. Source/log:
`/tmp/so101_fit_pose_observer.py`/`.log`; checkpoints:
`outputs/sim2real_audit/held_pose_observer/`. Real intrinsics, extrinsics and
joint mapping remain unverified.


Broad camera PPO600, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.921875, "insertion_rate": 0.015625, "lift_rate": 0.859375, "max_rack_contact_force_n": 4.854423522949219, "mean_peak_rack_contact_force_n": 0.18202342465519905, "mean_time_to_success_s": 7.600000381469727, "success_rate": 0.0078125, "successes": 1, "timeout_rate": 0.90625, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0859375}`. Log/command:
`outputs/sim2real_audit/broad_camera_ppo_evaluations/checkpoint_600.log`
and `.json`. Return code0; no physical devices opened.


Information-milestone camera PPO300, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.84375, "insertion_rate": 0.0, "lift_rate": 0.8203125, "max_rack_contact_force_n": 29.101346969604492, "mean_peak_rack_contact_force_n": 1.1743728639557958, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.859375, "unsafe_rack_contact_rate": 0.015625, "vial_lost_rate": 0.140625}`. Log/command:
`outputs/sim2real_audit/camera_information_evaluations/checkpoint_300.log`
and `.json`. Opening-visibility diagnostic: `{"any_visible_opening_episodes": 75, "episodes": 128, "lifted_frames": 97188, "visible_opening_after_lift_episodes": 70, "visible_opening_lifted_frames": 9655}`. Return code0; no physical devices opened.


Information-milestone camera PPO499, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.9375, "insertion_rate": 0.0, "lift_rate": 0.9296875, "max_rack_contact_force_n": 14.870397567749023, "mean_peak_rack_contact_force_n": 0.5071176290512085, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.953125, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.046875}`. Log/command:
`outputs/sim2real_audit/camera_information_evaluations/checkpoint_499.log`
and `.json`. Opening-visibility diagnostic: `{"any_visible_opening_episodes": 74, "episodes": 128, "lifted_frames": 121740, "visible_opening_after_lift_episodes": 69, "visible_opening_lifted_frames": 20938}`. Return code0; no physical devices opened.

User explicitly confirmed **accept any empty hole** in the four-hole rack.
The existing single-vial physical success check already accepts any of the
four openings; no occupied-hole model is needed while this task contains only
one vial and an initially empty rack. This replaces the earlier provisional
assumption with an explicit preference. No designated-hole marker is required.


Broad camera PPO999, ordinary camera-only actor,128 independent first
episodes, seed57: `{"episodes": 128, "grasp_rate": 0.8984375, "insertion_rate": 0.0, "lift_rate": 0.8671875, "max_rack_contact_force_n": 41.691734313964844, "mean_peak_rack_contact_force_n": 0.7935357561800629, "mean_time_to_success_s": null, "success_rate": 0.0, "successes": 0, "timeout_rate": 0.890625, "unsafe_rack_contact_rate": 0.015625, "vial_lost_rate": 0.109375}`. Log/command:
`outputs/sim2real_audit/broad_camera_ppo_evaluations/checkpoint_999.log`
and `.json`. Return code0; no physical devices opened.


### Restricted rack rotation and camera bottom-right target — 2026-10-07

The user's latest instruction **replaces the previous any-empty-hole preference**:
limit rack rotation and consistently use the camera's bottom-right opening.
The initial `preset=sim2real` rack yaw interval is **−15° to +15° about world Z**, centred
on the asset's default zero yaw. This is a proposed training envelope, not a measured
optimal interval. Broad independent rack/vial placement stays at radius 0.16–0.38 m,
sector −100° to +100°, with table, robot and mutual collision rejection. Vial heading
and axial spin remain unrestricted. The default preset retains its original behavior.

Operational definition: when all four opening centres first project inside the wrist
camera (2 px margin; positive depth), choose the rightmost of the lower two centres.
Latch that **physical opening** until the next reset. The view may occur while finding
or approaching the rack. It is not necessarily the initial image, because the rack can
start outside the wrist field of view. An optional clarification was sent about this
reference view; pending a reply, this is the stated assumption. Before a target is
selected, shaping points toward the rack centre and insertion/success cannot be paid.
Once selected, reward, critic/teacher target error, insertion and physical success all
use the same opening. The state/image observation dimensions and hardware action
interface are unchanged. Late insertion/release replay retains the physical opening
already occupied by its aligned vial; camera home-start evaluation does not use that
exception.

The selection uses simulated projected geometry to define supervision. It **does not
provide hole coordinates to the camera actor**, prove an unoccluded RGB view, or serve
as a detector for the physical camera. Restricting rack yaw does not remove the need to
search across the broad positional workspace. The saved simulation overlays are a
geometry check, not a deployment qualification.

Modified files: `mdp/workspace.py` (explicit rack yaw range), `mdp/events.py`
(range forwarding and per-reset target state), `config/so101/sim2real_cfg.py`
(preset values), `mdp/geometry.py` (image-coordinate selection), and `mdp/terms.py`
(latched target shared by supervision and success). The existing measured wrist-camera
transform is now shared between the initial-vial visibility filter and target projection.
Focused geometry/workspace/reward/config checks passed **30 tests**. An initially
over-tight stochastic endpoint assertion failed, then was corrected to check 5% of the
sampling interval; the full-yaw default retains its original RNG draw sequence.
Native gate command:

```bash
PYTHONPATH=/tmp uv run --no-sync --extra sim2real so101 play \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
  --num_envs 16 --seed 65 \
  --checkpoint outputs/sim2real_audit/observable_frozen_pickup800_start.pt \
  --external_callback so101_camera_corner_audit.install \
  preset=sim2real agent.student.localization_features=True
```

This uses a **privileged teacher with scripted search**, only to validate the new task
geometry before training. It is not the deployable camera actor. Log:
`outputs/sim2real_audit/diagnostics/so101_camera_corner_audit.log`; annotated frames:
`outputs/sim2real_audit/camera_corner_gate/`. Do not duplicate the running command.

#### Final results from the preceding any-hole experiments

- Broad camera PPO999: **0/128 placements**, 115 grasps, 111 lifts; 2 episodes over
  20 N, peak 41.69 N. PPO600 had 1/128 placements. No complete-task acceptance.
- Information-milestone PPO499: **0/128 placements**, 120 grasps, 119 lifts;
  no contacts over 20 N, peak 14.87 N. Openings were visible after lift in 69/128
  episodes and 17.2% of lifted frames. More viewing did not solve placement.
- The corrected held-vial observer has 23.10 mm root RMSE (95th 54.02 mm).
  Adding a small RGB CNN produced 26.88 mm RMSE (95th 68.18 mm) on held-out
  episodes; median 5.31 mm still hides large outliers. Neither observer is
  accurate enough for insertion and neither is integrated into hardware control.
  The first observer labels had failed to subtract vectorized environment origins;
  that training-data bug was corrected and the bad artifacts explicitly retained
  as `broad_held_pose_features_uncorrected.pt`/`held_pose_observer_uncorrected/`.
  RGB dataset: `broad_held_pose_images.pt`, 34,917 contact frames, actual seed64.
- Public RGB hole detector: after fixing the top plane to rack root +73 mm and
  comparing yellow hue with the brown background, the partial step1000 diagnostic
  had 1,261 detections, median 13.59 mm error, 95th 44.45 mm; 510 detections
  (40.4%) were over15 mm. Frames are correlated and this is not an independent
  episode success rate. The detector remains unqualified and is not used on hardware.
- Dynamic camera FK audit across teacher motion (through step1000): maximum
  position error **4.03e−7 m**, quaternion sign-invariant L2 **7.44e−7**,
  intrinsics difference **0** against the native camera sensor. This rules out a
  nominal FK/sensor-pose mismatch as the cause of these observer errors in this
  measured simulation audit; physical intrinsics/extrinsics remain unverified.

All results above precede the bottom-right target change. No policy has been exported
or described as a complete-task sim2real-ready policy on the strength of pickup alone.


Camera-bottom-right state teacher baseline, seed67,128 independent home-start episodes:
`{"episodes": 128, "grasp_rate": 0.8828125, "insertion_rate": 0.046875, "lift_rate": 0.84375, "max_rack_contact_force_n": 15.66420841217041, "mean_peak_rack_contact_force_n": 6.306583207449876, "mean_time_to_success_s": 15.760000801086425, "success_rate": 0.0390625, "successes": 5, "timeout_rate": 0.890625, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0703125}`. Command/hash/log in
`outputs/sim2real_audit/camera_corner_teacher_evaluations/checkpoint_baseline.json`
and `.log`. This is privileged-state simulation evidence, not camera-policy or physical acceptance.


Native camera-corner gate finished: **6/16 placements**, 14 grasps, 11 lifts,
6 insertions, one vial lost; no rack contacts over20 N, peak9.71 N. This
small privileged-teacher/search check demonstrates that the new target can be
reached, not that a deployable policy has learned it. Inspected
`camera_corner_gate/world_9_selected_1.png`: red marker identifies the lower
right opening in that camera image. Other frames are retained in the same folder.
Full software validation: **80 tests passed**,42 TorchScript deprecation warnings;
Ruff, codespell and `git diff --check` passed after the change.

Teacher adaptation is running from teacher2999, with 500 additional updates,
512 environments,128 steps/update, seed66, lambda0.99, fixed learning rate1e−4,
initial action std0.1, frozen actor normalization and cleared optimizer moments.
It includes dataset phase curriculum; its training success rate is therefore **not**
a home-start qualification. Independent128-home-start audits (seed67) run for the
old teacher baseline and adapted checkpoints100/300/499. Each JSON records the
checkpoint hash and exact command. Run:
`logs/rsl_rl/so101_vial_state/2026-10-07_07-59-23_sim2real_camera_bottom_right_teacher/`.

```bash
PYTHONPATH=/tmp uv run --no-sync --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 512 --max_iterations 500 --seed 66 \
  --checkpoint outputs/sim2real_audit/camera_corner_teacher_start.pt \
  --external_callback so101_bounded_ppo.install preset=sim2real \
  agent.num_steps_per_env=128 agent.save_interval=100 \
  agent.algorithm.lam=.99 agent.algorithm.learning_rate=.0001 \
  agent.algorithm.schedule=fixed agent.run_name=sim2real_camera_bottom_right_teacher
```

Do not launch a duplicate. `so101_prepare_corner_teacher.py` records checkpoint
preparation; training log is `so101_camera_corner_teacher.log`; audit orchestrator
is `so101_watch_corner_teacher.py`/`.log`, archived under
`outputs/sim2real_audit/diagnostics/`. The first launch incorrectly named the
state task with a `-State` suffix and failed before simulation. The actual
registered state task is suffixless, as corrected above; failure log retained as
`so101_camera_corner_teacher_wrong_task.log`. No physical arm was connected.

The wrist-camera policy will require fresh complete-task audits for this target
contract before export or real trials. All earlier any-hole scores are retained
as historical evidence and cannot be presented as scores for the revised task.


Camera-bottom-right state teacher 100, seed67,128 independent home-start episodes:
`{"episodes": 128, "grasp_rate": 0.7890625, "insertion_rate": 0.0234375, "lift_rate": 0.7265625, "max_rack_contact_force_n": 39.17052459716797, "mean_peak_rack_contact_force_n": 5.931404561037198, "mean_time_to_success_s": 13.100001017252604, "success_rate": 0.0234375, "successes": 3, "timeout_rate": 0.9296875, "unsafe_rack_contact_rate": 0.0078125, "vial_lost_rate": 0.046875}`. Command/hash/log in
`outputs/sim2real_audit/camera_corner_teacher_evaluations/checkpoint_100.json`
and `.log`. This is privileged-state simulation evidence, not camera-policy or physical acceptance.

### Consolidated progress report — 2026-10-07 08:12 PDT

Hardware/software: shared uv sim2real extra and `.venv` configured; Isaac Lab pinned
at the checked upstream develop revision including image/history LEAPP fix. Both arms
calibrated and teleoperation confirmed by the user. Wrong supply resolved. Focused
Sonix wrist camera image saved and inspected. Orange robot, yellow rack and brown
tabletop are scene defaults, outside the randomization preset.

Simulation alignment: workshop elbow correction, actual follower gripper span,
28.9 mm vial body/35.4 mm cap, matching visual/collision dimensions, diameter DR,
three-primitive vial contacts, reduced rolling/torsional friction, control cadence,
relative joint-target tracking and self-collision implemented. Domain randomization
covers the documented dynamics, contact, actuator, visual and sensor variations;
broad independent collision-filtered placements remain enabled and initial vial starts
are camera-frustum filtered. The latest rack yaw interval is −15°…+15° and the selected
bottom-right physical hole is held fixed after the first complete centre projection.
These settings are implemented; seamless real-world transfer has not been established.

Training: the historical any-hole state teacher reached93/128 home-start placements.
The camera branch reached120 grasps and119 lifts out of128 in its information-milestone
checkpoint, but zero placements. The larger309-successful-demonstration camera fit
managed only2/128 placements. The release-only clone reached123/128; this is a stage
score and cannot be compared with complete home-start placement. Prototype held-vial
observers and RGB hole detectors failed insertion-accuracy gates and were not deployed.
The simulation camera FK audit agrees with the native sensor through dynamic motion;
physical camera geometry is still unverified.

The new bottom-right task has a5/128 baseline with the old teacher. Adaptation100
scored3/128, with one rack contact above20 N (peak39.17 N), so no improvement is claimed.
At this report the512-environment adaptation run had reached update271/500, with
roughly11 minutes estimated for training alone. Independent128-home-start checkpoint
300/499 audits remain pending. Audits append their results to this guide automatically.

Deployment: LEAPP smoke/release-stage packages and their numerical checks exist.
The hardware inference bridge reads wrist RGB and calibrated joint state, supports
saved-image/debug captures and guarded dry runs, and rejects autonomous execution
while the joint map is unverified. No complete-task camera policy is accepted,
no complete-task policy is claimed ready for export/deployment, and no autonomous
physical motion has been run. Remaining work is reliable complete visual placement,
independent randomized acceptance and full-policy export checks, followed by physical
joint mapping, camera/rack alignment, loop timing and supervised robot trials.

Validation:80 software tests passed; Ruff, codespell and diff whitespace checks passed.
Exact experiments, failed attempts, fixes, paths, checkpoint hashes and commands are
retained in this guide and `outputs/sim2real_audit/`.


Camera-bottom-right state teacher 300, seed67,128 independent home-start episodes:
`{"episodes": 128, "grasp_rate": 0.8359375, "insertion_rate": 0.0390625, "lift_rate": 0.7890625, "max_rack_contact_force_n": 13.918502807617188, "mean_peak_rack_contact_force_n": 5.612992583075538, "mean_time_to_success_s": 11.880000495910645, "success_rate": 0.0390625, "successes": 5, "timeout_rate": 0.9296875, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.03125}`. Command/hash/log in
`outputs/sim2real_audit/camera_corner_teacher_evaluations/checkpoint_300.json`
and `.log`. This is privileged-state simulation evidence, not camera-policy or physical acceptance.

### Branch handoff for multi-GPU experiments — 2026-10-07

The user authorized committing and pushing the current tutorial changes to a branch
for experiments on a multi-GPU computer. Branch: `feat/so101-sim2real-multigpu`, remote:
`git@github.com:isaac-sim/IsaacLabTutorial.git`. This publishes the task code,
configuration, small corrected reset dataset, dependency lock, hardware inference
scripts, tests and guides. Logs, checkpoints, demonstration datasets, camera captures
and temporary diagnostic prototypes remain local and ignored by Git.

Added `docs/MULTIGPU_EXPERIMENTS.md` with fresh-clone installation, a single-GPU
smoke run, distributed-launch dry run, per-rank environment counts and a1024-episode
camera checkpoint audit. `so101 train_multigpu` forwards `preset=sim2real` correctly
as `presets=sim2real` and the upstream launcher injects `--distributed`. A dry run
with one rank succeeded; two ranks were correctly rejected because this machine
has only one visible GPU. Real multi-rank initialization, NCCL and throughput are
not validated here. `uv lock --check --offline` passed. README links the handoff
and removes the unsupported `agent.resume=True` example; explicit checkpoint loading
is the supported resume path. Current local training continues independently.

The handoff is an experiment baseline, not an accepted complete-task policy.
Historical helper commands referring to `/tmp` are retained as experiment history;
the portable baseline recipes depend only on committed code. Model artifacts must
be transferred separately if a compatible checkpoint is to be resumed.


The user subsequently requested a **new branch** and artifact cleanup before push.
Created `feat/so101-sim2real-multigpu`; the clone command in the multi-GPU handoff
uses this name. Generated Python/test/lint caches are removed after validation.
Ignored experiment outputs and the active training run are preserved locally and
excluded from the commit; no model artifacts or temporary helpers are published.
The repository pre-commit hooks initially flagged the existing `isinstance(value,
(int, float))` expression in the reset generator. It now uses `int | float`,
with the Boolean exclusion and finite-value semantics unchanged.

Final pre-push checks: all repository pre-commit hooks (Ruff check/format,
codespell) passed; the full suite passed80 tests in4.34s with42 existing
TorchScript deprecation warnings. `uv lock --check --offline` passed.
Generated test/lint/Python cache directories were cleaned. The staged set contains
only source, configuration, tests, README/guides, dependency metadata and the96.7kB
required reset dataset; no training checkpoints, captures, logs or scratch prototypes.
The current adaptation checkpoint300 scored5/128 complete placements (seed67),
no rack contacts over20 N, peak13.92 N: it matched the new-task baseline and did
not qualify the camera policy. Training and the final audit remain local.

Branch publication completed: implementation commit
`ac757e2e197267fa3c276f480d7e4e6bba50069a` was pushed successfully to
`origin/feat/so101-sim2real-multigpu`, and `git ls-remote` verified that exact
commit on GitHub. The branch tracks its remote and the working tree was clean
at verification. The portable handoff is `docs/MULTIGPU_EXPERIMENTS.md`.
No pull request was created; this is the requested experiment branch.


Camera-bottom-right state teacher 499, seed67,128 independent home-start episodes:
`{"episodes": 128, "grasp_rate": 0.7109375, "insertion_rate": 0.0390625, "lift_rate": 0.6796875, "max_rack_contact_force_n": 15.410750389099121, "mean_peak_rack_contact_force_n": 4.954314163886011, "mean_time_to_success_s": 13.180000686645508, "success_rate": 0.0390625, "successes": 5, "timeout_rate": 0.8984375, "unsafe_rack_contact_rate": 0.0, "vial_lost_rate": 0.0625}`. Command/hash/log in
`outputs/sim2real_audit/camera_corner_teacher_evaluations/checkpoint_499.json`
and `.log`. This is privileged-state simulation evidence, not camera-policy or physical acceptance.

### Local vision-policy work resumed after branch handoff — 2026-10-07

The user requested continued experiments on this local GPU toward a high-success
wrist-camera policy. The published branch remains the handoff baseline; additional
local changes will be tracked separately. The bottom-right state-teacher adaptation
finished500 updates, and checkpoint499 scored5/128 complete placements (seed67),
with no rack contacts over20 N. No improvement over the5/128 baseline is claimed.

A fresh first-episode audit at seed68 found **17/128 targets acquired**,99 grasps,
95 lifts and12 episodes with both lift and target acquisition; complete placement
was5/128. Log: `/tmp/so101_corner_failure_audit.log`; diagnostic callback:
`/tmp/so101_corner_failure_audit.py`. This points to target acquisition as a major
bottleneck, independently of the camera learner: even the privileged teacher rarely
gets a defined target. The all-four-centres-in-frame requirement was introduced by
the assistant, not explicitly requested by the user. An ablation now keeps the
bottom-right image-coordinate selection and episode latch, but permits selection
when the projected rack centre enters the frame (all centre depths must remain
positive and projections finite). Same checkpoint, seed68 and128 home episodes;
only the visibility gate changes. Callback/log:
`/tmp/so101_corner_centre_gate_audit.py`/`.log`. This is a diagnostic task change,
not a policy improvement or a deployment claim. Its result must be measured before
changing the production task or launching further expensive training.

The user asked whether a specific hole or flexible any-hole placement is easier
to train. Recommendation: **accept any empty hole**, but keep the controller's
chosen visible/reachable hole fixed during each placement attempt. Flexible success
expands the valid terminal states; a consistent per-attempt target keeps supervision
stable. A designated physical hole can be visually ambiguous in a symmetric rack,
while a camera bottom-right rule is observable but adds a constraint unnecessary
when all empty holes are equally useful. Earlier any-hole visual runs still failed
complete placement, so changing this rule is not sufficient by itself. The current
home-only bottom-right teacher run continues; a preference question was sent for
the next experiments, and production success is not yet switched back to any-hole.

The relaxed projected-grid-centre gate ablation completed: **40/128 targets
acquired**,97 grasps,90 lifts,30 episodes with both lift and target acquisition,
16/128 placements,2 rack contacts over20 N, peak24.52 N. Matched old full-view gate
at the same checkpoint/seed gave17 targets and5 placements. The local production
gate was updated accordingly; its image-coordinate and episode-latch tests were
extended for partial views. This remains separate from the published handoff snapshot.

Home-only teacher training launched from teacher2999 with cleared optimizer state,
fixed LR1e−4, std0.2, frozen actor normalization,512 environments,128 steps/update,
1000 updates, seed72. Reward prototype pays3 units once per episode for acquiring
a target while physically holding the lifted vial. Actor observations and physical
success are unchanged. Callback/preparation/log:
`/tmp/so101_corner_home_training.py`,
`/tmp/so101_prepare_corner_home_teacher.py`,
`/tmp/so101_corner_home_teacher.log`.
Run: `logs/rsl_rl/so101_vial_state/2026-10-07_08-45-40_sim2real_corner_home_teacher/`.
The first centre-gate diagnostic failed before stepping because `argsort` required
`dim=` with `stable=True`; corrected and rerun. Failure log retained as
`/tmp/so101_corner_centre_gate_initial_error.log`.


### Any empty hole restored and findings handoff — 2026-10-07

User selected any empty hole, superseding the camera-bottom-right requirement. Physical seating and insertion
now test the nearest physical opening. Teacher/reward supervision independently latches one projected in-frame
opening near the image centre until reset. Rack yaw remains ±15°; broad positional placement remains enabled.
Goal selection waits until episode step 2 because initial camera frames can be zero. Projection is privileged
training geometry, not a real-image detector or an occlusion guarantee.

The matched checkpoint 499/seed 68 visibility ablation increased target acquisition 17→40/128 and placement 5→16/128.
The subsequent any-hole teacher 2999 audit, seed 73, placed 90/128 (70.3%), grasped 121 and lifted 116; peak rack force
was 28.94 N with one episode above 20 N. Different checkpoint/seed prevents attributing the full increase to the
goal rule. That audit predates the final two-step latch delay. Software validation:80 tests passed in 4.51 s.

The obsolete bottom-right home-only training run was interrupted after iteration 883/1000. No training is
currently running. The any-hole source changes remain local and have not been pushed to the published branch.
README and multi-GPU instructions now describe the current local contract; earlier log entries remain historical.

User requested findings to compare with another agent's solution. Created [SIM2REAL_FINDINGS.md](SIM2REAL_FINDINGS.md)
with controlled results, perception/data errors, retained simulation fixes, exact evaluation distinctions and
remaining physical validation. No complete wrist-camera policy from this attempt has passed acceptance.


## 2026-10-08 — Consolidate the local and multi-GPU branches

Preserved all local edits in `560ff59` (`archive/so101-local-before-consolidation-20261008`).
Started `feat/so101-consolidated-sim2real` from fetched multi-GPU commit `86167f0`.
The user selected the proven placement setup first, with broad workspace expansion later.
Measured asset corrections are being combined with the multi-GPU camera fixes and fresh-training recipe.
See [CONSOLIDATION.md](sim2real/CONSOLIDATION.md) for the decision table and running results.

## 11. Current supervised real trial — 2026-10-08

The user requested trying the current vision checkpoint despite its result being below the original
90% simulation acceptance gate. This is an experimental supervised trial, not a qualified deployment.
The checkpoint scored **879/1,024 (85.84%)** on a fresh clean simulated audit. One episode reached
20.16 N rack contact. Pickup is the main observed failure stage. It has not run on the physical arm.

### Prepared files and checks

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

### Terminal and uv environment

Run the following commands from the tutorial repository. `uv run --script` uses each hardware
script's inline dependency pins (LeRobot 0.6.1 and CPU Torch 2.10), without modifying the Isaac Lab
training environment. Do not add `--extra sim2real` to these hardware commands: that project extra
now installs simulation-side LEAPP only, not LeRobot. No manual virtual-environment activation is needed.

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
```

Your account already belongs to `dialout`. If this terminal's `id -nG` output does not include it,
run `newgrp dialout` once, then continue in that shell. This replaces the `sg dialout -c` wrappers used
by the assistant's older shell; no repeat `usermod` or permission change is needed.

Set the confirmed device paths and prepared bundle:

```bash
export SO101_FOLLOWER_PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00'
export SO101_CAMERA='/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0'
export SO101_TRIAL_DIR="$PWD/outputs/consolidation_20261008/supervised_trial"
```

The follower currently resolves to `/dev/ttyACM1`; the leader resolves to `/dev/ttyACM0`.
Use the stable paths above rather than the earlier session's reversed ACM assignments.

### 11.1. Match the physical setup

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

### 11.2. Motor-driven home and physical pose verification

Use the motors to home; manual joint positioning is no longer the normal procedure. First move the
rack, vial and other objects out of the arm's swept path. This interpolates joint targets, not a
collision-planned Cartesian path. Observe the first move and keep access to motor power.

Preview from the current measured position (no motor writes):

```bash
uv run --script src/isaaclab_tutorial/utils/home_so101.py \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT"
```

Run the same command with `--execute` to move:

```bash
uv run --script src/isaaclab_tutorial/utils/home_so101.py \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT" --execute
```

The command seeds goals from current encoders before enabling torque, then interpolates at up to
**5°/s per joint**. After reaching the nominal target, it allows one second to settle, then applies
a feedback trim at up to 0.5°/s, capped at ±5° from nominal and clipped to common hardware/simulation
travel. This corrects small static actuator offsets without changing servo gains or calibration.
It checks calibration identity, target limits, raw encoder travel, 100 ms feedback stalls and 10° tracking error. The final pose must remain within 2° for 0.5 seconds. A four-tick
(0.35°) encoder endpoint tolerance accommodates tiny deviations from recorded endpoints; destination
limits are not widened. A closed gripper can start outside the simulation soft limits provided its
encoder remains within calibrated travel plus that endpoint tolerance.

**On success, torque stays enabled and the arm holds home.** The process exits and releases the
serial port so you can run inspection or the policy next. Support the arm during initial torque
configuration. Ctrl+C, lost feedback or excessive tracking error disables torque and the arm may
fall; keep a clear supported resting area and use the power switch for unexpected motion. A settling
timeout after the home target has been sent, with fresh feedback and error no larger than 10°, now
holds the measured pose and reports **home not confirmed** instead of releasing the arm. Both this
bounded-timeout hold and successful homing leave motors powered until another controller or power-off. Do not move
its joints by hand while it is holding.

This supervised setup move may use the provisional joint map to help verify it. It does not mark
that map verified or enable autonomous policy execution. Compare the held pose with the reference
image, then put the rack and vial back in the reference arrangement. Read the pose if needed:

```bash
uv run --script src/isaaclab_tutorial/utils/inspect_so101.py \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT" --watch
```

Press Ctrl+C to finish inspection; inspection leaves torque unchanged.

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

### 11.3. Read-only inference

After arranging the scene, this command can be run without enabling motor writes:

```bash
uv run --script src/isaaclab_tutorial/utils/deploy.py \
  --bundle "$SO101_TRIAL_DIR/leapp/leapp.yaml" \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT" --camera "$SO101_CAMERA" \
  --duration 20
```

It uses the real wrist camera, two-frame raw RGB history, joint feedback, and exported policy, then
prints timing statistics. There are no motor configuration, torque or target writes in this mode.

### 11.4. First motion, after physical mapping verification

Once the map is physically checked and marked verified, start with a five-second supervised trial:

```bash
uv run --script src/isaaclab_tutorial/utils/deploy.py \
  --bundle "$SO101_TRIAL_DIR/leapp/leapp.yaml" \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT" --camera "$SO101_CAMERA" \
  --duration 5 --execute
```

The deployment command checks calibration identity, common joint limits and the home pose before enabling motion.
It initializes motor goals to the measured pose before configuring/enabling the motors. There is no
automatic homing move. The controller uses the trained 30 Hz policy and 120 Hz measured-relative
targets; this is **full trained action scale**, not a slow-motion mode. A five-second trial is too short
to expect task completion (simulated successes average about 14.7 seconds).

Stay at the power switch with the arm's path clear. Ctrl+C stops the loop; software cannot guarantee
an instantaneous physical stop, so use the power switch if motion is wrong. On normal completion,
Ctrl+C or a caught exception, cleanup disables torque and the arm may drop. Arrange a clear padded
resting area and support it once motion has stopped, keeping fingers out of joints and jaws.

Review approach direction, collisions, gripper alignment and timing before continuing. To attempt a
complete placement, run the homing command again with the path clear, reset the objects to the verified
starting arrangement, then:

```bash
uv run --script src/isaaclab_tutorial/utils/deploy.py \
  --bundle "$SO101_TRIAL_DIR/leapp/leapp.yaml" \
  --joint-map "$SO101_TRIAL_DIR/joint_map.json" \
  --start-pose "$SO101_TRIAL_DIR/start_pose.json" \
  --port "$SO101_FOLLOWER_PORT" --camera "$SO101_CAMERA" \
  --duration 30 --execute
```

The real controller has no task-success detector and will continue until its time limit or interruption;
stop it after a successful placement. Recheck reported deadlines after the first motion trial because
read-only timings exclude goal writes. Record each attempt's outcome, failure stage and video when
available in this guide. No real success rate is claimed until those trials have been performed.

### Reproducing the export

The artifact directory is local and ignored by Git; the Python deployment scripts and this guide are tracked.
For a new export, choose an empty output directory and run:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m isaaclab_tutorial.utils.export_leapp \
  --model logs/rsl_rl/so101_vial_camera/2026-10-08_16-40-56_consolidated_visual_ppo_4/exported/policy.pt \
  --output /absolute/path/to/new/leapp
```

### Motor-driven homing implementation and checks — 2026-10-08

The user requested motor-driven home instead of manual positioning. Added `home_so101.py` using
the same isolated CPU uv environment. A read-only preview on the connected arm reports approximately
15.9 seconds at the default 5°/s from its current folded pose. Its elbow encoder was two ticks beyond
the recorded endpoint; added a bounded four-tick measurement tolerance while retaining exact target
limits. Motor movement was not executed by the agent. Mock-bus tests cover bounded setpoints, tracking
failure, endpoint tolerance, goal-before-torque initialization, no writes during preview and retained
torque after success. Physical homing and visual mapping verification remain user-observed steps.

### Homing port troubleshooting — 2026-10-08

The user reported a missing port. Both USB serial devices were present; the follower's stable
`5AE6079843-if00` path resolved to `/dev/ttyACM1`. A fresh read-only homing preview using that full
path successfully read all motors, without changing torque or sending targets. An unset terminal
variable or an outdated port argument is a possible explanation; the exact user-side error was not
provided, so this cause is not confirmed. Use this self-contained command from the repository,
without relying on earlier shell exports:

```bash
cd /home/mhaiderbhai/code/IsaacLabTutorial
uv run --script src/isaaclab_tutorial/utils/home_so101.py \
  --joint-map outputs/consolidation_20261008/supervised_trial/joint_map.json \
  --start-pose outputs/consolidation_20261008/supervised_trial/start_pose.json \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00
```

This previews only. Add `--execute` after clearing the arm's path to perform the supervised home.
If the error is instead `Permission denied`, check `id -nG` and use `newgrp dialout` if needed.
If the explicit path still fails, retain the exact error so missing-device, permission and motor-bus
failures can be distinguished.

### Confirmed terminal permissions issue — 2026-10-08

The user's full traceback identifies `SerialException: [Errno 13] Permission denied` on the correct
follower port. LeRobot wraps this with a generic missing-port suggestion; port discovery is not the
remedy here. Verified device permissions are `crw-rw---- root dialout`. The user account already belongs
to `dialout`, but the existing session has not picked up that supplementary group. `sg dialout` obtains
the group successfully, explaining why the agent's read-only checks worked.

In the user's terminal, run:

```bash
newgrp dialout
id -nG
```

Confirm `dialout` appears, then rerun the explicit-path `uv run --script ...home_so101.py` command in
that same shell. No additional `usermod`, device chmod or root-owned uv environment is needed. Logging
out of the desktop session and back in refreshes membership for future terminals as well; simply
opening a terminal from the old desktop session may retain the old groups. No motor commands were
sent while diagnosing this permissions failure.

### First physical homing attempt: settling timeout — 2026-10-08

The user executed homing, observed the arm close to the reference home, and then saw it go limp
with `Timed out reaching home within 2 degrees`. Read-only inspection afterward confirmed every
motor's Goal_Position was at its requested home value (within encoder quantization), and all
Torque_Enable registers were zero. The old error handler had disabled torque on timeout. The
subsequent resting positions cannot identify the joint error at the instant of timeout.

Fixed the missing diagnostics and that specific timeout behavior. Homing now prints named errors
once per second and saves target, measured angles, error history and result to
`outputs/so101_homing/latest.json` (override with `--report`). If all home targets were sent and
feedback remains fresh with errors within 10°, a settling timeout sends the measured positions as
hold targets, leaves torque enabled and explicitly reports `not_home_holding_measured_pose`.
It does not claim successful home, widen the 2° acceptance tolerance or bypass policy startup checks.
Tracking faults, bad feedback and Ctrl+C still release torque. Motor gains are unchanged pending
measurement of the residual error. Seven mock-bus tests pass, including steady-offset timeout,
reported offending joint, retained hold and torque release on a true tracking fault.

Readback evidence: `outputs/consolidation_20261008/supervised_trial/homing_timeout_readback.json`.
The next supervised retry uses the same explicit-path uv homing command; its automatic report will
identify which joint needs investigation. No new motion was commanded by the agent during this fix.

### Elbow steady-state homing error and bounded correction — 2026-10-08

The next user-observed run completed the commanded trajectory but remained at an elbow error of
**-2.30°** for approximately ten seconds. Other errors were pan -0.18°, lift +0.39°, wrist flex +0.12°,
wrist roll -0.60° and gripper +0.33°. It correctly reported home unconfirmed and kept torque enabled.
Archived the report at `outputs/consolidation_20261008/supervised_trial/homing_elbow_offset_before_trim.json`.

Read-only register checks confirmed all motors still enabled, P=16, I=0, D=32 and clockwise/
counterclockwise dead zones of one encoder tick. The elbow carried a nonzero load register reading.
These observations are consistent with a load-dependent static position error, but do not prove its
cause or verify the kinematic calibration. Post-timeout measurements were around 20.88° elbow after
switching its hold goal to 20.18°; those values are not the original trajectory's terminal measurement.
Raw registers are saved in `supervised_trial/homing_gains_readback.json`.

Added homing-only feedback trim after a one-second settling period: integrate 0.5 times the measured
angle error, limit adjustment to 0.5°/s, stop integrating within 0.5° error, and cap command offset at
±5° within the original hardware/simulation limits. The target home, 2° acceptance threshold,
10° tracking fault limit, gains and calibration remain unchanged. Success is based on measured
position, not the offset command. Diagnostic reports now include the commanded angle and trim.
Unresolved bounded settling timeouts still hold the measured pose without claiming success.

Nine homing tests pass, including correction of a simulated 2.29° static actuator error, rate/offset/
travel bounds, unresolved timeout hold and tracking-fault release. No corrective motor motion was
executed by the agent. The user can rerun the same homing command while observing; physical success
of this correction remains pending that retry.

### Motor-driven home confirmed; scene verification next — 2026-10-08

The user reran homing with feedback trim and received `Home reached`. The report records maximum
residual error 1.9466° at the elbow; other joints are within 0.60°. Elbow command trim was -0.7755°.
A subsequent independent read-only joint check confirms the held pose remains at those angles and
the saved calibration matches the motors. Archived the successful report at
`outputs/consolidation_20261008/supervised_trial/homing_success.json`.

Captured `supervised_trial/real_camera_at_home.png` without torque changes. The focused 640×480 view
shows the yellow rack toward the upper left, the vial near the center, and the jaws at the bottom.
Compared it with the nominal simulated wrist reference; the broad scene orientation is consistent,
but this alone does not verify joint zeros, gripper opening or exact rack/base placement. Saved the
joint readback as `supervised_trial/hardware_at_home.json`.

Next user step: leave the arm holding home and provide an external view including the complete arm,
base, rack and vial. Compare it with the simulation overview before marking the map verified and
running the prepared five-second policy trial. No policy rollout has occurred yet.
