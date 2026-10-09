# Sim2real coverage and the WowRobo wrist camera

> This report describes the multi-GPU source branch. The measured-model consolidation and new
> qualification status are recorded in [CONSOLIDATION.md](CONSOLIDATION.md).

The deployment target is the user’s **orange WowRobo SO-101, yellow rack and wooden desk with a gray mat**,
with the WowRobo wrist camera. The original workshop’s yellow robot and green mat were incorrect.
Deployment will use LEAPP with a custom inference script and LeRobot for robot control. Source-simulator
success is necessary, but does not establish successful transfer: historical frozen policies
lost substantial performance when moved to PhysX. No real-robot success rate has been measured.
The historical selected visual policy passed Newton audits at **94.43% clean / 94.04% noisy**;
see [RESULTS.md](RESULTS.md). Those scores do not qualify the expanded Transfer profile below.
Earlier camera diagnostics below predate the self-occlusion fix.

## October 8 multi-GPU transfer audit

The current physical trials in `../SO101_SIM2REAL.md` report unreliable pickup and no
confirmed full placement. Timing passed, but synchronized camera/joint trajectories were
not recorded, so visual localization, calibration and contact errors remain competing causes.
The appearance-only fine-tune scored 28.9% on broad colors; historical >90% scores do not
qualify that distribution or the corrected physical model.

New paired tasks `-Sim2Real-Transfer` and `-Camera-Sim2Real-Transfer` retain the existing
Sim2Real and Appearance tasks for comparison. Both new tasks share these physical settings:

| Uncertainty | Transfer profile | Audit finding |
| --- | --- | --- |
| Vial mass/inertia | 12–30 g; existing mass randomizer | Retained; body/cap diameters retain construction-time ±0.5 mm variation. |
| Vial contact | Friction 0.2–1.3, restitution 0–0.02 | Newton has one friction coefficient, not separate static/dynamic friction. |
| Jaw contact | Both jaw links, friction 0.2–1.3 | Newly varied. With maximum friction combining, fixed 0.7 jaws previously masked slippery vial samples. |
| Rack/support contact | Rack 0.2–1.0; support 0.3–1.3 | Newly varied; effective pair friction still uses the maximum, not the sampled vial value alone. |
| Table/mat rolling resistance | Rolling coefficient 0.0002–0.002 m; torsional 0.001–0.005 m, sampled per world at construction | Surface contacts use six contact dimensions; jaw/vial and rack contacts retain three. Engineering ranges pending physical identification. |
| Joint Coulomb friction / armature | All six joints ×0.6–2.5 / ×0.7–1.5 | Gripper now included. Multipliers do not create a nonzero property from a zero nominal value. |
| Joint viscous friction | All six joints ×0.6–3.5 | Gripper now included; draws use startup values, avoiding reset-to-reset accumulation. |
| Arm stiffness / damping | ×0.85–1.15 / ×0.7–1.5 | Retained. |
| Gripper stiffness / damping | ×0.6–1.7 / ×0.7–1.5 | Damping newly varied. |
| Command response | Per-joint gain ×0.9–1.1; shared arm/jaw delay 0 or 1 policy steps | 25% of episodes use 33.3 ms command delay; queues clear on partial resets. This is a bounded uncertainty assumption, not measured latency identification. |
| Calibration residual | Episode-constant ±0.01 rad, shared between visual actor position and target feedback | Added alongside existing per-step noise; teacher/critic retain true state. Does not replace physical joint-map checks. |
| Vial position / heading | Existing dataset and ±20 mm XY; extra ±15° heading | Bounded local expansion; not arbitrary-workspace or arbitrary-heading training. |
| Rack placement | ±5 mm XY and ±5° yaw | Added; overlap rejection uses the sampled rack frame. |
| Mat/support height | 0–4 mm above the 35 mm desk top, 390 mm square gray pad | Covers the reported 2–3 mm mat estimate; rigid support, not identified mat compliance. Rack/vial move together. |

The transfer profile defaults to home-only training after teacher bootstrap. Optional downstream
curriculum rows keep their validated poses and a flush support surface. The reset sampler checks
rack overlap before writing states. An exhausted random draw tries bounded offsets with the
original rack/heading; it raises instead of accepting an overlapping fallback. This matters because
some regenerated home rows fail the conservative footprint check even before added jitter.
The grasp proof compensates for support height; insertion and success retain the original rack-local
criteria and ten-step stability requirement. No success tolerance was enlarged.

Vision retains broad independent robot/rack/vial/body/cap/label colors, raw RGB and two-frame history,
mount ±3 mm/±3°, focal scale 0.95–1.05, principal point ±1.5 px, radial distortion ±0.04, gamma
0.85–1.15, exposure, white balance, blur and pixel/proprioceptive noise. Mount/projection/materials,
physics, command delay and persistent calibration residual remain active in clean play. Noisy audits
also enable per-step image/proprioceptive corruption. The gray pad is fixed in color; desk colors
still vary. Newton does not provide transparent-plastic optics, randomized wood texture or material
roughness here. These remain explicit visual gaps, not silently claimed coverage.

Gravity, robot link lengths, solver integration/compliance and broad workspace placement remain fixed.
They should be varied only with a measured discrepancy or a controlled sensitivity experiment;
unbounded simultaneous randomization can destroy grasp learning without improving transfer.
The default task uses the minimal robot collision model; the selected policy below was trained
and qualified with all robot colliders retained. This profile does not establish hardware readiness.

Validation artifacts live in `outputs/robustness_20261008_multigpu/`. The 32-environment native
Newton probe confirmed nonzero variation in vial/jaw/rack/support friction, height, command delay
and encoder bias. A follow-up probe confirmed nonzero stiffness, damping, armature and friction
variation on every joint, including the gripper; inspected wrist images show the rack, vial and jaws. The software suite passed
130 tests in the training environment; the six offline deployment tests also passed separately
in the pinned Torch 2.10 CPU environment. No hardware connections were opened.
The historical camera policy scored 1/256 on an initial expanded-scene diagnostic before the final
encoder-bias/reset-fallback changes; that number is not a final-profile qualification.

Four independent processes are explicitly scoped with `CUDA_VISIBLE_DEVICES`. Initial runs used
fresh state bootstraps on GPUs 0/1, historical-teacher continuation on GPU 2, and historical-vision
continuation on GPU 3. Subsequent phases trained state teachers, distilled visual students and
refined visual PPO independently. The final comparison uses GPU 0 for visual continuation and
GPU 2 for conservative full-collider visual continuation; other GPUs run frozen audits.
These are independent experiments, not distributed PPO. The campaign runner records
commands, source hashes, checkpoint paths, exact first-episode audits and stage failures. Its target
is at least 95% complete placement on development tests, followed by 1,024-episode independent
qualification seeds (four clean/noisy tests for vision). No new policy is qualified at launch.

### Qualified Transfer vision policy — 2026-10-09

Selected `authority_full_vision_168`, block 2, checkpoint 994. Both training and qualification
retain **all robot colliders**, broad appearance/physical randomization, home starts and the
original 30-second placement criterion. The frozen actor receives only two wrist RGB frames
and 24 proprioceptive values. Its teacher/critic state is not a deployment input.

| Independent audit | Seed | Placements / 1,024 | Success | Episodes above 20 N | Peak rack force |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 1168020 | 987 | 96.39% | 2 | 22.15 N |
| Noisy | 1168021 | 981 | 95.80% | 6 | 36.71 N |
| Clean confirmation | 1168022 | 992 | 96.88% | 3 | 28.18 N |
| Noisy confirmation | 1168023 | 988 | 96.48% | 0 | 17.58 N |

Two further tests of the selected checkpoint retained full physics, camera geometry and robot
colliders while disabling color events to use the nominal scene appearance. Clean seed 17801
scored **982/1,024 (95.90%)**; noisy seed 17802 scored **980/1,024 (95.70%)**. Their peaks
were 28.90 N and 25.62 N, with 4 and 3 episodes above 20 N. These supplementary tests are
stored in `selected_vision/nominal_clean/` and `selected_vision/nominal_noise/`.

All four audits pass the predeclared 95% gate. Combined rates are **96.63% clean**, **96.14%
noisy**, and **96.39% overall (3,948/4,096)**. These are measured simulator outcomes, not a
real-robot success rate. Eleven episodes exceeded the 20 N contact diagnostic threshold;
success qualification is not a contact-safety certification. Mean successful completion times
were 14.33–14.41 seconds. Final training processes were stopped after selection.

Artifacts are in `outputs/robustness_20261008_multigpu/selected_vision/`:

- `checkpoint.pt`: selected PPO weights, SHA-256
  `631a6383b48d7c8d831eb2a7b7916353ec53dee4b352b095a589e8075814ede1`.
- `policy.pt`: exported camera/proprioception actor.
- `leapp/leapp.yaml`, `leapp/visual_actor.pt`, `leapp/contract.json`: portable CPU bundle.
- `block_02_qualification_*.json`, `campaign.json`, `manifest.json`: episode outcomes,
  exact commands, GPU scope, checkpoint/source hashes and selection record.
- `cpu_parity.json`: Torch 2.10.0+cpu / LEAPP 0.7.1, 32 varied inputs, maximum absolute
  difference **0.0** against the training-environment actor. No hardware connection was opened.

The contract uses arm scales `[0.033, 0.040, 0.033, 0.033, 0.033]` and jaw scale `0.020`.
The existing deployment controller reads these values from the bundle. Keep raw RGB / 255,
two-frame history, 30 Hz policy inference and 120 Hz measured-relative target updates.
For a subsequent supervised physical trial, replace the existing command's `--bundle` with
this `leapp/leapp.yaml` and retain the checked hardware calibration/camera setup. The larger
shoulder command is an explicit evaluated control change, not a new joint zero or calibration.

Reproduce the selected full-collider simulation audit, with exactly one physical GPU exposed:

```bash
SELECTED=outputs/robustness_20261008_multigpu/selected_vision
CUDA_VISIBLE_DEVICES=0 SO101_EVALUATION_EPISODES=1024 \
  SO101_EVALUATION_OUTPUT="$PWD/transfer_replay.json" \
  uv run --no-sync isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real-Transfer \
  --agent rsl_rl_ppo_cfg_entry_point --checkpoint "$SELECTED/checkpoint.pt" \
  --num_envs 1024 --seed 1168020 --visualizer none \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  presets=newton_mjwarp,newton_renderer \
  env.actions.arm_action.scale.shoulder_lift=0.04 \
  env.scene.robot.spawn.func=isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg:_spawn_so101_for_wrist_camera
```

For noisy playback append `env.observations.wrist_rgb.enable_corruption=True` and
`env.observations.proprioception.enable_corruption=True`. GPU simulation may show small
run-to-run numerical differences. The engineering randomization ranges and visual limitations
above remain applicable to the selected policy.

### Contact diagnostic and retraining evidence (2026-10-09)

The initial expanded-profile teachers plateaued at 48.8–56.3% home-start placement. The historical
vision continuation scored only 2/256 clean and 1/256 noisy episodes after 200 updates. Those
campaigns were stopped and retained as evidence; none is a deployment candidate.

An idle-physics probe moved the robot out of reach and tracked 64 vials for ten seconds. The
three-dimensional sliding-only contact model ignored authored rolling/torsional resistance:
53/64 vials on the mat moved more than 1 cm, and the bare-table median displacement was 39 cm.
The Transfer tasks now enable six-dimensional contacts only on table/mat surfaces and clear
otherwise inactive rolling coefficients on other shapes to avoid maximum-pair mixing overriding
the intended small surface values. This reduced mat median displacement to 0.13 mm, with 3/64
moving more than 1 cm. This is a controlled simulation diagnostic, not measured real-material
identification. Artifacts: `probe_support.py`, `support_drift_rolling.json` under the artifact root.

Frozen weights from three saved state teachers were then audited on the corrected profile:

| Teacher | Successful placements / 512 | Grasp rate | Lift rate | Lost vials |
| --- | --- | --- | --- | --- |
| Fresh 142 | 403 (78.71%) | 93.95% | 85.55% | 0.59% |
| Fresh 143 | 429 (83.79%) | 94.73% | 88.28% | 0% |
| Warm 144 | 445 (86.91%) | 98.83% | 91.80% | 0% |

These use seed 14901 and are development audits, not final qualification. They precede a
subsequent action-gain correction: the policy command is now clipped before multiplying by
the sampled gain, preserving gains above one even when the policy command saturates. The best teacher had
no rack contacts above 20 N (peak 16.59 N). Outcomes are in `rolling_teacher_142.json` through
`rolling_teacher_144.json`. New state runs compare conservative home-only updates, faster
home-only updates, and a mixed curriculum. The optional broader reset curriculum contains
1,472 validated rows and preserves all original home rows; it adds approach/held poses from
an expanded heading distribution. Audits always use the packaged home distribution plus Transfer
randomization. Visual distillation begins from the 86.91% teacher while state refinement runs;
teacher-assisted rollout scores must not be presented as visual policy success.

Subsequent development audits retain the corrected command gain:

| Candidate | Audit | Placements | Note |
| --- | --- | --- | --- |
| Warm state +200 updates, seed 154 | 256 home starts | 226/256 (88.28%) | Full Transfer physics |
| Same teacher, full robot colliders | 256 home starts, seed 15401 | 224/256 (87.50%) | No rack contacts above 20 N; peak 16.36 N |
| Low-noise state +200, seed 156 | 256 home starts | 230/256 (89.84%) | Initial action std 0.08, entropy coefficient zero |
| Visual distillation 400 updates, seed 155 | Clean / noisy, 256 each | 102/256 (39.84%) / 91/256 (35.55%) | Teacher-free first-episode evaluation |
| Visual PPO +200, seed 157 | Clean / noisy, 256 each | 135/256 (52.73%) / 154/256 (60.16%) | Independent development seeds, not paired noise ablations |

These are intermediate, unqualified candidates. The full-collider check retains disabled robot
self-collision, matching the existing task. Further state refinement, fresh corrected-profile state
training, home-start visual PPO and mixed-curriculum visual distillation run on separate explicitly
scoped GPUs. Final policy selection requires the qualification audits described above.

### Actuation-authority diagnostic

The original 0.033 rad shoulder-lift command scale leaves difficult high-friction samples. For the
best saved teacher, the upper half of shoulder-lift friction samples succeeded 81.2% versus 98.4%
in the lower half. Other joints did not show this separation. This motivates a control-authority
experiment, not a claim that the physical arm's friction has been measured again.

Controlled 256-episode diagnostic runs use seed 15601 and unchanged weights. The direct diagnostic
baseline has the same recorded initial rack pose, support height, delay, bias, stiffness and
friction samples as the earlier CLI audit; small solver outcome differences remain.

| Frozen policy / diagnostic | Placement | Lift | Peak rack force | Episodes above 20 N |
| --- | --- | --- | --- | --- |
| Teacher, original scale / full friction | 226/256 (88.28%) | 92.58% | 18.14 N | 0 |
| Teacher, shoulder scale 0.040 | 237/256 (92.58%) | 97.66% | 18.85 N | 0 |
| Teacher, shoulder scale 0.045 | 239/256 (93.36%) | 97.66% | 25.07 N | 3 |
| Teacher, friction multipliers capped at 1.8 | 249/256 (97.27%) | 98.83% | 18.12 N | 0 |
| Vision, original scale / full friction | 193/256 (75.39%) | 81.25% | 20.88 N | 1 |
| Vision, shoulder scale 0.040 | 204/256 (79.69%) | 93.36% | 19.26 N | 0 |
| Vision, shoulder scale 0.045 | 203/256 (79.30%) | 93.75% | 23.14 N | 1 |

The narrower-friction diagnostic is **not** qualification of the full distribution. New teacher and
vision branches retain the full 0.6–2.5 friction multipliers and use the smaller 0.040 shoulder scale.
The default Transfer task remains at 0.033; the experiment is explicit through
`train_transfer --shoulder-scale 0.04`, which applies the same override to training and every audit.
Direct playback uses `env.actions.arm_action.scale.shoulder_lift=0.04`. All other arm scales remain
0.033 and the gripper remains 0.020. The 30 Hz policy / 120 Hz feedback cadence is retained.

Export a selected shoulder-authority actor with
`--action-scale 0.033 0.04 0.033 0.033 0.033 0.02`. The exporter validates and records these values;
the CPU controller consumes the saved contract. Offline tests verify both scale variants. The selected policy’s success and contact-force audit results are recorded above. Diagnostic
scripts and full outcomes are under `outputs/robustness_20261008_multigpu/authority_*.json` and
`*_authority_*.json`.

The state teacher with shoulder scale 0.040 passed two independent 1,024-episode audits:
977/1,024 (95.41%) and 992/1,024 (96.88%). Peak rack forces were 29.09 N and 23.69 N;
4 and 6 episodes respectively exceeded 20 N. These are simulation success qualifications,
not a demonstration of hardware contact safety. The selected teacher and exact commands are
recorded in `authority_state_165/campaign.json` under the artifact root.

The vision continuation's first 200 updates reached 248/256 (96.88%) clean and 242/256
(94.53%) noisy development placements. It remains unqualified until all larger audits pass.
A full-collider diagnostic of its intermediate checkpoint scored 244/256 (95.31%), with
peak rack force 16.08 N and no episodes above 20 N. The matching minimal-collider audit
scored 237/256 (92.58%). These are development comparisons, not independent qualification.

After a further 200 visual updates, checkpoint 796 reached 248/256 (96.88%) clean and
244/256 (95.31%) noisy development placements. Its four independent 1,024-episode audits
scored 972, 984, 982 and 974 successes (95.51% pooled). The first audit missed the 95% gate,
so this checkpoint was **not** selected despite its pooled score. Separate full-collider checks
scored 982/1,024 clean and 981/1,024 noisy. A full-collider noisy check with nominal scene
colors scored 982/1,024. These additional diagnostics do not replace the failed qualification.
The largest full-collider rack force was 25.60 N; the clean and noisy broad-color tests each
had one episode above 20 N. The nominal-color test had seven such episodes (peak 24.62 N).

For full-collider **camera** runs use `train_transfer --full-colliders`, or override
`env.scene.robot.spawn.func=isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg:_spawn_so101_for_wrist_camera`.
This preserves all robot colliders and hides the camera housing from its own view. The state
spawn function restores that housing and invalidates wrist-vision comparisons: an earlier
48.83% result using it was rejected, with the reason recorded in its launch manifest.
A 32-world paired capture verified identical images, poses and physical parameters after using
the correct camera spawn. Robot self-collision remains disabled as in the existing task.

A separate diagnostic doubled the 30-second horizon for visual checkpoint 597: placement rose
from 184/256 to 210/256, so slow recoveries explain some timeouts. Qualification retains 30 seconds.

## Physical appearance recovered from the earlier branch

The setup description was recovered from `origin/feat/so101-sim2real-multigpu`, whose
`docs/SO101_SIM2REAL.md` recorded the previously supplied photograph. No new photograph is required.
The default scene now colors only the printed robot material orange, preserving black servos/camera;
the rack remains yellow. A brown 0.9144 × 1.00584 m desk replaces the green mat. The support height
is unchanged to preserve the reset poses; its footprint follows the previous branch and is not a
new physical measurement. The Newton camera render was visually checked after this correction.

Camera-Sim2Real additionally samples printed-plastic and rack colors within narrow orange/yellow
ranges each episode. Desk colors are five correlated brown shades, avoiding independently sampled
RGB extremes that can create green surfaces. These are native per-environment material writes,
not color masks in the observation. The desk is still **plain colored**: wood grain, local reflections
and changing shadow directions are not yet modeled. Global photometric augmentation cannot fully
replace those effects. Historical green-mat scores do not qualify this corrected visual scene.

The old branch also records measured vial dimensions and follower joint calibration changes.
Those physics/kinematic changes have **not** been silently merged into the trained task: the supplied
joint-map template remains unverified, and the current simulation dimensions/zeros must be reconciled
with those measurements before claiming hardware readiness.

## Physical and proprioceptive variation

Both randomized tasks retain this physical distribution during evaluation:

| Quantity | Episode-level range |
| --- | --- |
| Vial mass | 12–30 g |
| Vial static/dynamic friction | 0.2–1.3 |
| Vial restitution | 0–0.02 |
| Arm joint friction | ×0.6–2.5 |
| Arm viscous friction | ×0.6–3.5 |
| Armature | ×0.7–1.5 |
| Arm stiffness / damping | ×0.85–1.15 / ×0.7–1.5 |
| Gripper stiffness | ×0.6–1.7 |
| Home-start vial XY jitter | ±20 mm, rejecting rack overlap with 1 mm clearance |

Visual training also corrupts joint position by ±0.01 rad, joint velocity by ±0.02 rad/s, and
joint target by ±0.005 rad. Clean play disables this observation corruption; noise audits retain it.
These distributions do not replace physical joint calibration or measured latency characterization.
Rack pose and object dimensions are fixed in this distribution. The physical rack placement must
match the modeled workspace, and the recorded vial dimensions must be reconciled with the asset.

## Camera uncertainty

The Camera-Sim2Real task samples these quantities independently per environment, once per episode:

| Quantity | Range around the authored camera |
| --- | --- |
| Mount translation | ±3 mm on each optical-frame axis |
| Mount rotation | ±3° on each local Euler axis |
| Horizontal/vertical focal length | Independently ×0.95–1.05 |
| Principal point | ±1.5 pixels per axis at policy resolution |
| Brown radial coefficient k1 | −0.04 to +0.04 |
| Gamma | 0.85–1.15 |

These are engineering starting ranges for small assembly/lens differences, **not measured factory
tolerances**. The mount stays fixed relative to the moving gripper during an episode. Resets start
from the original mount, so perturbations cannot accumulate. Camera geometry remains randomized in
play mode even when photometric corruption is disabled.

The updated Newton renderer supports a fixed OpenCV distortion model via `spawn.distortion`, but
still shares its ray field across environments and rejects differing per-environment intrinsics.
It does not read lens coefficients directly from USD assets. Native distortion therefore does not
replace our independently sampled per-episode camera geometry. The task
instead renders 80×60 with a wider aperture and the original pixel focal length, then projects into
48×64 with an inverse pinhole/Brown warp. Overscan provides rendered scene pixels at the boundaries;
identity projection is the original central crop. Mount changes use actual 3D camera poses. No
additional camera, policy input, reward, or placement criterion is introduced.

Existing exposure (0.75–1.25), contrast (0.85–1.15), per-channel white balance (0.90–1.10), brightness
(±0.05), blur (0–0.5 blend), and pixel noise (±0.025) remain. The principal-point variation replaces
the former one-pixel image shift. Geometry is applied before photometric augmentation;
max-channel intensity normalization is optional and disabled for the selected raw-RGB policy. History is oldest-first and repeats the initial frame at reset.

This avoids requiring *exact* mounting/intrinsics within the trained distribution. It does not make
camera orientation, field of view, image aspect ratio, or lens characteristics arbitrary. Compare a
real image against simulation at a known pose before deployment; enlarge only the ranges that fail
that comparison. The [WowRobo camera listing](https://shop.wowrobo.com/products/2mp-usb-camera-module-for-so-arm100-101-30fps-3m-cable)
specifies 2 MP and 30 FPS, but does not establish the intrinsics or uncertainty bounds above.

## What the research implies for this setup

[Tobin et al.](https://arxiv.org/abs/1703.06907) demonstrate visual domain randomization for transfer.
[Garcia et al.](https://arxiv.org/html/2307.15320v1) study camera parameters, appearance and dynamics
variation for visual manipulation and use real images to help select simulation distributions. Their
results support covering the actual deployment distribution; they do not prescribe our numeric
ranges or imply that an arbitrary collection of augmentations guarantees transfer.

The [NVIDIA SO-101 workshop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
uses camera extrinsics, lighting, robot/mat appearance and object placement variation. Global color
augmentation is not equivalent to changing local material colors, shadows or backgrounds. Our current
pipeline covers global photometric changes, blur and per-episode material colors. Wood-grain
textures, local reflections and changing light/shadow directions remain coverage gaps.

[Peng et al.](https://arxiv.org/abs/1710.06537) motivate varying physical dynamics for transfer.
The task already varies vial mass/contact properties, arm friction/viscous friction/armature,
arm gains, gripper stiffness and vial XY placement. The Transfer profile above additionally models
persistent encoder bias and a bounded command delay. Gear backlash, frame loss, variable camera
latency and all contact-model differences remain uncovered; randomizing motor gains is not a
substitute for those mechanisms.

The [official SO-101 calibration guide](https://huggingface.co/docs/lerobot/so101) requires motor
calibration. The [RobotStudio simulation notes](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md)
also distinguish calibration conventions and gripper coordinates. Camera randomization does not
remove the need to map LeRobot's joint units, signs, zeros and gripper percentage into the simulation
joint convention. RobotStudio's [hardware characterization discussion](https://github.com/TheRobotStudio/SO-ARM100/issues/134)
reports backlash and load-dependent behavior on tested hardware; it is useful evidence of a failure
mode, not a universal SO-101 tolerance specification.

## LEAPP / LeRobot deployment contract

Owning the inference script lets us make preprocessing and action interpretation explicit. It does
not remove capture, computation or serial-bus latency. The deployment integration must preserve:

- RGB ordering, the intended 4:3 field of view, 64×48 policy images, raw bytes scaled to [0, 1],
  and two-frame history for the selected visual policy. Do not apply max-channel normalization. Simulation overscan and
  randomized projection are training operations, not distortions to add to real camera frames.
- Joint order and radians, measured joint velocity, the last target actually sent, and the previous
  clipped action. The visual actor has 24 proprioceptive inputs. Learned normalization belongs in
  the export and must not be applied a second time in the script.
- A 30 Hz policy/camera cadence and the existing **120 Hz relative-target update**. The active Newton
  environment does not fuse action decimation: it recomputes `measured_q + scaled_action` each physics
  substep. Arm scale is 0.033 rad and gripper scale is 0.02 rad; the gripper target is soft-limit bounded.
  A 30 Hz position write held for four substeps is a different controller. Check bus throughput before
  choosing between reproducing the 120 Hz loop and retraining for a measured slower loop.
- Camera receipt time, inference duration, joint-read/write times and missed deadlines. A host receipt
  timestamp is not a hardware exposure timestamp. Measure frame age as well as inference speed before
  setting delay randomization; a fast neural network alone does not establish low camera-to-action lag.

The generic Isaac Lab LEAPP deployment environment writes targets once per policy step. Its successful
export validation alone would not demonstrate parity with this task's substep feedback controller.
Validate the chosen export and custom loop against simulation trajectories, including reset/history
behavior, before connecting the robot. LEAPP and LeRobot do not themselves compensate for domain gaps.

## Historical diagnostics and subsequent fixes

A real Newton/renderer probe confirmed 80×60 rendering, 64×48 observations, mount attachment while
moving, and isolation when resetting one environment. Projection tests cover the original central
crop, focal/center geometry, radial inversion and overscan bounds. Camera draws persist through an
episode and are refreshed only for reset environments.

The old selected policy was evaluated with 256 home-start episodes per condition, seed 3301, using
Newton physics and renderer. These are diagnostic samples, not qualification of a newly trained model:

| Camera geometry | Successes | Success |
| --- | ---: | ---: |
| Nominal mount and projection | 234 / 256 | 91.41% |
| Mount variation only | 125 / 256 | 48.83% |
| Projection variation only | 233 / 256 | 91.02% |
| Both | 127 / 256 | 49.61% |

The substantial mounting sensitivity demonstrates why the old >90% nominal-camera scores cannot be
claimed for the expanded distribution. Four single-GPU PPO continuations compared learning rates
3e-5 and 1e-4 with mixed-phase versus home-only starts. At checkpoint iteration 600 (source iteration
448), matching 256-episode audits scored 49.22%, 48.83%, 50.39%, and 46.88%, respectively. Those runs
were stopped after this audit; short PPO continuation had not improved camera robustness.

The subsequent normalized-image DAgger audits also failed to recover performance: mixed-start
runs at iteration 400 scored 38.28% (1e-4) and 35.16% (5e-4); home-only runs at iteration 200 scored
42.58% and 32.81%. Ordinary-RGB mixed-start runs at iteration 200 scored 46.09% and 28.91%.
These were still on the incorrect workshop appearance and were stopped after the user's correction.

Four corrected-scene continuation runs compared ordinary RGB at learning rates 1e-4/5e-4 with the original
pixel normalization at 1e-4, plus ordinary-RGB student-only DAgger at 1e-4. The first three optionally
mix teacher actions into rollouts, with probability decreasing linearly from one to zero over 300
iterations. This changes data collection, not the task or inference policy. All use the full camera,
material and physical variation. Teacher-assisted training metrics are not student acceptance scores;
independent home-start audits are required.
Checkpoint-200 audits on the corrected scene (256 first home-start episodes, seed 3901) scored
9.77% for ordinary-RGB slow warm-up, 16.80% for fast warm-up, 15.23% for normalized warm-up, and
31.25% for ordinary-RGB student-only DAgger. These are interim results, not qualified models.
At iteration 400, the same four runs scored **41.41%, 32.03%, 42.19%, and 34.38%**, respectively.
Distillation was stopped after this comparison. Four PPO refinements started from the best
normalized and ordinary-RGB students, each with mixed-stage versus home-only starts. They use
learning rate 1e-4, gamma 0.999, entropy coefficient 0.001 and 1,024 environments. Each 200-update
block is followed by a separate 256-home-start audit; candidates above 92% receive a fresh
1,024-episode audit on a different seed. These intermediate runs preceded the camera housing fix below.
The frozen state teacher scored **969/1,024 (94.63%)** on the corrected desk scene, seed 2203.
It retained full physical randomization and used the same 30-second any-hole criterion.

PPO improves the same corrected-scene students: at 600 updates, independent 256-home-start
success is 68.75% (normalized/home), 69.53% (normalized/mixed), 72.66% (RGB/home), and 66.41%
(RGB/mixed), seed 4101. These runs retained the full camera and appearance variation.

### Physically valid home perturbations

A geometric audit found that the old unconstrained XY jitter placed 9/256 vials visibly inside
the rack in the RGB/home 600-update audit. All nine failed. The unperturbed dataset rows were clear.
The sampler now rejects home perturbations whose projected body/cap rectangles overlap the rack
base, with 1 mm clearance, and resamples within the same ±20 mm bounds. After 16 unsuccessful
attempts it retains the original validated pose. Non-home curriculum poses are untouched. This
uses a conservative separating-axis footprint check; it does not change success or insertion tolerances.

`home_rack_clearance=0.001` in audit metadata identifies this corrected reset profile. Earlier
scores above predate it and must not be presented as directly matched comparisons. Regression tests
cover rotated rack frames, repeated home draws, unchanged non-home states, and unchanged XY bounds.

With the corrected sampler, the frozen state teacher scores **991/1,024 (96.78%)** on seed 2203,
with 13.82 s mean successful duration and no episode exceeding the 20 N rack-contact diagnostic.
The earlier 94.63% state result used unconstrained reset jitter; the improvement is not a policy update.

After 1,000 PPO updates, the four visual development audits (256 episodes, seed 4101) score:

| Preprocessing | Home starts | Mixed-stage starts |
| --- | ---: | ---: |
| Per-pixel intensity normalization | 211/256 (82.42%) | 199/256 (77.73%) |
| Ordinary RGB | 212/256 (82.81%) | 198/256 (77.34%) |

The 800-update audits and final 200-update training block use the corrected sampler; earlier blocks
predate the fix. Camera mounting, projection, material and physical variation remain enabled in these
clean-observation audits. Training additionally includes image/proprioception corruption.

Four continuations use 2,048 environments each, home-only starts, and the best corrected-sampler
checkpoint for each preprocessing choice. For each, standard exploration (inherited standard deviation,
learning rate 1e-4, entropy coefficient 0.001) is compared with low-noise refinement (initial standard
deviation 0.02, bounds 0.01–0.05, learning rate 5e-5, no entropy bonus). Gamma remains 0.999. Each
200-update block receives a development audit; candidates above 90% receive a separate 1,024-episode
qualification on seed 4201. After 200 updates, standard exploration scored 83.20% normalized /
83.98% raw RGB; low-noise refinement scored 77.34% / 82.03%. The low-noise runs were stopped.
Two additional RGB learning-rate comparisons were started but stopped when the rendering problem
below was discovered; no outcome is claimed for those incomplete comparisons.

### Camera housing obstruction and final qualification

Depth/RGB recordings revealed that the randomized optical frame could sit inside the fixed camera
housing mesh. Blocked views hit geometry only 0.01–0.4 mm away; Newton does not enforce the near
clipping plane. Consequently, the earlier camera sensitivity and training results include artificial
self-occlusion and must not be interpreted solely as calibration sensitivity or inadequate learning.

Camera tasks now hide the camera assembly's visual subtree, while preserving all jaw visuals,
collision shapes and randomization ranges. State-task visuals are unchanged. Mount reset invalidates
the image buffer so the first observation uses the new pose. Backface culling alone did not remove
all blocked views and was not selected as the fix.

The frozen raw-RGB PPO checkpoint then scored **971/1,024 (94.82%)** on seed 4201 and
**958/1,024 (93.55%)** with observation noise on seed 4202. Independent confirmation seeds 4301
and 4302 scored **94.43% clean / 94.04% noisy**, with no episode above the 20 N rack-force diagnostic.
The normalized candidate scored 92.58% in the initial clean comparison. The selected policy uses
raw RGB with two-frame history. No success criterion, camera range or physics distribution was
relaxed. See [RESULTS.md](RESULTS.md) for artifacts and exact reproduction commands.

Raw evidence and checkpoints live outside Git in the `camera_randomization_20261007` artifact directory.

## Explicit visual LEAPP bundle

The generic manager export is not a supported deployment artifact for this custom camera/history
term: an export probe (after adapting the previous-action callback signature) produced a graph
without an RGB input. Export completion alone therefore was insufficient. Use the explicit actor
bundle, which exposes both tensors and checks the LEAPP runtime against the original actor on eight
independently varying input pairs:

```bash
CUDA_VISIBLE_DEVICES='' uv run --extra leapp python -m isaaclab_tutorial.utils.export_leapp \
  --model /absolute/path/to/exported/policy.pt \
  --output /absolute/path/to/new/leapp_bundle --history 2 --no-normalize-intensity
```

This takes the **TorchScript policy export**, not a training checkpoint. It copies a flat-signature
actor into a self-contained LEAPP directory and writes `contract.json` with the source hash, image
history/preprocessing, joint conventions, action scale and parity error. LEAPP's documented
[prebuilt-model interface](https://nvidia-isaac.github.io/leapp/guides/export.html#bringing-your-own-model)
packages the existing model; the separate runtime parity check verifies its behavior. The optional
`leapp` extra pins the tested version. It is not required for training.

The custom controller calls `InferenceManager.run_policy` with `policy/proprioception` and
`policy/wrist_rgb`, and receives `policy/action`. This artifact intentionally contains the actor;
image preprocessing/history and the 120 Hz target loop are implemented in
`src/isaaclab_tutorial/utils/deploy.py`. Neither a LeRobot hardware rollout nor timing on the actual
camera/serial bus has been tested here. See [deployment instructions](DEPLOYMENT.md).

Raw RGB is the export default. Use `--normalize-intensity` only for an actor trained with
max-channel normalization. This flag records the preprocessing contract; it does not change the
neural network.
