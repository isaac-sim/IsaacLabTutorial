# Sim2real coverage and the WowRobo wrist camera

The deployment target is the same WowRobo-assembled SO-101 and wrist camera as the workshop asset.
Deployment will use LEAPP with a custom inference script and LeRobot for robot control. Source-simulator
success is necessary, but does not establish successful transfer: the existing frozen policies already
lose substantial performance when moved to PhysX. No real-robot success rate has been measured.

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

Newton's current renderer caches shared camera rays from one focal length. Changing each camera's
intrinsic matrix alone would therefore not generate the desired per-environment images. The task
instead renders 80×60 with a wider aperture and the original pixel focal length, then projects into
48×64 with an inverse pinhole/Brown warp. Overscan provides rendered scene pixels at the boundaries;
identity projection is the original central crop. Mount changes use actual 3D camera poses. No
additional camera, policy input, reward, or placement criterion is introduced.

Existing exposure (0.75–1.25), contrast (0.85–1.15), per-channel white balance (0.90–1.10), brightness
(±0.05), blur (0–0.5 blend), and pixel noise (±0.025) remain. The principal-point variation replaces
the former one-pixel image shift. Geometry is applied before intensity normalization and photometric
augmentation. History is oldest-first and repeats the initial frame at reset.

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
pipeline covers global photometric changes and blur; local material/texture and scene-light
randomization remain a coverage gap, especially if the real mat, rack, vial or surroundings differ.

[Peng et al.](https://arxiv.org/abs/1710.06537) motivate varying physical dynamics for transfer.
The task already varies vial mass/contact properties, arm friction/viscous friction/armature,
arm gains, gripper stiffness and vial XY placement. These distributions do not explicitly model
gear backlash, persistent encoder-zero error, camera/control delay, frame loss or all contact-model
differences. Randomizing motor gains is not a substitute for those mechanisms.

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

- RGB ordering, the intended 4:3 field of view, 64×48 policy images, per-pixel maximum-channel
  normalization, and two-frame history for the selected visual policy. Simulation overscan and
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

## Evidence

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

Four teacher-guided distillation continuations then compare 1e-4 and 5e-4 learning rates with the
same two reset distributions. They start from the original visual actor and qualified state teacher.
All runs use the complete camera geometry profile, photometric/proprioceptive corruption, the
existing physical distribution, and Newton rendering.
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
  --output /absolute/path/to/new/leapp_bundle --history 2
```

This takes the **TorchScript policy export**, not a training checkpoint. It copies a flat-signature
actor into a self-contained LEAPP directory and writes `contract.json` with the source hash, image
history/preprocessing, joint conventions, action scale and parity error. LEAPP's documented
[prebuilt-model interface](https://nvidia-isaac.github.io/leapp/guides/export.html#bringing-your-own-model)
packages the existing model; the separate runtime parity check verifies its behavior. The optional
`leapp` extra pins the tested version. It is not required for training.

The script will call `InferenceManager.run_policy` with `policy/proprioception` and
`policy/wrist_rgb`, and receive `policy/action`. This artifact intentionally contains the actor;
image preprocessing/history and the 120 Hz target loop belong to the custom deployment script.
It is not an end-to-end hardware controller. Neither a LeRobot hardware rollout nor timing on the
actual camera/serial bus has been tested here.
