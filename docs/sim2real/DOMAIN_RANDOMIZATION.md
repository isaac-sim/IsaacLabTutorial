# Sim2real coverage and the WowRobo wrist camera

The deployment target is the user’s **orange WowRobo SO-101, yellow rack and bare wooden desk**,
with the WowRobo wrist camera. The original workshop’s yellow robot and green mat were incorrect.
Deployment will use LEAPP with a custom inference script and LeRobot for robot control. Source-simulator
success is necessary, but does not establish successful transfer: historical frozen policies
lost substantial performance when moved to PhysX. No real-robot success rate has been measured.
The current selected visual policy passes fresh-seed Newton audits at **94.43% clean / 94.04% noisy**;
see [RESULTS.md](RESULTS.md). Earlier camera diagnostics below predate the self-occlusion fix.

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
