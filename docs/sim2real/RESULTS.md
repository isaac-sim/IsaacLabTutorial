# Randomized state and vision policies — 2026-10-07

> These measurements use the October 7 dependency pin (`efbbde338`). The October 8
> update to PR #8379 changes the runtime; these are historical results, not new-stack qualification.

The current **Newton MJWarp + Newton renderer** vision policy exceeds 90% on both clean and
corrupted observations. The scene uses the orange robot, yellow rack and bare brown desk, with the
full camera mounting, projection, appearance and dynamics randomization. Attempts last 30 seconds;
a stable, upright, released vial seated in any rack opening succeeds.

## Current acceptance results

Each audit counts exactly one first home-start episode per environment, with 1,024 environments.
Camera geometry, material colors and physics variation remain enabled in clean-observation audits.
Noise audits additionally enable the training image and proprioception corruption. The policy is
frozen and deterministic; GPU simulation is not necessarily bitwise reproducible.

| Policy / audit | Seed | Successes / attempts | Success | Mean successful duration | Episodes above 20 N rack force |
| --- | ---: | ---: | ---: | ---: | ---: |
| State teacher | 2203 | 991 / 1,024 | **96.78%** | 13.82 s | 0 |
| Vision, clean qualification | 4201 | 971 / 1,024 | **94.82%** | 14.71 s | 0 |
| Vision, observation noise | 4202 | 958 / 1,024 | **93.55%** | 14.65 s | 0 |
| Vision, fresh-seed confirmation | 4301 | 967 / 1,024 | **94.43%** | 14.54 s | 0 |
| Vision, fresh-seed noise confirmation | 4302 | 963 / 1,024 | **94.04%** | 14.67 s | 0 |

The two confirmation seeds were run after selecting the RGB checkpoint. These are simulated
success rates, not real-robot results. All successes of this selected visual policy used the original
opening; tests cover acceptance in all four openings. The force threshold is a diagnostic, not an
additional success gate. Peak forces in the two confirmation audits were 10.63 N and 12.84 N.

Home perturbations now reject vial footprints overlapping the rack, retaining the same ±20 mm
sampling bounds and 1 mm clearance. The earlier 94.63% state score used unconstrained jitter;
**the state checkpoint did not change**. Do not attribute that score difference to additional training.

## What unlocked vision

PPO initially plateaued around 83–84% despite continued training. Recorded wrist images showed
that mount randomization could move the optical frame into the camera's own visual housing.
Newton's ray tracer does not enforce the camera near plane: blocked pixels hit surfaces only
0.01–0.4 mm away. Several policies were completing motions after receiving essentially blank images.

The camera task now hides only the camera assembly's visual subtree. Gripper jaws, robot servos,
collision geometry, rewards, success tolerances and all randomization ranges are retained. This
represents a moving camera assembly without allowing its fixed housing mesh to obstruct the
randomized optical center. The state task's visual configuration is unchanged. A second fix invalidates
the cached image after changing the mount, so the first frame also uses the new pose.

With the same frozen RGB checkpoint, a 256-episode diagnostic reached 94.92% after removing the
housing obstruction; the production fixes then passed the larger audits above. Replayed depth/RGB
checks confirm valid views both immediately after reset and on subsequent frames. These renderer
profile changes mean the final scores are not directly matched learning-curve improvements over the
old obstructed-camera profile.

## Selected artifacts and training lineage

Set `VISION_ROOT` to the external `camera_randomization_20261007/selected/vision` directory.
Generated models, logs, exports and raw audits are deliberately outside Git.

- `model.pt`: PPO checkpoint, SHA-256
  `158493c1b4c1a3eb66571f370611f8ebe6a0563de79e5a31e9a83c472e4954fb`.
- `exported/policy.pt`, `exported/policy.onnx` and its `.data` companion.
- `leapp/leapp.yaml`, `leapp/visual_actor.pt`, `leapp/contract.json`.
- `qualification.json`, `noise_stress.json`, `confirmation.json`, `noise_confirmation.json`.
- `manifest.json`, `cpu_runtime_parity.json`, `deployment_compute_benchmark.json`.

The selected actor consumes **two raw RGB frames**, oldest first, at 64×48, plus 24 proprioceptive
values. Convert RGB bytes to [0, 1]; **do not apply max-channel intensity normalization** to this
checkpoint. Repeat the first image to initialize history. Learned proprioceptive normalization is
included in the export. Use `--agent rsl_rl_ppo_cfg_entry_point` to load the checkpoint.

The lineage is a warm-started DAgger student, 400 corrected-scene distillation updates, then 1,000
home-start PPO updates at 1,024 environments and 200 more at 2,048. PPO used learning rate 1e-4,
gamma 0.999 and entropy coefficient 0.001. The critic used privileged state; the actor used only RGB
and proprioception. Earlier checkpoints and source hashes remain in the external experiment archive.
The selected weights were frozen before the housing fix; a separate 20-update training check validates
training on the fixed scene and does not replace the audited model.

Four experiments ran concurrently, each scoped to one GPU with `CUDA_VISIBLE_DEVICES`. Comparisons
covered raw versus intensity-normalized RGB, mixed-stage versus home starts, and low-noise versus
standard-exploration PPO. Lowering action standard deviation abruptly to 0.02 did not improve the
200-update audits. The renderer investigation resolved the larger bottleneck without adding policy
architecture or MDP complexity. Detailed intermediate results are in
[DOMAIN_RANDOMIZATION.md](DOMAIN_RANDOMIZATION.md).

LEAPP export parity and the separate Torch 2.10 CPU deployment runtime both matched the Torch 2.13
training actor with **zero maximum absolute error** on eight varied input pairs. Deployment still
requires verified robot joint calibration and physical validation; see [DEPLOYMENT.md](DEPLOYMENT.md).

## Training from random initialization

A separate campaign in `from_scratch_20261007` verifies fresh initialization rather than treating
continuation smoke tests as convergence evidence. Four experiments ran in parallel, each with one
GPU selected by `CUDA_VISIBLE_DEVICES`. All artifacts and exact command/configuration snapshots
remain outside the repository.

Two state actors and critics started randomly, without checkpoint arguments. Both used 800 updates
on the base state task, followed by broader Sim2Real training at learning rate 3e-4, adaptive
scheduling, gamma 0.999 and entropy coefficient 0.005. Optimizer state was preserved at the transition.
Development audits ran every 200 updates; the independent qualification seed was 7102.

| Training seed | Bootstrap updates | Randomized updates | Qualification | Mean successful duration |
| --- | ---: | ---: | ---: | ---: |
| 42 | 800 | 200 | **951/1,024 (92.87%)** | 14.24 s |
| 43 | 800 | 400 | **945/1,024 (92.29%)** | 14.49 s |

Neither qualification had an episode above the 20 N rack-contact diagnostic. These are new models,
separate from the previously selected 96.78% teacher. Training directly with the broad randomized
task from random weights did not succeed within the initial 400-update comparisons; bootstrapping
on the base task was materially more effective. Repeated RSL-RL continuation blocks reuse boundary
iteration labels, so checkpoint labels 995 and 1194 represent 1,000 and 1,200 actual updates here.

The complete fresh pipeline used the seed-42 teacher above and a new random visual network
(seed 46): **400 DAgger updates at 1,024 environments, then 200 PPO updates at 2,048 environments**.
DAgger used the default 5e-4 learning rate and no teacher-action warm-up. PPO used home-only starts,
learning rate 1e-4, fixed scheduling, gamma 0.999, entropy coefficient 0.001 and initial action
standard deviation 0.2. The actor retained two raw RGB frames and the full training corruption.
Neither state nor visual initialization used historical policy weights.

| Fresh-pipeline audit | Seed | Successes / attempts | Success | Mean successful duration | Episodes above 20 N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean qualification | 7401 | 927 / 1,024 | **90.53%** | 14.52 s | 0 |
| Noisy qualification | 7402 | 944 / 1,024 | **92.19%** | 14.97 s | 1 |
| Fresh clean confirmation | 7501 | 935 / 1,024 | **91.31%** | 14.68 s | 0 |
| Fresh noisy confirmation | 7502 | 935 / 1,024 | **91.31%** | 14.82 s | 0 |

The noisy qualification's peak rack force was 22.36 N; this outlier is retained, not excluded from
the score. Confirmation peaks were 13.01 N and 11.41 N. The policy was frozen before the two
confirmation seeds. This independently validates the end-to-end fresh-training path; it does not
replace the stronger earlier 94.43% / 94.04% model at the top of this report.

A separate fresh visual actor using the existing 96.78% teacher scored **93.95% clean / 93.07% noisy**
(962 and 953 successes over 1,024 attempts, seeds 7202/7203), with no episode above 20 N. It used
200 DAgger updates with a 200-update teacher-action warm-up, then 400 PPO updates. These results
establish fresh visual learning, but are distinct from the wholly fresh teacher/student chain.
Ordinary DAgger with that existing teacher peaked at 84.38% in the 600-update development audit
and fell to 82.03% at 1,000; PPO refinement was useful rather than merely a checkpoint-format test.

The fresh-pipeline artifacts are `from_scratch_20261007/selected/state` and
`from_scratch_20261007/selected/vision`.
The visual directory includes TorchScript/ONNX exports, a LEAPP bundle, both qualification audits,
both confirmation audits, and a manifest with the complete training lineage. Training-runtime and
isolated Torch 2.10 CPU LEAPP outputs match exactly on eight varied input pairs. Checkpoint hashes:

- Fresh state: `d8a23b2837ae81b6c1c37171d33921ab3a7711c4acb2c354f4e5a0e0d26eaa50`.
- Fresh vision: `e3e754f2779b8c2df2affe86f8432849e47ae71a289433250106d7922986cb85`.

To replay the fresh visual confirmation, use the visual audit command below with this artifact's
`VISION_ROOT` and seed 7501; use seed 7502 plus the corruption overrides for noisy confirmation.
The README now gives this validated fresh-training recipe and separates the PPO initializer from
the trained checkpoint used for evaluation.

The explicit distillation-to-PPO converter loads the student actor and its teacher's compatible
63-input critic, records source hashes and initializes a new optimizer. It has passed both the actual
RSL-RL loader regression test and Newton visual training. A separate 12-update check of the public
`-Camera` PPO task also completed with randomly initialized actor/critic and no checkpoint; that
short check establishes execution, not a qualified visual success rate.

## Reproduce the visual audits

```bash
export VISION_ROOT=/absolute/path/to/camera_randomization_20261007/selected/vision
CUDA_VISIBLE_DEVICES=0 SO101_EVALUATION_EPISODES=1024 \
  SO101_EVALUATION_OUTPUT=outputs/vision_confirmation.json \
  uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real \
  --agent rsl_rl_ppo_cfg_entry_point --num_envs 1024 --seed 4301 \
  --checkpoint "$VISION_ROOT/model.pt" --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2 \
  env.observations.wrist_rgb.image.params.normalize_intensity=False
```

For the noise confirmation, use seed 4302 and append
`env.observations.wrist_rgb.enable_corruption=True env.observations.proprioception.enable_corruption=True`.
Use the thread/TMPDIR settings from the README. The state teacher remains at the historical archive's
`selected/state/model.pt` (SHA-256
`54a37f4fe82096b0f1176c76f10129bbb4eca8ac44ae48fced0dd10f066b2305`).
Run it with task `IsaacTutorial-Place-Vial-SO101-Sim2Real` and `presets=newton_mjwarp`.

## Validation and transfer limits

The main suite passes **99 tests**; its deployment module is skipped because OpenCV is absent from
the simulation environment. The isolated LeRobot environment runs the three deployment tests.
Tests cover any-hole geometry and termination, bounded teacher actions, image history/projection,
mount reset behavior, housing visibility scope, checkpoint initialization and exact episode accounting.
[PERFORMANCE.md](../PERFORMANCE.md) records startup, environment throughput and deployment compute.

Historical policies on the earlier yellow/green scene scored 93.85% state and 91.50% vision in
randomized Newton, but only 61.72% state and 44.53% vision in a 128-episode frozen PhysX/OVRTX
transfer check. Those are different checkpoints/profile conditions and are not transfer measurements
of the current visual policy. They remain evidence that source-simulator success alone is insufficient.
The current policy has **not** been qualified on another physics engine or the real SO-101.
