> **Scene revision:** the current scene restores the user's orange robot and brown desk and adds
> material variation. The historical vision scores below were measured on the earlier yellow/green
> workshop scene. Camera-only overrides do not reproduce those historical images on the current tree;
> use the recorded historical commit for reproduction. Current-scene student qualification is pending.
>
> Current-scene state audit: **969/1,024 (94.63%)**, seed 2203, same frozen selected state checkpoint
> and full physical randomization. The corrected-scene vision audits reach at most 72.66%
> after 600 PPO updates, before the rack-clearance reset fix; see the [randomization assessment](DOMAIN_RANDOMIZATION.md).

# Sim2real training and any-hole placement — 2026-10-07

**Camera profile update:** the current Camera-Sim2Real configuration additionally randomizes camera
mounting and projection. The qualification scores below predate that change and use the original
camera geometry. See [the camera sensitivity audit](DOMAIN_RANDOMIZATION.md#evidence) for the new
distribution; the previous visual checkpoint is not qualified under it.

The task accepts a vial placed in any of the rack's four openings. It still requires the vial to be
upright, released from the jaws, seated at the original depth, nearly motionless, and stable for ten
control steps. Insertion and held-vial shaping use the nearest opening. No target selector, extra
reward stages, or new action semantics were added. The audited policies currently favor the original
opening; tests verify that each of the other three openings is equally valid.

Training logs, checkpoints, exports and raw audit JSON are archived outside this repository.
Set `ARTIFACT_ROOT` to the archived `sim2real_any_hole_20261007` output directory before using
the commands below. They are not bundled or downloaded by `uv sync`.

## Latest recheck after cleanup and dependency publication

The selected checkpoints were re-evaluated after installing Isaac Lab directly from published
commit `efbbde338f5222eb759cd67397fdc84c088f6c5c`, using the cleaned task configurations.
No training updates were applied to the selected checkpoint files.

| Policy | Seed | Successes / attempts | Success | Mean successful duration | Episodes above 20 N rack force |
| --- | ---: | ---: | ---: | ---: | ---: |
| State | 2203 | 961 / 1,024 | **93.85%** | 13.79 s | 3 |
| Vision, clean observations | 2203 | 937 / 1,024 | **91.50%** | 13.90 s | 1 |
| Vision, training observation corruption | 2204 | 931 / 1,024 | **90.92%** | 14.31 s | 0 |

The differences from the earlier qualification below are small rerun variation. These are repeated
seeds, not a new blinded test. All successes in these audits used the original opening; any-hole
acceptance is covered by geometry/termination tests, but the learned policies have not demonstrated
balanced use of the four openings. The 20 N metric is diagnostic, not a success-condition gate.
Raw rechecks are archived under `performance/audit_state`, `performance/audit_vision`, and
`performance/audit_vision_noise`. Startup and training speeds are in [PERFORMANCE.md](../PERFORMANCE.md).

## Evaluation protocol

- Newton MJWarp physics; Newton renderer for the wrist-camera policy.
- 30-second attempts, as requested. Every audit starts at home; intermediate curriculum starts are
  used only in training. One first episode per environment is counted, with 1,024 environments.
- Fresh ±20 mm vial XY offsets and the full training physics distribution remain enabled in play.
- Vial mass 12–30 g; static/dynamic friction 0.2–1.3; restitution 0–0.02;
  gripper stiffness ×0.6–1.7; arm friction ×0.6–2.5; viscous friction ×0.6–3.5;
  armature ×0.7–1.5; stiffness ×0.85–1.15; damping ×0.7–1.5.
- Clean-image audits disable observation corruption, but retain all physical randomization.
  Appearance/proprioception stress audits additionally enable the training observation corruption.
- All training processes are scoped to one physical GPU through `CUDA_VISIBLE_DEVICES`.
  The initial four parallel branches were two state PPO learning rates, camera distillation,
  and camera PPO. Freed GPUs then ran home-start camera refinements.

Archived historical sim2sim results used the original single-hole criterion and different physics
profiles. They are not directly comparable. Earlier 20-second scores also have a different deadline.
These are simulated results, not measurements from the real robot.

## Qualified state policy

The selected state checkpoint scored **963/1,024 (94.04%)** on qualification seed 2203, after
**951/1,024 (92.87%)** on selection seed 2201. Successful qualification attempts averaged **13.89 s**.
Two qualification episodes exceeded the existing 20 N rack-contact diagnostic threshold; placement
success does not imply an absence of force spikes.

Checkpoint and exports:

- `$ARTIFACT_ROOT/selected/state/model.pt`
- `$ARTIFACT_ROOT/selected/state/exported/policy.pt`
- `$ARTIFACT_ROOT/selected/state/exported/policy.onnx` (keep its `.data` companion)
- `$ARTIFACT_ROOT/selected/state/qualification.json`

The JSON records checkpoint SHA-256, invocation, effective randomization, observation corruption,
criterion version, per-episode outcomes, and successful hole indices. The state actor uses privileged
60-dimensional state; it is the teacher, not the deployable camera controller.

## Camera pipeline

The selected PPO-refined camera policy scored **935/1,024 (91.31%)** on qualification seed 2203,
after **923/1,024 (90.14%)** on selection seed 2201. With the training camera and proprioception
corruption enabled, seed 2204 scored **937/1,024 (91.50%)**. These point estimates do not guarantee
a greater-than-90% deployment success probability.
The final comparison used the clean, noise and transfer audits to choose between two candidates; these
reported qualification scores are not a blinded test after that final selection. Clean qualification
successes averaged **13.81 s**. One clean qualification episode exceeded the 20 N rack-contact
diagnostic threshold; none did in the noise stress audit.

The selected continuation uses PPO learning rate 3e-5 and discount factor 0.999; the faster 1e-4
alternative had similar selection performance but worse qualification and transfer scores.

The selected visual artifacts are in `$ARTIFACT_ROOT/selected/vision/`:
`model.pt`, `exported/policy.pt`, `exported/policy.onnx` and its `.data` companion,
`qualification.json`, and `noise_stress.json`. The checkpoint is a visual **PPO** model initialized
from a distilled student: use `--agent rsl_rl_ppo_cfg_entry_point`, not the distillation runner,
when loading it. Distillation remains functional; PPO refinement produced the strongest student.

The visual policy consumes wrist RGB and 24-dimensional proprioception. It uses two consecutive
48×64 RGB frames, stacked oldest first into six channels, with each pixel divided by its largest
RGB channel (minimum denominator 1e-6). Learned observation normalization is included in the policy
export. History must be initialized by repeating the first frame when an episode starts.

DAgger uses the frozen state teacher and labels the actions after clipping them to [-1, 1], matching
what the environment executes. The warm-start utility preserves a trained visual encoder and can
expand a single-frame encoder into an averaged frame history. PPO refinement retains the same camera
inputs and uses privileged state only in the critic.

Training includes episode-consistent exposure, contrast, white-balance and brightness variation,
image shifts, blur, pixel noise, and proprioceptive noise. Policy outputs still require the environment's
action clipping and joint-target integration; the exported network alone does not implement the robot
control loop.

Detailed experiment commands, logs, checkpoint paths and development audits are under
`$ARTIFACT_ROOT/`. The original warm-start lineage is in the archived `sim2real_20261007` campaign.

## Reproduce the selected policies

```bash
export ARTIFACT_ROOT=/absolute/path/to/archived/sim2real_any_hole_20261007
CUDA_VISIBLE_DEVICES=0 SO101_EVALUATION_OUTPUT=outputs/state_recheck.json \
  .venv/bin/isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Sim2Real --num_envs 1024 --seed 2203 \
  --checkpoint $ARTIFACT_ROOT/selected/state/model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=1 SO101_EVALUATION_OUTPUT=outputs/vision_recheck.json \
  .venv/bin/isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real --agent rsl_rl_ppo_cfg_entry_point \
  --num_envs 1024 --seed 2203 \
  --checkpoint $ARTIFACT_ROOT/selected/vision/model.pt --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2
```

To reproduce the **historical** vision profile, also disable the added camera variation:
`env.events.camera_mount.params.position_range=0.0 env.events.camera_mount.params.rotation_range=0.0
env.observations.wrist_rgb.image.params.focal_scale_range=[1.0,1.0]
env.observations.wrist_rgb.image.params.principal_point_pixels=0.0
env.observations.wrist_rgb.image.params.radial_distortion_range=[0.0,0.0]`.
The overscan's identity projection is the original central view. For the historical corruption audit,
also restore `env.observations.wrist_rgb.image.params.gamma_range=[1.0,1.0]` and
`env.observations.wrist_rgb.image.params.shift_pixels=1`.

For the visual noise audit, use seed 2204 and append
`env.observations.wrist_rgb.enable_corruption=True env.observations.proprioception.enable_corruption=True`.
Policy inference is deterministic; GPU physics need not be bitwise deterministic across executions.

The full test suite passes **83 tests**. Geometry tests check all four hole centers against the USD
collision rails, reject dividers/rim placements, and retain contact, release, uprightness and stability
requirements. Warm-start tests verify preservation of the pretrained encoder's function and teacher
weights. Evaluation tests cover exact episode accounting and effective randomization metadata.


## Transfer remains below target

The selected Newton-trained policies were frozen and evaluated on randomized PhysX, with OVRTX
rendering for vision, for 128 home-start episodes at seed 2205. State scored **79/128 (61.72%)**;
vision scored **57/128 (44.53%)**. These diagnostics are substantially below the Newton qualification
scores. Broad training randomization and successful source simulation do **not** establish real-robot
readiness. The policies have not been evaluated on the real robot.

A short mixed-engine state continuation did not resolve this gap, so it was not selected:

| State checkpoint | Newton, 128 episodes | PhysX, 128 episodes |
| --- | ---: | ---: |
| Selected Newton policy | 112/128 (87.50%) | 79/128 (61.72%) |
| +50 PhysX updates, learning rate 3e-5 | 109/128 (85.16%) | 80/128 (62.50%) |
| +50 PhysX updates, learning rate 1e-4 | 109/128 (85.16%) | 83/128 (64.84%) |

This table uses seed 2205 and 128 environments throughout; it must not be compared directly to the
1,024-environment Newton qualification as a paired test. The continuations trained on PhysX, so
those rows are adaptation results rather than frozen source-only transfer. Their 1,024-environment
startup attempts were stopped before any updates after seven minutes of scene construction and
restarted with 128 environments. All experiment logs and checkpoints were retained.

The fast visual continuation transferred at 47/128 (36.72%), compared with 57/128 (44.53%) for the
selected conservative continuation. The earlier visual PPO checkpoint scored 55/128 (42.97%).
No task tolerances or randomization ranges were reduced to obtain the final source scores.

Both selected exports have exact tensor equality with their checkpoint actors; TorchScript and ONNX
outputs agree within 1e-6 on the recorded smoke-test inputs. Export checks are recorded alongside the
models. `$ARTIFACT_ROOT/manifest.json` records selected hashes, experiment commands,
software versions and an archived source snapshot. The remaining sim2real work is to explain and
reduce the transfer gap using measured robot dynamics and camera calibration, then validate hardware
success separately from the simulator scores.
