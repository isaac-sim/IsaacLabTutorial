# SO-101 vial placement

<p align="center"><img src="media/demo.gif" alt="Original SO-101 workshop demonstration" width="100%"></p>

The animation shows the original workshop appearance. Current Sim2Real scenes use the orange robot
and bare brown desk.

This Isaac Lab tutorial trains an SO-101 arm to place a vial in **any of four rack holes**.
The working training path uses **Newton MJWarp physics and the Newton renderer** for vision.
State training needs no renderer. The state teacher scores **96.78%**. The visual policy scores
**94.43% clean / 94.04% with observation noise** on fresh 1,024-episode audits, using the orange
robot, brown desk and full camera/appearance/dynamics randomization. A separate end-to-end run
from random state and visual weights achieves **92.87% state / 91.31% clean and noisy vision**.
See [results and evaluation protocol](docs/sim2real/RESULTS.md),
[startup and runtime measurements](docs/PERFORMANCE.md),
[branch changes](docs/CHANGES.md), and [Isaac Lab dependency changes](docs/ISAACLAB_CHANGES.md).
These simulator results do not establish real-robot performance.
See the [camera randomization and deployment assessment](docs/sim2real/DOMAIN_RANDOMIZATION.md)
for measured camera sensitivity, research sources, and the LEAPP/LeRobot control contract.

## Setup

The validated installation targets Linux x86-64 with Python 3.12, Git LFS, an NVIDIA GPU,
a CUDA 13-compatible driver, and `uv`. The dependency revisions are pinned in
`pyproject.toml` and `uv.lock`; no neighboring source checkout is required.

```bash
git lfs pull
uv sync --locked
uv run pytest -q
uv run ruff check src tests
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1
```

The local temporary directory avoids shared-machine permissions on downloaded assets.
The worker limits above were used for four simultaneous single-GPU experiments.
The package registers its tasks through `isaaclab.tasks` and uses Isaac Lab's standard launchers.

| Task suffix after `IsaacTutorial-Place-Vial-SO101` | Inputs | Training |
| --- | --- | --- |
| (none) | State | PPO teacher bootstrap |
| `-Camera` | Wrist RGB + proprioception | PPO from scratch |
| `-Camera-Distillation` | Wrist RGB + proprioception | State-teacher distillation |
| `-Sim2Real` | State | Randomized PPO continuation |
| `-Camera-Sim2Real` | Wrist RGB + proprioception | Randomized distillation or PPO |

## State teacher

Bootstrap a teacher, then continue under broader randomization. Set `STATE_CHECKPOINT` to the
checkpoint you want to continue; paths below are placeholders, not bundled trained models.
The first command has no checkpoint argument: it initializes random actor and critic weights.

```bash
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --max_iterations 800 \
  --seed 42 --run_name state --visualizer none presets=newton_mjwarp

export STATE_CHECKPOINT=/absolute/path/to/state/model.pt
CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Sim2Real --num_envs 4096 \
  --checkpoint "$STATE_CHECKPOINT" --max_iterations 200 \
  --seed 42 --run_name randomized_state --visualizer none presets=newton_mjwarp \
  agent.algorithm.learning_rate=3e-4 agent.algorithm.schedule=adaptive \
  agent.algorithm.gamma=0.999 agent.algorithm.entropy_coef=0.005
```

The fresh-state validation reached 92.87% on 1,024 randomized home-start attempts after
800 bootstrap and 200 randomized updates. Audit after each 200-update block and retain a qualified
checkpoint; curriculum training success is not a home-start acceptance score.
Checkpoints and configuration snapshots go to `logs/rsl_rl/`. Each experiment sees one GPU;
use separate values of `CUDA_VISIBLE_DEVICES` for parallel experiments.

## Camera student

Distill the randomized teacher using RSL-RL's DAgger runner. The student acts; the frozen teacher
labels visited states. Labels are clipped to the same `[-1, 1]` range as executed actions.
This initializes a **new random visual student** and loads only the frozen state teacher.
Do **not** pass `--reset_optimizer` when initializing a new student directly from a PPO teacher.
After the randomized-state stage, update `STATE_CHECKPOINT` to its resulting checkpoint.

```bash
export STATE_CHECKPOINT=/absolute/path/to/randomized_state/model.pt
CUDA_VISIBLE_DEVICES=1 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real --num_envs 1024 \
  --checkpoint "$STATE_CHECKPOINT" --max_iterations 400 --seed 46 \
  --run_name randomized_distillation --visualizer none \
  presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2 \
  env.observations.wrist_rgb.image.params.normalize_intensity=False
```

To reuse a visual encoder with a new teacher, create a distillation checkpoint. This also supports
expanding a one-frame encoder to a two-frame history without changing its initial response to
repeated frames. Then train that checkpoint with `--reset_optimizer` and the matching history.

```bash
uv run python -m isaaclab_tutorial.utils.initialize_distillation \
  --teacher "$STATE_CHECKPOINT" --student /absolute/path/to/visual/model.pt \
  --output outputs/student_init.pt --history 2
```

The current selected visual model uses raw RGB, two-frame history and home-start PPO refinement
at learning rate `1e-4` and discount `0.999`. It exceeds 94% on fresh clean and noisy
1,024-episode Newton audits; see the exact profile in [RESULTS.md](docs/sim2real/RESULTS.md).
Convert the distillation checkpoint before starting PPO. The converter copies the student actor
and the state teacher's privileged critic, records source hashes, resets the iteration count and
starts without an optimizer. No pretrained visual PPO checkpoint is required.

```bash
export STUDENT_CHECKPOINT=/absolute/path/to/distillation/model.pt
export PPO_INITIALIZATION=/absolute/path/to/new/visual_ppo_init.pt
uv run python -m isaaclab_tutorial.utils.initialize_ppo \
  --teacher "$STATE_CHECKPOINT" --student "$STUDENT_CHECKPOINT" --output "$PPO_INITIALIZATION" --action-std 0.2
CUDA_VISIBLE_DEVICES=1 uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real \
  --agent rsl_rl_ppo_cfg_entry_point --num_envs 2048 \
  --checkpoint "$PPO_INITIALIZATION" --reset_optimizer --max_iterations 200 \
  --seed 46 --run_name visual_ppo --visualizer none \
  presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2 \
  env.observations.wrist_rgb.image.params.normalize_intensity=False \
  agent.algorithm.learning_rate=1e-4 agent.algorithm.schedule=fixed \
  agent.algorithm.gamma=0.999 agent.algorithm.entropy_coef=0.001 \
  env.events.reset_from_dataset.params.phase_weights=[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
```

This fresh-teacher recipe was validated with 400 distillation and 200 PPO updates, audited in
200-update distillation and 100-update PPO blocks. It scored 91.31% on both fresh clean and noisy
1,024-episode confirmations. Audit your resulting checkpoint; arbitrary initializations need not
reproduce the same score. The earlier selected policy remains stronger at 94.43% / 94.04%.

## Evaluation

Sim2Real play mode starts from home and **retains physical and camera-geometry randomization**. It disables observation
corruption by default; enable it explicitly for a camera/noise stress audit. An attempt lasts up to
30 seconds. The callback counts exactly one first episode per environment and saves audit metadata.

```bash
export STATE_CHECKPOINT=/absolute/path/to/trained/randomized_state/model.pt
export VISION_CHECKPOINT=/absolute/path/to/trained/visual_ppo/model.pt
CUDA_VISIBLE_DEVICES=0 SO101_EVALUATION_OUTPUT=outputs/state_audit.json \
  uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Sim2Real --num_envs 1024 --seed 7102 \
  --checkpoint "$STATE_CHECKPOINT" --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=1 SO101_EVALUATION_OUTPUT=outputs/vision_audit.json \
  uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real \
  --agent rsl_rl_ppo_cfg_entry_point --num_envs 1024 --seed 7501 \
  --checkpoint "$VISION_CHECKPOINT" --deterministic \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  --visualizer none presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2 \
  env.observations.wrist_rgb.image.params.normalize_intensity=False
```

Use the PPO agent entry point for the selected PPO-refined visual checkpoint. A distillation
checkpoint uses the default distillation runner. Camera history must match the checkpoint.
The randomized camera task now defaults to the selected two-frame raw-RGB preprocessing; older
single-frame or intensity-normalized checkpoints require explicit matching overrides.

## Task design

The policy produces actions at 30 Hz. At each 120 Hz physics substep, five actions set arm targets
up to 0.033 rad from measured joint positions and one sets the jaw target up to 0.02 rad away.
The same policy action is held across four substeps; the measured-position reference is refreshed.
The vial remains a free rigid body.
State observations have 60 values. Vision uses 24 proprioceptive values and 48×64 RGB frames;
the selected model stacks two raw RGB frames oldest first, scaled from bytes to [0, 1].
Repeat the first frame to initialize history after reset. Exports include learned normalization,
but the robot controller still needs action clipping and joint-target integration.

Three physical milestones—grasp, lift, insertion—pay once per episode. Dense approach progress
and held-vial pose error guide learning. Insertion and placement distance use the closest opening;
there is no selected-hole action or extra task stage. Success requires a seated, upright, released,
nearly motionless vial for ten consecutive control steps. The original tolerances remain intact.

Both randomized tasks share vial mass/contact and motor variation, plus fresh ±20 mm home-start
vial offsets. Vision adds episode-consistent camera mounting/intrinsic variation, color/exposure/gamma
variation, blur and noise. Overscanned 80×60 rendering supplies the randomized 64×48 policy view.
The [randomization assessment](docs/sim2real/DOMAIN_RANDOMIZATION.md) lists all physical ranges. Training samples the eight-phase
reset dataset with extra home starts; evaluation uses home starts only.

The source lives in `src/isaaclab_tutorial/tasks/place_vial/`: shared MDP terms, reset handling,
and five configurations under `config/so101`. Required workshop USD assets and the reset dataset
are versioned; trained checkpoints, logs, exports and recordings are generated artifacts and are
excluded from Git. The original demonstration media remain as tutorial material.

To inspect or regenerate a reset dataset without replacing the packaged one:

```bash
CUDA_VISIBLE_DEVICES=0 uv run generate-so101-resets \
  --output outputs/reset_poses.pt --visualizer none presets=newton_mjwarp
CUDA_VISIBLE_DEVICES=0 uv run view-so101-resets \
  --dataset outputs/reset_poses.pt --visualizer newton presets=newton_mjwarp
```

Optional PhysX/OVRTX transfer diagnostics use `uv sync --extra ovphysx --extra ovrtx` and
`presets=physx,ovrtx`. They are not the qualified training backend; historical transfer was below target and the current visual policy has not been qualified on them.
