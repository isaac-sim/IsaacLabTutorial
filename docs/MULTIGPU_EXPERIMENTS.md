# SO-101 experiment handoff to a multi-GPU machine

> Historical local-branch record. The current consolidation status and commands are tracked in
> [sim2real/CONSOLIDATION.md](sim2real/CONSOLIDATION.md) and the repository README.

Branch: `feat/so101-sim2real-multigpu`. This is an experiment baseline, not an accepted
sim2real policy. The [working log](SO101_SIM2REAL.md) records the completed setup,
physics changes, failed experiments, historical scores and remaining physical checks.

## Install the pinned environment

Use a Linux NVIDIA GPU machine with a driver compatible with the pinned CUDA 13
PyTorch stack. This branch locks Isaac Lab and the dependencies used on the local
machine; do not update them while comparing experiments.

```bash
git clone --branch feat/so101-sim2real-multigpu https://github.com/isaac-sim/IsaacLabTutorial.git
cd IsaacLabTutorial
uv sync --locked --extra sim2real
nvidia-smi
uv run --locked --extra sim2real python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

The small required reset dataset is tracked in ordinary Git. Workshop USD assets
are fetched through Isaac Lab's configured asset root. Logs, trained checkpoints,
camera captures and large demonstration datasets in `logs/` and `outputs/` are
local artifacts and are **not included in the branch**. Historical diagnostic
commands using `/tmp/so101_*.py` are not portable training recipes. The commands
below use only committed code and start fresh runs.

## Verify one GPU before scaling

First verify a short camera training run on the new machine:

```bash
uv run --locked --extra sim2real so101 train \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 32 --max_iterations 2 --seed 70 \
  preset=sim2real agent.run_name=sim2real_single_gpu_smoke
```

Then check the distributed launch command without starting workers. Set
`--num_gpus` to the number of visible GPUs; the launcher checks that count even
with `--dry_run`:

```bash
uv run --locked --extra sim2real so101 train_multigpu \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_gpus 2 --num_envs 256 --max_iterations 5000 --seed 70 \
  --dry_run preset=sim2real agent.num_steps_per_env=128 \
  agent.run_name=sim2real_camera_multigpu
```

Remove `--dry_run` to train. `--num_envs` is **per GPU**: two ranks with 256
environments collect 512 environments overall. Start with 256 per rank for the
recurrent camera model and adjust after checking GPU memory. Changing rank count
also changes the global PPO batch, so record it alongside the seed and task config.
The launcher supplies `--distributed`; do not launch separate independent trainers
and treat them as a distributed run. Verify that every rank reaches initialization
and synchronization, and that the run finishes with a checkpoint and exit code 0.

For a fresh privileged state teacher, use the same launcher with
`--task IsaacTutorial-Place-Vial-SO101`; its observations and checkpoints differ
from the camera task. State success does not qualify the visual policy.

## Evaluate the resulting camera checkpoint

Use the exact checkpoint path from the run, not an unrelated latest checkpoint:

```bash
uv run --locked --extra sim2real so101 play \
  --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera \
  --num_envs 1024 --seed 71 --checkpoint /path/to/model_ITERATION.pt \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  preset=sim2real
```

This audit measures 1024 independent randomized home-start episodes. Play retains
physics/material/reset randomization but disables observation corruption; run a
separate test with sensor corruption before claiming robustness to those inputs.
Physical success now accepts any empty opening. Teacher/reward supervision latches
an in-frame opening near the image centre, with rack yaw limited to ±15 degrees.
Positions remain broadly randomized and vial heading/spin remain unrestricted.

Record complete placement, grasp/lift/insertion, timeouts, vial loss and rack forces;
pickup alone is not task success. Repeat with independent seeds. Only export an
accepted camera checkpoint using the LEAPP command in the README. Physical joint
mapping, camera/rack alignment and supervised real trials remain required.

## Validation performed before this handoff

The local single-GPU scene/training pipeline runs and 80 software tests pass. The
multi-GPU launcher dry run was verified with one visible GPU; requesting two on
this machine correctly failed its device-count check. Actual multi-rank training,
NCCL communication and scaling must be verified on the multi-GPU host. No
machine-specific NCCL overrides are committed.
