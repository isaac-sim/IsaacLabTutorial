# Startup and runtime — 2026-10-07

The latest measurements use the final orange-robot/brown-desk scene, collision-free home sampling,
full randomization and the camera housing/cache fixes. Each process sees one **RTX 6000 Ada (48 GB)**
through `CUDA_VISIBLE_DEVICES`; the shared host is a Threadripper PRO 7965WX with 24 cores.
OpenMP, MKL and USD worker counts are one.

The pinned runtime is Newton 1.5 revision `2cb4cd0260d8de90f96847c04915a758b2e7cc65`,
MuJoCo/MJWarp 3.11, Warp 1.16, PyTorch 2.13.0+cu130 and RSL-RL 5.4.1. Isaac Lab is the
[published dependency](ISAACLAB_CHANGES.md). State has no renderer; vision uses the Newton renderer.

## Environment startup and throughput

Three fresh processes per workload, each with 50 warm-up steps and 500 measured vector steps,
seed 73. The three workloads ran simultaneously on separate GPUs, sharing the CPU. Asset/kernel
caches already existed. Values below are medians across the three runs.

| Workload | Environments | Environment creation | Instrumented startup | Vector step | Transitions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| State, Newton | 4,096 | **6.04 s** | **8.33 s** | 57.92 ms | **70,721** |
| Vision, Newton + Newton renderer | 1,024 | **4.96 s** | **7.22 s** | 47.26 ms | **21,667** |
| Vision, Newton + Newton renderer | 2,048 | **6.23 s** | **8.49 s** | 70.14 ms | **29,198** |

Instrumented startup sums the benchmark's imports, task configuration, application launch,
environment construction and first step. It excludes outer CLI/bootstrap work and is not a
first-install compilation measurement. Startup ranges were 8.25–8.66 s for state, 7.10–7.81 s for
vision-1,024, and 8.40–8.95 s for vision-2,048. Earlier launches before these final measurements
occasionally took approximately 30 s; cached startup is not a guaranteed cold-start bound.

A transition is one environment's control step; vector latency advances all environments once.
These host-return timings include simulation, rendering, observations, rewards and resets, but
**exclude policy inference and optimizer updates**. Vision renders 80×60, projects to 64×48,
stacks two raw-RGB frames, and retains training augmentation and material variation.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1 \
  uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101-Sim2Real --num_envs 4096 --seed 73 \
  --num_steps 500 --warmup_steps 50 --output_path outputs/benchmark_state \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1 \
  uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real --num_envs 1024 --seed 73 \
  --num_steps 500 --warmup_steps 50 --output_path outputs/benchmark_vision \
  --visualizer none presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2 \
  env.observations.wrist_rgb.image.params.normalize_intensity=False
```

Use `--num_envs 2048` for the larger visual workload and the README's `TMPDIR` setting.
Raw reports and commands are outside Git under `camera_randomization_20261007/performance_final/`.

## Actual training

All three production runners completed checks on the final scene. PPO resumed the selected state
or visual checkpoint; distillation initialized a fresh visual student from the state teacher.
These checks establish training execution and throughput, not a replacement for the frozen audited
policy. State PPO and distillation ran concurrently; the visual PPO check overlapped other audit
activity. Shared-host load and episode behavior affect throughput.

| Runner | Environments | Rollout / iteration | Learning / iteration | Total / iteration | Transitions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| State PPO | 4,096 | 4.063 s | 0.115 s | **4.185 s** | **62,660** |
| Vision PPO | 2,048 | 5.097 s | 1.868 s | **6.960 s** | **18,836** |
| Vision distillation | 1,024 | 1.707 s | 0.255 s | **1.965 s** | **16,706** |

PPO collects 64 steps per environment per iteration; distillation collects 32. Values are medians
of iterations 3–12 for state/distillation and 3–20 for visual PPO, excluding the initial warm-up
iterations. Collection includes policy execution. Learning and rollout medians need not sum exactly
to the median total. Logs are `fixed_state_training_check`, `fixed_scene_training_check` and
`fixed_distillation_check` in the same external artifact directory. Earlier profile measurements
remain archived, but should not be substituted for these current-scene timings.

## LEAPP deployment compute

The selected raw-RGB visual bundle was checked in the isolated **Torch 2.10 CPU** / LeRobot runtime.
Eight varied input pairs matched the Torch 2.13 training actor with **zero maximum absolute error**.
The controller benchmark uses a synthetic 640×480 RGB frame, one CPU thread, batch size one,
30 warm-up calls and 1,000 measured calls:

- Image resize, history/proprioception assembly, LEAPP inference and target calculation:
  **0.393 ms median / 0.405 ms p95**.
- Bundle/controller construction: **8.95 ms**, after imports and an earlier parity-test load.
- First controller preprocessing/inference call: **14.06 ms**, with model execution already warmed
  by the parity test. This is not a cold-process first inference.

Camera capture, USB/exposure latency, motor-bus I/O and Python imports are excluded. The measurement
ran while a visual training process was active. It shows that network computation is inexpensive
on this CPU; it does not establish end-to-end latency or 120 Hz robot-bus throughput.
Raw results accompany the selected bundle in `selected/vision/deployment_compute_benchmark.json`.
See [DEPLOYMENT.md](sim2real/DEPLOYMENT.md) for the 30 Hz policy / 120 Hz target-update contract.

The independently trained fresh-pipeline bundle was checked again with the same isolated CPU
protocol, with all training GPUs idle. Its eight cross-runtime inputs also matched exactly.
Preprocessing, inference and target calculation measured **0.390 ms median / 0.404 ms p95**;
construction took 8.77 ms and the first controller call took 13.61 ms after the parity warm-up.
These retain the exclusions above. Results are in
`from_scratch_20261007/selected/vision/deployment_compute_benchmark.json`.
