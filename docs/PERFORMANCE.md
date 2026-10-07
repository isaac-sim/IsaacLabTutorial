# Startup and runtime — 2026-10-07

Measured on one **NVIDIA RTX 6000 Ada (48 GB)** per process, on an AMD Threadripper PRO 7965WX
host. Every process uses `CUDA_VISIBLE_DEVICES=0`, with OpenMP/MKL/USD worker counts set to one.
Newton is revision `2cb4cd0260d8de90f96847c04915a758b2e7cc65`, MJWarp/MuJoCo 3.11.0,
Warp 1.16.0, PyTorch 2.13.0+cu130 and RSL-RL 5.4.1.

## Environment benchmark

Three fresh processes per configuration, 50 warm-up steps and 500 measured vector steps,
seed 73. Asset and kernel caches already existed; these are not first-install compilation times.
Measurements use the actual randomized training environments, including resets, rewards and
observations. Vision includes Newton rendering, image augmentation and two-frame RGB history.
State has no camera or renderer. The policy and optimizer are absent from this benchmark.

| Workload | Environments | Environment creation | Instrumented startup total | Vector step | Transitions/s | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| State, Newton | 4,096 | 6.20 s | 7.41 s | 55.88 ms | 73,298 | 2.23 GB |
| State, Newton | 1,024 | 3.98 s | 5.26 s | 32.59 ms | 31,417 | 0.94 GB |
| Vision, Newton + Newton renderer | 1,024 | 4.95 s | 6.23 s | 45.58 ms | 22,465 | 1.62 GB |

Values are medians across three runs. Instrumented startup sums Python imports, task configuration,
application launch, environment construction and first step as recorded by Isaac Lab's runtime
benchmark; it does not include every outer CLI/bootstrap cost. The first state-4096 run took 9.66 s
for environment creation (11.00 s instrumented startup), versus 5.98–6.20 s on repeats. Vision
creation ranged 4.88–6.95 s. State throughput ranged 69,541–76,005 transitions/s at 4,096 environments;
vision ranged 22,285–22,807 at 1,024.

A transition is one environment's control step. Vector-step latency advances all environments once.
These are host-return timings, not individually CUDA-synchronized kernel timings or robot inference
latency. The environment performs its usual synchronization. Brief dependency tests ran on GPU 3
for part of this measurement window; shared host activity and caching can affect startup variability.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1 \
  uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101-Sim2Real --num_envs 4096 --seed 73 \
  --num_steps 500 --warmup_steps 50 --output_path outputs/benchmark_state \
  --visualizer none presets=newton_mjwarp

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1 \
  uv run isaaclab benchmark runtime \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Sim2Real --num_envs 1024 --seed 73 \
  --num_steps 500 --warmup_steps 50 --output_path outputs/benchmark_vision \
  --visualizer none presets=newton_mjwarp,newton_renderer \
  env.observations.wrist_rgb.image.params.history_length=2
```

Set `TMPDIR` as in the README. Raw JSON, invocations and logs are retained in the external artifact
archive's `performance/` directory, rather than committed as generated benchmark files.

## Actual training

Separate sequential runs exercised the production runners for 12 iterations each, on GPU 0.
PPO resumed the selected checkpoints with fresh optimizers; distillation initialized a **new student**
from the selected state teacher. Each visual run used two RGB frames and the Newton renderer.
These checks validate execution and measure speed; their short continuations do not replace the
qualified checkpoints. Reported throughput is the median of iterations 3–12.

| Runner | Environments | Process start → environment ready | First iteration complete | Rollout / iteration | Learning / iteration | Total / iteration | Transitions/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| State PPO | 4,096 | 9.44 s | 13.33 s | 3.762 s | 0.111 s | 3.89 s | 67,528 |
| Vision PPO | 1,024 | 7.97 s | 13.77 s | 3.444 s | 0.888 s | 4.35 s | 15,074 |
| Vision distillation | 1,024 | 7.96 s | 10.84 s | 1.577 s | 0.265 s | 1.84 s | 17,784 |

PPO collects 64 control steps per environment per iteration; distillation collects 32. The first
iteration column includes model/checkpoint setup plus collection and learning. Environment-ready
timestamps include the outer CLI and process startup and therefore differ from the instrumented
benchmark totals above. These training timings are single runs, not three-run medians; rollout
state and episode resets also affect speed. They exclude checkpoint-write overhead from steady
iteration timing. Distillation learns while the student acts, so its speed is not directly comparable
to a teacher-only rollout.

The benchmark/training installation's runtime source was checked against the published dependency;
only formatting and the wheel builder's asset-directory relocation differed. The final dependency
was then rebuilt and installed from its Git pin, with policy acceptance audits and two-iteration training smoke tests repeated afterward.
