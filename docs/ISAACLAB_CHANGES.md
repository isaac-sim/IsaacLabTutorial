# Isaac Lab dependency changes

The tutorial now pins `StafaH/IsaacLab:fix/review-simulation-training-fixes` at
[`a5edcec217`](https://github.com/StafaH/IsaacLab/commit/a5edcec21741e98c5918f4c8ae88341fcd357a5f),
the inspected head of [PR #8379](https://github.com/isaac-sim/IsaacLab/pull/8379).
The exact revision is pinned in `pyproject.toml` and `uv.lock`.

## October 8 dependency update

Newton now applies explicitly configured `spawn.distortion` using native OpenCV pinhole/fisheye
rays, including non-square focal lengths and an off-center principal point. It still does not read
lens coefficients directly from USD assets. The ray field is shared across environments: differing
per-environment intrinsics are rejected, and a configured distortion model retains fixed calibration.
Consequently the task keeps its 80×60 overscan and per-episode inverse projection to 64×48; removing
that path would remove independently randomized camera geometry. No native distortion is layered
on top of the task's image warp.

This revision requires Newton 1.6.1, MuJoCo/MJWarp 3.12 and Warp 1.17, and brings RSL-RL 5.5.1.
The old Newton 1.5 override fails during scene import (`path_particle_map`), so it was updated too.
The contact initializer now accesses the builder through the simulation backend registry.
Removed OVPhysX body-iteration defaults are replaced by a PhysX-only vial asset property; the
robot retains its authored articulation setting. External-force iteration behavior remains explicit.
The optional OVRTX pin follows the dependency's 0.5.0.377615 version.

Validation on the new pin:

- Repository tests: 100 passed; one deployment test module skipped because the training environment
  does not include OpenCV (deployment uses its separate environment).
- The PR's GPU OpenCV distortion integration test passed for pinhole and fisheye rendering, with
  analytic ground-distance checks. A direct GPU check also confirmed that different intrinsics
  across environments raise the documented error.
- The randomized vision task completed a 16-environment runtime smoke test using Newton physics
  and Newton rendering. State PPO and visual PPO each completed two fresh-initialization updates.
- A 256-first-episode clean audit of the previously qualified fresh-pipeline visual checkpoint
  (seed 7501, full camera/dynamics randomization) scored **216/256 = 84.38%**, with zero episodes
  above 20 N rack force and a 12.61 N peak. Its previous 1,024-episode clean confirmation was
  91.31%. Different batch sizes prevent a paired comparison, but this check does not qualify the
  updated runtime above 90%; retain the old pin for reproducing the published acceptance results.
- Ruff, formatting, spelling and lockfile checks passed. Raw checks and smoke-run artifacts are
  outside Git in `IsaacLabTutorial-artifacts/pr8379_20261008/`.

The new `--deterministic` flag also requests Warp-wide deterministic physics. A frozen-policy
check with that flag failed with `Deterministic counter buffer overflow` in Newton's narrow-phase
collision kernel. README evaluation commands omit it: RSL-RL still uses deterministic inference
actions, while GPU physics is not promised to be bitwise reproducible.

The performance and policy-success reports dated October 7 describe the **previous** dependency,
not a qualification of this new physics/runtime stack.

## Historical fork changes used for the October 7 qualification

The previous pin was `sim2real/newton-camera-training-fixes` at
[`efbbde338`](https://github.com/StafaH/IsaacLab/commit/efbbde338f5222eb759cd67397fdc84c088f6c5c),
based on `c58715b1d` plus `b00fe838d`. The following records describe that revision.

## Newton and shared runtime

- **Reset synchronization:** `ManagerBasedRLEnv` calls `sim.forward()` after automatic and manual
  resets so camera observations see the new articulation poses without advancing physics.
- **External body forces:** Newton holds caller forces across solver substeps and fused decimation.
  Per-substep callbacks do not accumulate stale forces; actuator force writes remain intact.
- **HDR backgrounds:** an explicit camera background is converted from sRGB to linear color and
  applied only to pixels without geometry hits, before ISP processing. Black geometry and the
  default behavior without an explicit background are preserved.
- **Camera calibration warning:** Newton also detects distortion authored on USD camera assets,
  including schemas unavailable to core USD. It warns that it renders a centered square-pixel
  pinhole. This does not implement lens distortion; real-camera calibration still matters.

## Training and configuration

- **Fresh-optimizer continuation:** `--reset_optimizer` restores model weights and iteration while
  retaining the new optimizer configuration. It supports PPO and already-initialized distillation
  checkpoints; ordinary resume is unchanged. A fresh student initialized directly from a PPO
  teacher must use ordinary loading. Play accepts the flag consistently across RSL-RL versions.
  The flag does not independently reset an RND module's internal optimizer.
- **External tasks in workers:** direct/distributed RL entry points load the `isaaclab.tasks`
  entry-point group so downstream task registrations exist on every worker.
- **Literal preset overrides:** numeric/list/other non-string Hydra overrides replace resolved
  preset values instead of being interpreted as preset names.

## Optional PhysX and OVRTX diagnostics

These fixes support transfer experiments. The qualified policies were trained on Newton.

- **Compliant materials:** material randomization preserves negative restitution slots, which
  PhysX uses to encode compliant-contact stiffness, while still randomizing friction.
- **Clone correctness:** OVPhysX falls back to full USD ingestion when runtime replication would
  lose acceleration-spring contact behavior, including materials inside USD instances. This is a
  correctness fallback and can substantially increase startup time for large scenes.
- **Forward kinematics:** `OvPhysxManager.forward()` refreshes articulation transforms after
  state writes without stepping the simulation.
- **Camera isolation:** renderer backends can opt out of sharing an instance across sensors.
  OVRTX does so because its camera bindings belong to the instance.
- **First frame:** each OVRTX camera completes its initial frame after buffers/poses are bound,
  avoiding multi-camera initialization hangs without advancing physics.
- **Repeated scene setup:** existing OVRTX scene-partition attributes are reused safely.

The inherited `b00fe838d` commit additionally exposes OVPhysX solver/friction/CCD settings and
scene-wide/default body/articulation iteration counts, supports a USD Preview Surface fallback
for unavailable MDL materials, and diagnoses filtered contact-sensor body/joint name collisions.
The tutorial's fixed-jaw sensor remains unfiltered because its `gripper` body shares a joint name;
the moving-jaw sensor is filtered to the vial.

## Limits relevant to this task

Newton uses `condim=3`; authored rolling/torsional friction coefficients are therefore inactive.
PhysX uses explicit motor/contact calibration to approximate Newton but is not dynamically
identical. Historical broad-randomization transfer scores were below target; the current visual
policy has not been qualified on PhysX or the real robot (see [results](sim2real/RESULTS.md)).

The dependency includes regression tests and per-package release notes for these changes.
Validation results and the published commit are recorded in the [branch report](CHANGES.md).
