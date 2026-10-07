# Isaac Lab dependency changes

The tutorial pins the separately published `StafaH/IsaacLab` branch
`sim2real/newton-camera-training-fixes`. Published commit: [`efbbde338`](https://github.com/StafaH/IsaacLab/commit/efbbde338f5222eb759cd67397fdc84c088f6c5c).
The exact revision is also pinned in `pyproject.toml` and `uv.lock`.
It is based on `c58715b1d`, plus the existing `b00fe838d` compatibility commit and the fixes below.
These are framework changes; task randomization, rewards and observations remain in the tutorial.

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
identical. The current broad-randomization transfer scores remain below target (see
[results](sim2real/RESULTS.md)); none of these fixes establish real-robot success.

The dependency includes regression tests and per-package release notes for these changes.
Validation results and the published commit are recorded in the [branch report](CHANGES.md).
