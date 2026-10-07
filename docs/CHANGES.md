# Branch report — 2026-10-07

This report compares the final tree of `overhaul/so101-vial-newton` with upstream `main`
commit `c64024e55fc7c0ed8614a6309b7b1ed2ad1faaf5`, fetched for this review. The SO-101 tutorial
already exists on that main revision; replacing older Jetbot examples is not a new change here.

## Task and training

- Accepted placement in any of the four physical rack openings. Insertion, pose shaping and
  placement-distance features use the nearest opening. The rack-coordinate observation frame,
  six actions, milestone structure and release/stability tolerances remain unchanged.
- Added two shared Sim2Real configurations with 30-second attempts, episode-level contact/mass
  and arm/gripper randomization, ±20 mm home-start vial offsets, and a curriculum weighted toward
  home starts. Evaluation retains the physical distribution but always starts at home.
- Added conservative state PPO continuation defaults. Camera distillation and PPO use the same
  visual inputs, with optional RGB history, intensity normalization, shifts and blur alongside
  episode-consistent photometric augmentation. The qualified visual policy uses two frames.
- Added a small checkpoint initializer that retains a visual encoder while replacing its frozen
  state teacher and can expand one-frame convolution weights to a repeated-frame history.
  Saved metadata records both source hashes. Standard RSL-RL runners perform training.
- Qualified state and visual policies above 90% in randomized Newton simulation. Vision's selected
  checkpoint includes PPO refinement after distillation; it must be loaded with the PPO runner.
  [Results](sim2real/RESULTS.md) document selection, audit conditions and the remaining transfer gap.

## Physics, assets and rendering

- Bundled the workshop vial/rack/mat USD assets and their license, rather than using main's remote
  workshop paths. This retains the exact training geometry and editable contact/material setup.
  Required assets and the reset dataset remain tracked through Git LFS.
- Bound a shared workshop contact material to the robot and scene colliders. PhysX receives the
  compliant acceleration-spring material and explicit solver iteration settings; Newton retains
  its MJWarp soft-contact parameters and `condim=3`.
- Added a PhysX preset with the correct robot USD variant and explicit armature, joint-friction,
  viscous-friction and gripper calibration. These diagnostics approximate Newton; they do not
  establish identical engine behavior or improve the reported source success by loosening tolerances.
- Enabled vial contact reporting and used an unfiltered fixed-jaw sensor to avoid the OVPhysX
  body/joint name collision. The moving-jaw sensor remains vial-filtered.
- Used neutral scene lighting, shadows and a shared camera background configuration. The Newton
  renderer remains the working visual training path; OVRTX is an optional transfer diagnostic.
- Kept reset generation compatible with the pinned framework's simulation-launch API.

## Evaluation and reproducibility

- Recorded exact first-episode accounting, criterion version, selected hole, reset row/pose,
  effective randomization, observation corruption, runtime details and checkpoint hashes in audits.
- Replaced the machine-local Isaac Lab dependency with a published Git commit and pinned the
  tested Newton revision and numerical runtime versions. Optional standalone PhysX/OVRTX extras
  remain separate from the default Newton installation.
- Added tests for all four openings, rejection of divider/rim placements, camera augmentation/history,
  student initialization, Sim2Real configuration and audit metadata.
- Documented actual startup and environment/training throughput in [PERFORMANCE.md](PERFORMANCE.md),
  with hardware, workload and timing boundaries. No benchmark logs or generated policy binaries
  are included in the source tree.

## Cleanup

The final public surface has five task IDs: the three original tutorial tasks plus the randomized
state and camera tasks. Experimental recurrent, RGB-D, extra-camera, visual-state, recording and
one-off Sim2Sim curriculum variants were removed. The common MDP stays shared; Sim2Real randomness
is declared directly in one events configuration. Effective state and camera Sim2Real configurations
were compared before and after this refactor and were identical.

Historical run directories, generated videos, exports, checkpoints and diagnostic reports were
moved to an external artifact archive, preserving the successful models and evidence. Three old
tracked experiment videos and their recording utility were removed from the final branch tree.
Original tutorial demonstration media and required task assets remain. This cleanup does not rewrite
historical Git commits. Ignore rules now exclude generated policies and retain the reset dataset.
The README presents only supported tasks and executable Newton workflows.

## Isaac Lab

The separately published dependency branch is `StafaH/IsaacLab:sim2real/newton-camera-training-fixes`.
[ISAACLAB_CHANGES.md](ISAACLAB_CHANGES.md) describes every framework change, including the inherited
solver/material compatibility commit. In brief: reset kinematics, Newton force lifetime and HDR
backgrounds, camera warnings/isolation/initialization, compliant material preservation and clone
fallback, fresh-optimizer checkpoint loading, external task registration, and literal preset overrides.

The dependency is published as [`efbbde338`](https://github.com/StafaH/IsaacLab/commit/efbbde338f5222eb759cd67397fdc84c088f6c5c).
Its changed areas and inherited compatibility tests passed **518 tests**, with **6 skips** for
unsupported/unavailable cases. The repository's full formatter/pre-commit checks passed.
The tutorial passed **83 tests** and lint/format checks after installation from the published pin.
All three training checks (state PPO, vision PPO, fresh visual distillation) completed 12 iterations.
After rebuilding from the published pin, all three also passed two-iteration training smoke tests.
Required asset validation and lockfile consistency checks passed. Pre-fix source failed 43 of the
new regression cases; the patched source passed them (one case remained skipped).
After rebuilding from the published dependency, the same selected policies scored **93.85% state**,
**91.50% vision**, and **90.92% vision with observation corruption**, each over 1,024 attempts.

The lock deliberately preserves the tested Linux x86-64 numerical stack (PyTorch 2.13/CUDA 13,
MuJoCo/MJWarp 3.11), rather than adopting main's newer MuJoCo stack or its separate multi-platform
PyTorch index selection. Other platforms are not claimed as validated by this branch.
