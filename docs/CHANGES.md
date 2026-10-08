# Branch report — 2026-10-07

This report compares the final tree of `overhaul/so101-vial-newton` with upstream `main`
commit `c64024e55fc7c0ed8614a6309b7b1ed2ad1faaf5`, fetched for this review. The SO-101 tutorial
already exists on that main revision; replacing older Jetbot examples is not a new change here.

## Task and training

- Corrected random home resets so XY perturbations cannot place the vial inside the rack. Body/cap
  footprint checks resample only overlapping home starts within the existing ±20 mm bounds, with
  1 mm clearance. Non-home curriculum states and placement tolerances are unchanged. Earlier
  unconstrained-reset scores are distinguished from this corrected profile in the results.

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
- Qualified state and historical visual policies above 90% in randomized Newton simulation. The
  corrected desk scene retains 94.63% state success; vision is not yet qualified with the expanded
  camera/appearance distribution. The historical visual checkpoint includes PPO refinement after
  distillation and must be loaded with the PPO runner.
  [Results](sim2real/RESULTS.md) document selection, audit conditions and the remaining transfer gap.

## Physics, assets and rendering

- Restored the real setup recorded on the prior feature branch: orange printed robot parts with
  black servos, the same yellow rack, and a bare brown desk instead of a green mat. The support
  plane height is retained. Camera-Sim2Real varies those material colors per environment/episode.
  Newton rendering was checked; the plain surface does not yet model actual wood grain.

- Added episode-fixed wrist-camera mounting (±3 mm, ±3°) and projection variation (focal length,
  principal point and mild radial distortion), plus gamma variation. A wider Newton render feeds
  the original 64×48 policy image. This needs no new Isaac Lab patch. The old policy's success drops
  to 49.61% under combined camera variation in a 256-episode diagnostic; historical >90% vision
  scores apply to the original camera profile. [The assessment](sim2real/DOMAIN_RANDOMIZATION.md)
  records research, coverage gaps and the LEAPP/LeRobot deployment contract.

- Bundled the workshop vial/rack USD assets and a local desk asset and their license, rather than using main's remote
  workshop paths. This retains editable contact/material setup; the desk correction enlarges the support footprint.
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

- Added a standalone LEAPP/LeRobot controller with explicit image/history processing, calibrated
  joint transforms, 30 Hz inference and 120 Hz measured-relative target updates. Default execution
  is read-only. An isolated CPU runtime avoids conflicting with Isaac Lab dependencies; the real
  actor matched the training runtime on eight input pairs with zero maximum absolute error.
- Added optional teacher rollout warm-up to the existing distillation algorithm. Probability
  decays to zero over a configured number of iterations; default behavior stays student-only.

- Recorded exact first-episode accounting, criterion version, selected hole, reset row/pose,
  effective randomization, observation corruption, runtime details and checkpoint hashes in audits.
- Extended audits with effective observation parameters, so camera geometry and history settings
  accompany each result. Added an explicit LEAPP visual-actor packager with changing-image runtime
  parity checks; the generic environment exporter did not expose this custom camera term as an input.
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
The tutorial passed **96 tests** across the training and isolated deployment environments,
including optional LEAPP export and deployment-contract tests (93 training-environment tests
and 3 isolated deployment tests).
All three training checks (state PPO, vision PPO, fresh visual distillation) completed 12 iterations.
After rebuilding from the published pin, all three also passed two-iteration training smoke tests.
Required asset validation and lockfile consistency checks passed. Pre-fix source failed 43 of the
new regression cases; the patched source passed them (one case remained skipped).
After rebuilding from the published dependency, the same selected policies scored **93.85% state**,
**91.50% vision**, and **90.92% vision with observation corruption**, each over 1,024 attempts.

The lock deliberately preserves the tested Linux x86-64 numerical stack (PyTorch 2.13/CUDA 13,
MuJoCo/MJWarp 3.11), rather than adopting main's newer MuJoCo stack or its separate multi-platform
PyTorch index selection. Other platforms are not claimed as validated by this branch.
