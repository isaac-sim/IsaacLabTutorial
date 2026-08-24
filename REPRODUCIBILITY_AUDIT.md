# Reproducibility audit

Date: 2026-08-24
Scope: install/lock state, task registration, assets, reset data, state PPO, native vision distillation, exact
evaluation repeatability, checkpoint/video integrity, and copy-paste documentation.

## Outcome

- State training is reproducible. Three complete 4,096-environment seeds passed exact canonical evaluation at
  97.17%, 99.41%, and 99.61%.
- The retained distilled vision checkpoints are valid inference artifacts, but training is not reproducible from the
  documented recipe. This distinction is intentional in the README and manifest.
- Vision-from-scratch remains a diagnostic policy and was not resumed in this state/distillation regression audit.
- All five retained MP4 files decode, have the declared frame count/resolution, contain non-constant frames, and
  match their manifest hashes.

## Dependencies and configuration

- `uv lock --check` passes.
- `uv sync --frozen --dry-run` reports no changes.
- A detached worktree at the audit commit created a new virtual environment with `uv sync --frozen`, installed all
  125 locked packages, passed all 90 tests and Ruff, and completed a two-update 4,096-environment state-training
  smoke run at about 84k simulator steps/s.
- The lockfile pins every Isaac Lab package to commit `fa4de7d35b64db94e3ce004e1d4640cf0de0507e`, matching the
  manifest. The unrelated adjacent `../IsaacLab` checkout is newer, but the virtual environment imports the pinned
  wheel rather than that checkout.
- Every registered SO-101 task and agent configuration entry point resolves in an automated test.
- Task/reward/training source contains no `SO101_*` shell configuration reads. Obsolete environment-variable
  commands were removed from the historical handoff tail.
- Each experiment was scoped with `CUDA_VISIBLE_DEVICES=<one GPU>` and addressed that job's GPU as `cuda:0`; no
  distributed or multi-GPU optimizer was used.

## Reset artifacts

The main dataset is 96,842 bytes (1,024 rows); the diagnostic bridge is 164,514 bytes (1,781 rows). Both are now
ordinary Git blobs rather than Git LFS objects, so a fresh checkout does not need LFS to obtain the reset states.
Their file hashes and tensor-content hashes are checked by the test suite.

The documented generator command was corrected: the custom `generate_resets` CLI does not accept
`presets=newton_mjwarp`. Two valid same-seed generation runs on one RTX 6000 Ada took 173.78 and 178.15 seconds and
about 2.05 GB maximum host RSS. Both produced schema-valid, balanced 1,024-row artifacts, but their content hashes
were different from each other and from the tracked artifact:

| Artifact | Content SHA-256 |
|---|---|
| Tracked training reset data | `b2709b9b4549f788769d183a039b62d0d80e918990b9bb145b2c0ef57c397a00` |
| Fresh seed-42 generation A | `98250f726266fa8e625e57a70bf59364524059fd723a60847f1d5a77d5a84c46` |
| Fresh seed-42 generation B | `931e2e3f77df768df8ea4da9201b2ff89245e64a594f2103cbe2893170b1cccb` |

This is expected GPU contact/acceptance variation, but it means generation is a maintainer workflow after an
intentional physics/reset change—not a fresh-user installation step. Training reproducibility uses the tracked,
hash-checked dataset.

## Exact inference repeatability

The evaluator always counts 1,024 completed canonical-home episodes. A deterministic actor does not make Newton GPU
contact/render execution bitwise deterministic, so repeated rates can differ slightly.

| Policy | Recorded audit | Audit seed 99 / repeat | Finding |
|---|---:|---:|---|
| Fresh state seed 42 | 97.17% | 97.36% / 97.46% | Stable, 0 unsafe impacts |
| Recovered historical state teacher | 100.00% | 100.00% | Stable, 0 unsafe impacts |
| Distilled vision seed 42 | 92.87% | 92.38% / 93.26% | Retained inference remains above 90% |
| Distilled vision seed 45 | 90.82% | 89.94% | Near-threshold artifact is not robustly above 90% |

The state result has substantial acceptance margin. A distilled result near 90% should be evaluated repeatedly and
must not be promoted based on one favorable run.

## Training evidence

### State PPO

The recipe is ordinary PPO: 400 full-horizon updates, 200 continued full-horizon updates with optimizer state, then
100 canonical-home polishing updates. Each run uses 4,096 environments on one GPU. Existing independent from-zero
results are:

| Seed | Exact canonical success | Grasp | Lift | Unsafe rack impact |
|---:|---:|---:|---:|---:|
| 42 | 97.17% | 100.00% | 100.00% | 0.00% |
| 45 | 99.41% | 99.61% | 99.61% | 0.00% |
| 43 | 99.61% | 100.00% | 100.00% | 0.00% |

Seed 43 was trained during this audit from random initialization with exactly the README recipe. Its three stages
took 1,213.09, 584.54, and 276.49 seconds (34m34s total, excluding startup), and the retained final checkpoint is
`checkpoints/candidates/state_vial_farther_repro_seed43_model697.pt`.

### Native state-to-vision distillation

Earlier fresh audits already rejected both the purported one-stage 4,096-environment recipe (5.27%/43.65%) and a
reconstructed five-stage 1,024-environment recipe (45.41%/42.68%, best dense checkpoint 45.90%). The retained
>90% checkpoints were proven to be two five-update branches from one historical parent, not independent seeds.

This audit additionally trained a fresh seed-46 teacher-controlled visual warm start for all 600 configured updates
at 1,024 environments. Its exact canonical result was only 5.76% success, 48.93% grasp, and 20.31% lift. Therefore
the reproducibility gap is already present in the initial visual behavior-cloning stage; it cannot be attributed
only to later student-visited DAgger updates.

## Automated and artifact checks

- Full suite: 90 tests passed, including the expanded registration and artifact-integrity coverage.
- Ruff passes.
- All accepted and diagnostic checkpoint SHA-256 values match the manifest.
- Both reset artifacts load through the production schema validator and match the manifest.
- All five videos decoded at 30 FPS. Frontal rollouts are 1920x1080; wrist videos are the intended 64x64.
- Play creates `checkpoints/candidates/exported/` as an upstream side effect. That generated directory is now ignored
  so exact evaluation no longer dirties the working tree.

## Known limitations

- Newton occasionally exits 139 during scene creation. In this audit, one state launch and one camera launch failed
  before environment creation and succeeded when the exact command was retried. This is startup flakiness, not a
  learned-policy failure, but unattended automation should retry startup-only exit 134/139 failures.
- Exact simulation is statistically repeatable, not bitwise repeatable. Use complete episodes, multiple seeds, and
  adequate acceptance margin.
- Distillation inference is useful; distillation training remains the unresolved regression. Do not describe it as
  end-to-end reproducible until independent from-zero runs pass.
