# Vision-from-scratch status and restart plan

Date: 2026-08-24

This is the authoritative pause point for the SO-101 vial-placement work. State training is reproducibly solved.
Retained state-to-vision distillation inference is solved, but fresh distillation training is not reproducible after
cleanup. Vision from scratch is not solved. The best retained scratch checkpoint reliably acquires and lifts the vial
from the real canonical home pose, but never inserts or releases it.

## Retained results

| Policy | Checkpoint | Exact canonical result |
|---|---|---|
| State | `checkpoints/candidates/state_vial_farther_repro_seed45_model697.pt` | 99.41% success over 1,024 episodes; independent seed 42 reached 97.17% |
| Distilled vision, seed 42 | `checkpoints/candidates/distilled_geometry_spatial_seed42_model_1109.pt` | 92.87% success over 1,024 episodes |
| Distilled vision, seed 45 | `checkpoints/candidates/distilled_geometry_spatial_seed45_model_1109.pt` | 90.82% success over 1,024 episodes |
| Scratch vision (diagnostic) | `checkpoints/candidates/vision_transport_alignment_m1850_std003.pt` | 0% success, 97.56% grasp, 96.48% lift over 1,024 episodes |

The retained distilled checkpoints are both five-update `3e-5` branches of the same historical seed-45 parent, not
independent from-zero seeds. A fresh one-stage audit reached 5.27% and 43.65%. A reconstruction of the historical
five-stage, 1,024-environment recipe reached 45.41% and 42.68%; none of 22 densely audited refinement checkpoints
exceeded 45.90%. See `README.md` and `checkpoints/manifest.json` for exact commands and provenance.

The vision actor uses one fixed 64x64 wrist RGB image plus 24 proprioceptive values (joint position, joint velocity,
joint target, and previous action). Object state, rack state, contacts, reset phase, and milestone state are not actor
inputs. The privileged critic is training-only. The compact actor is a two-layer ELU CNN with spatial softmax and a
256/256/128 ELU MLP.

The retained scratch checkpoint has SHA-256
`bda3b4522ec6c197962590df5df7ddb225271ea853e43910ed9300b1fc0d3016`. Its corrected canonical audit measured:

- 1,024 exact canonical episodes;
- 97.56% grasp and 96.48% lift;
- 96.09% true transport clearance (lowest vial point above the rack-clearance plane);
- 0% held insertion and 0% placement success;
- mean upright alignment 0.543 and mean rack-center radial error 18.24 cm;
- 3.52% vial loss and no rack contact.

On all 885 connected phase-four bridge rows it retained 100% grasp/lift and 100% true clearance, but alignment was
only 0.545 and radial error was 17.02 cm. This makes it the strongest honest canonical-start scratch baseline, not a
solved placement policy. Example frontal and wrist-camera rollouts are in `checkpoints/videos/vision_scratch/`.

## What is complete

- The vial starts farther from the rack so the fabricated wrist camera can see the relevant geometry. The camera
  mount itself was not moved.
- Canonical resets use the real home pose with the vial outside the open jaws; no near-vial or between-jaws start is
  accepted.
- State PPO is solved with a small physical MDP and a hardcoded 0.033-rad arm increment. The reproducible recipe is
  600 full-horizon updates followed by 100 canonical-home polishing updates; two fresh seeds independently passed
  the 1,024-episode acceptance gate.
- Two retained spatial-softmax distillation actors solve deployment inference with wrist RGB plus proprioception;
  their shared-parent training provenance and failed fresh reproduction are now documented explicitly.
- State, distilled, and current scratch rollout videos exist.
- Exact evaluators distinguish grasp, lift, true rack clearance, held insertion, release, and final seating. A former
  `above_seated_rate` diagnostic measured root height, not transport clearance; conclusions based on it were
  corrected.
- All training commands use hardcoded configuration values. Each experiment is scoped with
  `CUDA_VISIBLE_DEVICES=<physical GPU>` and still addresses `--device cuda:0` inside that one-device namespace.

## Scratch experiments and findings

### What learned useful skills

The original simple curriculum from random initialization did learn real canonical visual acquisition. Grasp became
visible around canonical-close iterations 400--500, reached 100% in the acquisition stage, and reached 96--97% exact
canonical lift after the lift-progress stage. A separate phase-five policy reached 90.3% exact insertion/release from
downstream resets. The unresolved problem is connecting post-lift reorientation/transport to acquisition without
forgetting acquisition.

Spatial softmax helped substantially and should remain the default. The current 64x64 image contains enough task
detail; increasing resolution is not the next lever. The state critic and geometry diagnostics show that the wrist
view is valid after moving the vial spawn, although rack occlusion still makes post-lift geometry the hard region.

### Approaches that did not solve the connection

- Uniform full-horizon scratch PPO learned downstream reset phases while remaining at 0% canonical grasp. Runs as
  long as 700--800 updates still had 0% canonical grasp and about 100% vial loss.
- A 90%-canonical/full-horizon reset mixture also produced misleading aggregate training metrics and 0% exact
  canonical grasp.
- Joint behavior cloning of acquisition, transport, insertion, and release specialists did not produce one coherent
  closed-loop policy. Geometry-augmented BC, one DAgger relabel pass, and several learning-rate ablations also failed.
- Full-network PPO at arm noise 0.10 forgot lift within 25 updates. Noise 0.03 preserved lift and made small radial
  progress, but did not reorient or insert. Fixed rates 3e-5 and 1e-5 were too conservative; 1e-4 was the useful
  refinement rate.
- Residual, post-lift, split-gripper, frozen-output, bottleneck, and proximal-compensation variants did not connect
  the skills. They either accumulated drift, specialized to one reset distribution, or changed the shared action
  map enough to destroy acquisition.
- Checkpoint interpolation and output-row surgery exposed a real tradeoff but did not yield a canonical policy.
  A full-network interpolation reached alignment 0.662 with 97.3% bridge clearance yet only 0--2.2% canonical grasp.
  Even changing only wrist-flex output rows could reduce canonical grasp to 0%.
- The dense product `vertical_alignment * clearance_progress` bifurcated into upright-but-low and high-but-tilted
  specialists. Higher proximal exploration demonstrated that both behaviors are learnable, but not jointly under
  that objective.

### Final four-run batch

The last batch started from the untouched acquisition actor, used ordinary frozen-normalization PPO, 4,096
environments, 25 updates, and fixed mixtures of canonical and connected bridge resets. It ablated 50/50 versus
75/25 canonical/bridge occupancy and upright-lift weights 5 versus 20.

| Reset mix / weight | Exact canonical grasp | Exact canonical lift | Success | Bridge alignment | True bridge clearance |
|---|---:|---:|---:|---:|---:|
| 50/50, weight 5, seed 935 | 0.0% | 0.0% | 0.0% | 0.247 | 97.97% |
| 50/50, weight 20, seed 932 | 58.59% | 33.79% | 0.0% | 0.529 | 100% |
| 75/25, weight 5, seed 933 | 29.79% | 0.0% | 0.0% | 0.461 | 100% |
| 75/25, weight 20, seed 934 | 0.0% | 0.0% | 0.0% | 0.271 | 100% |

These runs reject fixed reset mixtures as the next primary strategy. Bridge occupancy supplies repeated per-step
gradients while canonical acquisition supplies mostly sparse milestone credit, so even a nominal 75% canonical mix
can overwrite acquisition rapidly. The lower reward weight alone did not prevent this.

## Most important insights

1. Always select with the fixed 1,024-episode canonical evaluator. Reset-mixture occupancy metrics are not evidence
   of full-task competence.
2. Acquisition, lift, upright clearance, radial transport, and release are each individually learnable with the
   same vision/proprioception boundary. The failure is interference and long-horizon credit, not image resolution or
   basic observability.
3. The scratch actor has no explicit progress flag. It must infer task phase from RGB and proprioception. That is
   desirable for deployment, but training distributions that revisit visually/proprioceptively similar states with
   conflicting actions can destabilize the shared actor.
4. Large action means and near-continuous saturation of elbow/wrist-flex outputs are a warning sign. The best
   acquisition actor already saturates elbow and wrist flex for much of the rollout. Downstream updates therefore
   alter the same outputs acquisition critically depends on.
5. Static policy surgery is not a substitute for closed-loop learning. Bridge improvements that look strong can
   completely fail from home.
6. The simplest remaining avenue is to preserve acquisition with explicit on-policy rehearsal or a small behavior
   anchor while optimizing one downstream physical objective at a time. This is simpler and better motivated than
   adding more heads, gates, or scripted phases.

## Recommended restart plan

Keep the retained spatial-softmax actor and ordinary PPO. Do not resume the fixed-mixture upright-lift sweep.

1. Reproduce the retained acquisition checkpoint from random initialization using the known canonical-close ->
   close-lift -> canonical-lift-progress sequence. Exact-audit near iterations 400--500 and at every stage boundary.
   This verifies the cleaned repository before new research.
2. Add one small behavior-anchor loss on canonical on-policy observations while PPO optimizes connected post-lift
   rows. The reference action is the frozen acquisition checkpoint, used only during training. Ablate anchor weights
   `{0.1, 1, 10}` and audit every 10--25 updates. This is the leading experiment because it directly addresses the
   measured forgetting mechanism without deployment-time logic.
3. In parallel, test canonical rehearsal rollouts collected under the current actor, with the acquisition actor used
   only to label early acquisition states. Prefer a single DAgger-style loss plus PPO; do not combine this with extra
   residual heads or phase gates.
4. Optimize the post-lift objective in two clean stages: first true clearance plus upright alignment, then radial
   transport/insertion. Require at least 90% canonical grasp/lift before a checkpoint is allowed into the next stage.
5. Once one checkpoint produces held insertion from canonical starts, enable the already proven simple release
   credit and train release jointly with continued acquisition rehearsal.
6. Accept only after at least 90% success over 1,024 exact canonical episodes in multiple seeds. Then record frontal
   and wrist-camera videos and perform a from-zero reproduction.

Use four GPUs as four independent single-GPU experiments, never distributed training. Start at 4,096 environments;
fall back to 2,048 or 1,024 only on a measured out-of-memory failure.

## Commands at the pause point

Audit the retained scratch checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run so101-vial play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101-Camera-Scratch-Discovery-HorizonReleaseDirect \
  --agent rsl_rl_vision_tuned_long_horizon_frozen_finetune_scratch_cfg_entry_point \
  --num_envs 1024 \
  --checkpoint checkpoints/candidates/vision_transport_alignment_m1850_std003.pt \
  --deterministic \
  --external_callback so101_vial_place.evaluation.install_episode_counter \
  --device cuda:0 --visualizer none presets=newton_mjwarp
```

Train a fresh state policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run so101-vial train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 \
  --max_iterations 400 --seed 42 --run_name state_horizon \
  --device cuda:0 --visualizer none presets=newton_mjwarp
```

Evaluate state or distilled policies with the same 1,024-episode callback and the matching task/agent entry point.
When resuming distillation, use the accepted state checkpoint as the teacher, the geometry-spatial distillation
configuration, wrist RGB plus proprioception for the student, and privileged state only for the training teacher.

Run repository checks before restarting experiments:

```bash
uv run pytest -q
uv run ruff check src tests
```

Newton allocator startup exits 134/139 are transient on this machine. Retry the exact command after confirming with
`nvidia-smi` that no orphaned process still occupies that GPU.
