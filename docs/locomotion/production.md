# Production evidence for tutorial contributors

[The production inventory](production.json) tracks all requested visuals **V01–V45** and experiences **I01–I08**. It records actual output paths and SHA256 values, teaching labels, acceptance criteria, evidence and remaining gaps. It is an audit of the package, not a claim that every planned asset has shipped.

| Status | Meaning |
| --- | --- |
| `produced` | A diagram, plot or capture is present; review its acceptance criteria before publishing. |
| `partial` | A working component or related output exists; the stated gap remains. |
| `missing` | The requested output has not been produced. |
| `needs_hardware` | Calibrated robot measurements or real captures are required. |
| `delegated_followup` | Work is owned separately and has not been integrated. |

Activities distinguish `implemented` from `partial`. This records working controls and observable outcomes; it does not imply that every visual enhancement or hardware trial is complete.

Verify the inventory without starting a simulator:

```bash
uv run --project environments/locomotion python scripts/locomotion/check_production.py
```

The verifier rejects missing IDs, missing files, and changed bytes. When replacing an output, update its provenance, inspect its caption/alt text and teaching claim, then record the new checksum. A green inventory check proves consistency, not scientific validity or hardware transfer.

## Reproduce a visual

Start from its evidence file and the exact source/environment lock. A measured plot needs the actual recorded data, policy hash, training and evaluation conditions, reset/command protocol, and signal units. An analytic plot must identify its equation and illustrative inputs. A simulator image/video needs the actual robot asset, camera/renderer, time scale and run identity. Keep those classes of evidence distinct.

The committed [asset lock](evidence/asset-lock.json) freezes the nominal and series-play robot bytes. The [PPO policy contract](evidence/policy-contract.json) records the exact 61-input/14-action API1 bundle checked in the standard Rust runtime. The [evaluation protocol reference](reference/evaluation-protocols.md) defines starts, commands, response criteria and missing-data handling. Individual evidence receipts retain producer, data and output hashes for the measured figures.

Only small JPEG derivatives live in this repository. Full trajectory receipts, checkpoints and cloud source/asset archives remain collected run artifacts. Videos need a durable artifact URL and checksum before linking them in a lesson; a temporary local output path is not a published video.

The [prepared delivery manifest](evidence/media-delivery.json) binds the five native movies, their recording inputs, the learner workload dataset and an optional nominal/delay PPO policy bundle to exact filenames, sizes and hashes. All ten files have reviewed public URLs; the [download reference](reference/downloads.md) explains their uses and the optional policy bundle. The policy bundle keeps the actual model, run metadata and YAML parameters for each profile. It includes no robot assets or hardware qualification.

The [clean-checkout walkthrough](evidence/clean-walkthrough.json) records 13 executed core stages from an isolated source archive and separately installed pinned environment. It covers a fresh short PPO run, evaluation, puppet/observation controls, unchanged and changed reward rescoring, the investigator, frozen driving, a supported reversal, matched sensor-lag recordings and ONNX parity. It used existing dependency/asset caches on the same host. The fresh short policy survived 20 seconds but failed all moving-command responses; that result is separate from the earlier published short policy's fall. Native mouse-event checks and actual reward retraining retain their own evidence; no manual desktop or robot walkthrough is claimed here.

The native panel reference images use [one portable physical snapshot](evidence/playground-snapshot.json), extracted from an actual playground recording. Recreate them with `capture_playground.py --receipt docs/locomotion/evidence/playground-snapshot.json --mode observations --output outputs/panel/observations.jpg` (or `--mode joints`) in the pinned locomotion environment. The tool draws the live workshop's actual widgets over the recorded body poses using public Newton/ImGui APIs; it creates no solver or physics steps. Captures label their recorded-state scope and retain camera, asset, source, input and image hashes. Renderer timing text can vary between machines.

The supported actuator activity also has [portable native traces and a physical snapshot](evidence/actuator-data.json). `plot_actuator.py --data ...` rebuilds its reversal/clock figures without simulation; `capture_actuator.py --receipt ...` renders the same native widgets over the retained physical state with no replay physics. Its [validation](evidence/actuator-validation.json) keeps mouse-event checks, recorder noninterference, actual repeated resets and their tiny residual differences distinct.

The [robot reference plates](reference/robot-asset.md) use one recorded reset pose for front, side, three-quarter, joint/frame and collider views. `capture_reference.py` rebuilds their native geometry and annotations; `inspect_contacts.py` independently reads the actual task's enabled colliders. [Validation](evidence/reference-validation.json) confirms the kinematic/projection checks, unchanged pose and byte-identical repeated captures.

The [standard Duck signal map](reference/hardware-signal-map.md) traces the pinned SDK's actual encoder/IMU interfaces, unit and frame conversions, policy slots, raw-action feedback and ordered target writes. Its [editable Mermaid source](reference/hardware-signal-map.mmd) supplies the vector master; [the receipt](evidence/hardware-signal-map.json) records Mermaid CLI 12.2.0 commands, SVG/PNG masters and the small JPEG derivative. Every joint/ID/actor-slot row was checked against the SDK tables and the exported 61-input/14-action contract. This completes the source/interface schematic requested by V25; actual cable/calibration checks and hardware timing/footage remain measured gates.

## Keep the learner in control

Open each lesson with an action or prediction, show an observable result, then reveal the explanation. Each slider must state whether it changes the next command, merely rescales a recorded score, or launches a fresh experiment. GUI callbacks request changes at a control boundary; they must not advance physics independently.

Ghosts replay independently simulated closed-loop trials with matched actual starts and command schedules. Preserve world-space drift, failure time and incomplete coverage. A ghost is a labeled reference, not a guaranteed alternate future or a hardware measurement.

For checkpoint-progress media, use the collect/render commands in [lesson 8](08-training.md). The collector verifies actual model counters and event logs before labeling the recordings. The native movie uses identical fixed cameras; an early-ending trial holds its last pose with its own sample time. Keep the movie and the relative-path receipt archive together when publishing, and update their URLs and checksums together.

The compact first-attempt screenshot captures the actual investigator's adjacent curves and replay sections. Build a one-run report with its matching evaluation first, then use the optional browser capture tool:

```bash
uv run --project environments/locomotion --locked --with playwright python scripts/locomotion/capture_first_attempt.py --report-directory outputs/first-attempt-investigator --output outputs/first-attempt-investigator.jpg
```

It requires installed Chrome/Chromium and records the browser version, report/HTML/image hashes and captured rectangle. The replay command appears only when the report builder finds the exact local checkpoint; a report rebuilt elsewhere can correctly show an availability note instead. Do not change the HTML to manufacture an installed policy or a successful evaluation.

## Fixed-policy sensitivity sweeps on OSMO

[The sweep workflow](../../scripts/locomotion/osmo_effect_sweep.yaml) evaluates one mechanism per GPU using the [lesson 11 plan](11-effects.md). Choose the image digest, optional image credential and pool/platform as in lesson 12. Prepare an input archive from the selected run, retaining the exact model and its YAML/run metadata:

```bash
tar -czf policy-input.tar.gz -C YOUR_RUN model_999.pt run.json params/env.yaml params/agent.yaml
sha256sum policy-input.tar.gz YOUR_RUN/model_999.pt
osmo workflow submit scripts/locomotion/osmo_effect_sweep.yaml --pool YOUR_POOL --set workflow_name=duck-sensitivity-preflight mechanism=command_delay_ms eval_seed_count=3 case_name=command_delay_ms-5 --set-string image=YOUR_VALIDATED_IMAGE_DIGEST checkpoint_sha256=YOUR_MODEL_SHA256 policy_archive_sha256=YOUR_ARCHIVE_SHA256
```

Supply `image_credential=YOUR_CREDENTIAL_NAME` with `--set-string` when required by the registry. The workflow fetches the pinned published source; override its full `source_revision` only with another published producer commit. Upload the archive first, then the readiness marker, using the returned workflow name. The source-directory trailing slashes are deliberate: they copy directory contents and avoid the [upstream named-source Landlock issue](https://github.com/gokrazy/rsync/issues/66) in affected bundled clients.

Use a client that preserves those trailing slashes or includes the named-source upload fix. The tested `6.3.1.210ffea0` wrapper removed the slash and reported successful transfers with zero files; the isolated `6.4.0.1f5b2c86` client preserves it. Verify the task's actual archive-digest check before treating input delivery as successful. Changing a CLI binary does not change the frozen simulation source or policy.

```bash
mkdir -p policy-upload/archive policy-upload/ready
cp policy-input.tar.gz policy-upload/archive/
printf 'ready\n' > policy-upload/ready/input-ready
osmo workflow rsync upload WORKFLOW evaluate policy-upload/archive/:/osmo/run/workspace/microduck-input/
osmo workflow rsync upload WORKFLOW evaluate policy-upload/ready/:/osmo/run/workspace/microduck-input/
```

The task checks the archive digest before extracting it, then checks the selected model digest before any evaluation. Input delivery, execution, and the collection window are bounded. Inspect the preflight's retained reports and archive before production. For the full plan, submit `nominal` once and each of the six mechanism names with `eval_seed_count=10` and an empty `case_name`. Use the same source, image, model and GPU type/driver for every job.

Download `collection/` from the reported output directory, compare `trial-artifacts.tar.gz` with `sha256.json`, inspect `sweep.json`, then release the task promptly:

```bash
osmo workflow rsync download --no-progress WORKFLOW evaluate /osmo/data/output/collection/:outputs/WORKFLOW
osmo workflow exec WORKFLOW evaluate --entry "bash -lc 'touch /osmo/run/workspace/microduck-input/collected'"
```

Failed processes retain their diagnostic archive. A completed mechanism job proves its declared selection finished; only the complete 25-condition, ten-start collection supports the full sweep figure. No new policy is trained by this workflow.

Audit the seven extracted `sweep/` directories with `summarize_effect_sweeps.py --sweep PATH`, repeating `--sweep` once for nominal and each mechanism, then choose `--output YOUR_FIGURES`. It checks the actual archived model by SHA256 rather than basename, realized optimizer progress, source and asset bytes, GPU type/driver, all reports/receipts, and the real initial mechanics and command schedule before recomputing outcomes. Recognized invalid diagnostics remain attempted outcomes; a valid receipt from a failed process or an unexplained process failure is rejected. The [portable study](evidence/sensitivity-study.json) and [validation](evidence/sensitivity-validation.json) retain all 250 production attempts. Rebuilding with `--data` redraws those audited numbers without repeating the raw audit. SVG masters are written beside the JPEG derivatives for external delivery.

Rough terrain and fall recovery are later tutorial work. Hardware stance/timing/walking evidence remains a measured gate; keep those entries open until the calibrated robot trials exist.
