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

The native panel reference images use [one portable physical snapshot](evidence/playground-snapshot.json), extracted from an actual playground recording. Recreate them with `capture_playground.py --receipt docs/locomotion/evidence/playground-snapshot.json --mode observations --output outputs/panel/observations.jpg` (or `--mode joints`) in the pinned locomotion environment. The tool draws the live workshop's actual widgets over the recorded body poses using public Newton/ImGui APIs; it creates no solver or physics steps. Captures label their recorded-state scope and retain camera, asset, source, input and image hashes. Renderer timing text can vary between machines.

## Keep the learner in control

Open each lesson with an action or prediction, show an observable result, then reveal the explanation. Each slider must state whether it changes the next command, merely rescales a recorded score, or launches a fresh experiment. GUI callbacks request changes at a control boundary; they must not advance physics independently.

Ghosts replay independently simulated closed-loop trials with matched actual starts and command schedules. Preserve world-space drift, failure time and incomplete coverage. A ghost is a labeled reference, not a guaranteed alternate future or a hardware measurement.

Rough terrain and fall recovery are later tutorial work. Hardware stance/timing/walking evidence remains a measured gate; keep those entries open until the calibrated robot trials exist.
