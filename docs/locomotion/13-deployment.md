# 13. Export without changing the policy

First check the learned actor across the native/export boundary:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest --export --output outputs/export-check
```

This exports JIT and ONNX models using RSL-RL's supported exporter and checks JIT actions against native actions on the actual evaluation observations. The receipt records the maximum absolute difference and float32 tolerances (`atol=1e-5`, `rtol=1e-4`). The current check covers JIT execution; ONNX execution needs its own runtime parity check before deployment.

Normalization is embedded in the exported actor. Supply the raw assembled 61-value actor observation exactly once; do not normalize it a second time. Preserve the verified 14-action order, reference offsets, scale, bias convention, timing, and inference/reset state.

A hardware adapter must first pass a **no-actuation dry run**: assemble calibrated sensor observations, run inference, log bounded targets and timestamps, check sensor age and missed deadlines, and define the actual robot watchdog/stop behavior. Only then progress through operator-controlled standing and low-speed trials using the real robot's approved limits.

Export parity establishes software consistency. Simulated robustness establishes behavior under those simulated conditions. Demonstrated transfer additionally requires calibrated hardware trials and their measured outcomes. That hardware adapter and evidence remain a subsequent checkpoint.

**Record:** what the export check proves, and one robot-side failure that identical exported actions would not prevent.
