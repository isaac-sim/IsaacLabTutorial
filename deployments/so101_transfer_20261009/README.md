# Qualified SO-101 Transfer vision bundle

This compact deployment snapshot is checked in so the robot computer can retrieve the
selected policy with the source branch. Training checkpoints and run logs remain under ignored
`outputs/`; this archive contains the CPU actor, control contract, standalone deployment script,
qualification manifest and CPU parity result.

From the repository root after pulling `feat/so101-consolidated-sim2real`:

```bash
(cd deployments/so101_transfer_20261009 && sha256sum -c deployment_bundle.tar.gz.sha256)
mkdir -p outputs/robot_transfer_20261009
tar -xzf deployments/so101_transfer_20261009/deployment_bundle.tar.gz -C outputs/robot_transfer_20261009
```

Use `outputs/robot_transfer_20261009/so101_transfer_vision/leapp/leapp.yaml` as the
`--bundle` in the existing robot test command. Retain the checked joint map, follower port,
camera and home pose. See the extracted `DEPLOYMENT.txt` for the sensor-only command;
no robot connection or motion is performed by extracting this archive.

The contract uses shoulder-lift scale **0.040**, other arm scales **0.033**, and jaw scale
**0.020**, at 30 Hz inference / 120 Hz measured-relative feedback. The frozen actor passed
four independent full-collider audits: 96.39%, 95.80%, 96.88%, 96.48% (3,948/4,096 total).
CPU parity on Torch 2.10.0 / LEAPP 0.7.1 matched exactly across 32 inputs.

Real placement success remains unmeasured. Eleven qualification episodes exceeded 20 N
rack contact, with a 36.71 N peak. See the
[full audit](../../docs/sim2real/DOMAIN_RANDOMIZATION.md#qualified-transfer-vision-policy--2026-10-09)
for randomization coverage and limitations. Paths under `/home/` or `outputs/` in the archived
manifest identify training provenance; inference uses the files beside `leapp.yaml`.
