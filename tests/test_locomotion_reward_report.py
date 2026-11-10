# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Published plot conclusions must follow the selected evidence, including a different failed policy."""

import json
import runpy
from pathlib import Path


def test_reward_plot_text_follows_changed_weights_failures_and_available_horizons():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/locomotion/evidence/reward-comparison.json").read_text())
    figure_text = runpy.run_path(str(root / "scripts/locomotion/plot_reward_comparison.py"))["figure_text"]
    _, caption = figure_text(report)
    assert "baseline earns more under both objectives" in caption.lower()
    report["changed_setting"]["candidate_weight"] = -0.5
    baseline = report["policies"]["baseline"]
    baseline["metrics"].update(alive_time_s=2, fell=True, coverage=0.1)
    baseline["qualification"]["passed"] = False
    report["policies"]["candidate"]["qualification"]["passed"] = True
    title, caption = figure_text(report)
    assert "-0.5" in title and "baseline: 2.00 s (fell)" in caption
    assert "candidate: qualified" in caption
    assert "not a matched-duration ranking" in caption
    assert "baseline earns more" not in caption.lower()
