# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional real-browser boundary: failed trials stay inspectable and budget controls use real data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

browser_api = pytest.importorskip("playwright.sync_api", reason="Install playwright to run the optional browser check")


def test_invalid_evaluation_remains_visible_while_budget_and_curves_work(tmp_path, training_report):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("A locally installed Chrome/Chromium is required for this optional check")
    root = Path(__file__).resolve().parents[1]
    report = training_report
    # This is the numerical-failure schema produced by evaluate.py, not a truncated successful score.
    report["runs"][0]["evaluation"].update(valid=False, metrics={"invalid_reason": "nan_state"})
    fixture = tmp_path / "report.json"
    fixture.write_text(json.dumps(report))
    output = tmp_path / "rendered"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/locomotion/training_workshop.py"),
            "--report",
            str(fixture),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    errors = []
    with browser_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((output / "index.html").as_uri())
            assert not errors
            assert "nan_state" in page.locator("#replays").inner_text()
            assert page.locator("#chart path").count() == 2
            page.select_option("#envs", "256")
            assert page.locator("#experience").inner_text() == "307,200"
            assert page.locator("#updates").inner_text() == "1,000"
            path = page.locator("#chart path").first.get_attribute("d")
            page.select_option("#metric", "Metrics/base_velocity/error_vel_xy")
            assert page.locator("#chart path").first.get_attribute("d") != path
            assert not errors
        finally:
            browser.close()
