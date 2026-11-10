# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional browser boundary: incomplete or invalid evidence cannot become successful response bars."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

browser_api = pytest.importorskip("playwright.sync_api", reason="Install playwright for the optional browser check")


def test_portable_conditions_keep_missing_and_invalid_evidence_visible(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("A locally installed Chrome/Chromium is required")
    root = Path(__file__).resolve().parents[1]
    reports = [
        json.loads((root / f"docs/locomotion/evidence/{name}.json").read_text())
        for name in ("ppo-starts-matrix", "ppo-delay40-matrix")
    ]
    partial = copy.deepcopy(reports[1])
    partial.update(complete=False, trials=[partial["trials"][0]])
    partial["trials"][0].update(valid=False, metrics={"invalid_reason": "nan_state"}, qualification={"passed": False})
    reports.append(partial)
    command = [sys.executable, str(root / "scripts/locomotion/qualification_workshop.py")]
    for index, report in enumerate(reports):
        for trial in report["trials"]:
            trial["receipt_path"] = str(tmp_path / "missing-recording.json")
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report))
        command += ["--report", str(path)]
    output = tmp_path / "rendered"
    subprocess.run([*command, "--output", str(output)], cwd=root, check=True)
    errors = []
    with browser_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((output / "index.html").as_uri())
            assert page.locator("#grid-body button.qualified").count() == 30
            page.select_option("#condition", "1")
            assert page.locator("#grid-body button.fell").count() == 30
            page.get_by_role("button", name="Training seed 42, evaluation seed 0: Fell", exact=True).click()
            assert "1.800 s" in page.locator("#trial-status").inner_text()
            assert "unavailable" in page.locator("#segments").inner_text()
            assert page.locator("#gains rect").count() == 1, "Only the acceptance band is available."
            assert page.locator("#ghost-command").is_hidden()
            page.select_option("#condition", "2")
            assert "INCOMPLETE" in page.locator("#summary").inner_text()
            assert "not run" in page.locator("#trial-status").inner_text()
            page.get_by_role("button", name="Training seed 0, evaluation seed 0: Invalid trial", exact=True).click()
            assert "nan_state" in page.locator("#trial-status").inner_text()
            assert page.locator("#gains rect").count() == 1
            page.set_viewport_size({"width": 420, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert not errors
        finally:
            browser.close()
