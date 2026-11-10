# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture actual single-run training curves, realized budget and fixed-protocol replay information."""

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.report_directory / "report.json").read_text())
    if len(report["runs"]) != 1 or not report["runs"][0].get("evaluation"):
        parser.error("Choose a single fresh-PPO investigator report with its actual checkpoint evaluation attached.")
    run = report["runs"][0]
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        parser.error("Install Chrome/Chromium for this optional capture tool.")
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 2100}, device_scale_factor=2)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((args.report_directory.resolve() / "index.html").as_uri())
            page.select_option("#metric", "Train/mean_episode_length")
            page.select_option("#axis", "transitions")
            page.locator("#cursor").fill("100")
            page.locator("#cursor").dispatch_event("input")
            assert page.locator("#chart path").count() == 1
            assert f"{run['logged_budget']['collected_transitions']:,}" in page.locator("#run-table").inner_text()
            assert f"{run['checkpoint']['optimizer_steps']:,}" in page.locator("#run-table").inner_text()
            evaluation_text = page.locator("#replays").inner_text()
            assert f"{run['evaluation']['metrics']['alive_time_s']:.2f} s" in evaluation_text
            if run.get("replay_command"):
                assert run["replay_command"] in evaluation_text
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            first = page.locator("section[aria-labelledby='curves-heading']").bounding_box()
            last = page.locator("section[aria-labelledby='replay-heading']").bounding_box()
            clip = {
                "x": first["x"],
                "y": first["y"],
                "width": first["width"],
                "height": last["y"] + last["height"] - first["y"],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.output), type="jpeg", quality=88, clip=clip)
            assert not errors, errors
            proof = {
                "scope": (
                    "Actual Chrome capture of adjacent curves and replay sections; "
                    "no UI was hidden or modified for capture."
                ),
                "browser_version": browser.version,
                "playwright_version": version("playwright"),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256((args.report_directory / "report.json").read_bytes()).hexdigest(),
                "html_sha256": hashlib.sha256((args.report_directory / "index.html").read_bytes()).hexdigest(),
                "checkpoint_sha256": run["checkpoint"]["sha256"],
                "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "capture_css_rectangle": clip,
                "device_scale_factor": 2,
                "recorded_budget": run["logged_budget"],
                "evaluation": run["evaluation"],
                "diagnostic": "Train/mean_episode_length",
                "axis": "Fresh transitions",
                "javascript_errors": errors,
            }
            args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")
            print("Captured actual single-run curves, budget and evaluation/replay information in Chrome.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
