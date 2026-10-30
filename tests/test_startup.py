"""Task configurations must remain safe to load before the selected runtime starts."""

import subprocess
import sys
import textwrap


def test_task_config_does_not_load_usd_before_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                from isaaclab.cli import load_external_tasks
                from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

                load_external_tasks()
                load_cfg_from_registry("IsaacTutorial-Place-Vial-SO101", "env_cfg_entry_point")
                assert "pxr" not in sys.modules, "Task configuration loaded OpenUSD before runtime startup"
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
