"""External-project bridge to Isaac Lab's official unified entrypoints."""

import sys

import warp as wp


def main() -> int:
    """Dispatch task workflows without relying on monorepo script paths."""
    wp.config.enable_backward = False
    if len(sys.argv) >= 2 and sys.argv[1] in {
        "capture_scene",
        "capture_wrist",
        "benchmark",
        "generate_resets",
        "inspect_robot",
        "view_resets",
        "promote_distillation",
        "set_exploration_std",
    }:
        from .benchmark import benchmark_main
        from .checkpoint_tools import exploration_std_main, promote_main
        from .reset.generator import generate_main, view_main
        from .robot_diagnostics import inspect_robot_main
        from .scene_preview import capture_main
        from .wrist_preview import capture_wrist_main

        command = sys.argv[1]
        return {
            "capture_scene": capture_main,
            "capture_wrist": capture_wrist_main,
            "benchmark": benchmark_main,
            "generate_resets": generate_main,
            "inspect_robot": inspect_robot_main,
            "view_resets": view_main,
            "promote_distillation": promote_main,
            "set_exploration_std": exploration_std_main,
        }[command](sys.argv[2:])

    from isaaclab_rl import entrypoints

    commands = {
        "train": entrypoints.run_train_cli,
        "play": entrypoints.run_play_cli,
        "zero_agent": entrypoints.run_zero_agent_cli,
        "random_agent": entrypoints.run_random_agent_cli,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        available = ", ".join(commands)
        print(f"usage: isaaclab <command> [args]\ncommands: {available}", file=sys.stderr)
        return 2
    command = sys.argv[1]
    args = sys.argv[2:]
    # Simple-agent entrypoints otherwise default to Kit, which is not a
    # dependency of this Newton-native external project. Training and play
    # remain on Isaac Lab's stable Torch manager frontend; MJWarp is the
    # configured physics backend, not the experimental Warp env frontend.
    if (
        any("IsaacTutorial-Place-Vial-SO101" in arg for arg in args)
        and command in {"zero_agent", "random_agent"}
        and "--visualizer" not in args
        and "--viz" not in args
    ):
        args = ["--visualizer", "newton_gl", *args]
    return commands[command](args)
