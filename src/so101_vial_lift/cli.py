"""External-project bridge to Isaac Lab's official unified entrypoints."""

import sys

import warp as wp


def main() -> int:
    """Dispatch the four supported workflows without relying on monorepo script paths."""
    wp.config.enable_backward = False
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
        any("so101_vial_lift:" in arg for arg in args)
        and command in {"zero_agent", "random_agent"}
        and "--visualizer" not in args
        and "--viz" not in args
    ):
        args = ["--visualizer", "newton", *args]
    return commands[command](args)
