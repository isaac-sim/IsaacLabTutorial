#!/usr/bin/env bash
# Prepared local supervised trial. See docs/sim2real/FIRST_REAL_TRIAL.md.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
trial_root=outputs/consolidation_20261008/supervised_trial
follower_port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079843-if00
wrist_camera=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
common=(--joint-map "$trial_root/joint_map.json" --start-pose "$trial_root/start_pose.json" --port "$follower_port")
case "${1:-dry-run}" in
    inspect)
        exec uv run --script src/isaaclab_tutorial/utils/inspect_so101.py "${common[@]}" --watch
        ;;
    dry-run)
        exec uv run --script src/isaaclab_tutorial/utils/deploy.py "${common[@]}" \
            --bundle "$trial_root/leapp/leapp.yaml" --camera "$wrist_camera" --duration "${2:-20}"
        ;;
    run)
        exec uv run --script src/isaaclab_tutorial/utils/deploy.py "${common[@]}" \
            --bundle "$trial_root/leapp/leapp.yaml" --camera "$wrist_camera" --duration "${2:-5}" --execute
        ;;
    *)
        echo "Usage: $0 {inspect|dry-run [seconds]|run [seconds]}" >&2
        exit 2
        ;;
esac
