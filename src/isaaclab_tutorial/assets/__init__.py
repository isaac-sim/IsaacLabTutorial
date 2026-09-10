"""Paths for tracked tutorial assets."""

from pathlib import Path

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_ROOT = Path(__file__).parent
VIAL_RACK_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Vial_Rack"
VIAL_USD = f"{VIAL_RACK_ASSET_DIR}/vial.usda"
RACK_USD = f"{VIAL_RACK_ASSET_DIR}/rack.usda"
MAT_USD = f"{VIAL_RACK_ASSET_DIR}/mat.usda"
RESET_DATASET = ASSET_ROOT / "reset_poses.pt"


def validate_assets() -> list[str]:
    """Return missing local asset dependencies without requiring a USD runtime."""
    return [str(RESET_DATASET)] if not RESET_DATASET.is_file() else []
