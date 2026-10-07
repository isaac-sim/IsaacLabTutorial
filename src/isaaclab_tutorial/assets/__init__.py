"""Paths for tracked tutorial assets."""

from pathlib import Path

ASSET_ROOT = Path(__file__).parent
VIAL_USD = ASSET_ROOT / "workshop" / "vial.usda"
RACK_USD = ASSET_ROOT / "workshop" / "rack.usda"
DESK_USD = ASSET_ROOT / "workshop" / "desk.usda"
RESET_DATASET = ASSET_ROOT / "reset_poses.pt"


def validate_assets() -> list[str]:
    """Return missing local asset dependencies without requiring a USD runtime."""
    return [str(path) for path in (VIAL_USD, RACK_USD, DESK_USD, RESET_DATASET) if not path.is_file()]
