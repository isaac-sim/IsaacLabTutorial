"""Paths and variant selections for tracked tutorial assets."""

from pathlib import Path

ASSET_ROOT = Path(__file__).parent / "assets"
SO101_USD = ASSET_ROOT / "so101" / "so101_new_calib.usda"
VIAL_USD = ASSET_ROOT / "workshop" / "vial.usda"
RACK_USD = ASSET_ROOT / "workshop" / "rack.usda"
MAT_USD = ASSET_ROOT / "workshop" / "mat.usda"
RESET_DATASET = ASSET_ROOT / "reset_poses.pt"

SO101_VARIANTS = {"Robot": "robot", "Sensor": "sensors", "Physics": "physics"}


def validate_assets() -> list[str]:
    """Return missing local asset dependencies without requiring a USD runtime."""
    return [str(path) for path in (SO101_USD, VIAL_USD, RACK_USD, MAT_USD) if not path.is_file()]
