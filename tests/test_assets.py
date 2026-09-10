from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tutorial.assets import (
    MAT_USD,
    RACK_USD,
    VIAL_RACK_ASSET_DIR,
    VIAL_USD,
    validate_assets,
)


def test_all_declared_assets_exist():
    assert validate_assets() == []


def test_vial_rack_assets_use_isaaclab_asset_root():
    expected_asset_dir = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Vial_Rack"
    expected_assets = {
        f"{expected_asset_dir}/mat.usda",
        f"{expected_asset_dir}/rack.usda",
        f"{expected_asset_dir}/vial.usda",
    }

    assert expected_asset_dir == VIAL_RACK_ASSET_DIR
    assert {MAT_USD, RACK_USD, VIAL_USD} == expected_assets
