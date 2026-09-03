import re

from so101_vial_place.assets import (
    ASSET_ROOT,
    RACK_USD,
    VIAL_USD,
    validate_assets,
)


def test_all_declared_assets_exist():
    assert validate_assets() == []


def test_vial_preserves_visual_mesh_and_cap_shoulder_collision():
    wrapper = VIAL_USD.read_text()

    assert "@Vial_opaque.usda@</Vial>" in wrapper
    assert 'over "collider"' in wrapper
    assert "bool physics:collisionEnabled = false" in wrapper
    assert 'def Cylinder "body_collider"' in wrapper
    assert "double radius = 0.015670387" in wrapper
    assert 'def Cylinder "cap_collider"' in wrapper
    assert "double radius = 0.016947908" in wrapper


def test_text_usd_dependencies_resolve():
    missing = []
    for usd in ASSET_ROOT.rglob("*.usda"):
        text = usd.read_text(errors="ignore")
        for reference in re.findall(r"@([^@\n]+)@", text):
            if "://" not in reference and not (usd.parent / reference).resolve().exists():
                missing.append(f"{usd}: {reference}")
    assert missing == []


def test_rack_uses_detailed_visuals_and_a_primitive_four_hole_collider():
    wrapper = RACK_USD.read_text()
    source = (RACK_USD.parent / "Vial_rack_simple.usda").read_text()

    assert "@./Vial_rack_simple.usda@</World>" in wrapper
    assert "double3 xformOp:translate = (-0.0298317129, -0.0298575352, 0)" in wrapper
    assert 'over "Mesh"' in wrapper
    assert "bool physics:collisionEnabled = false" in wrapper
    assert 'def Xform "Collision"' in wrapper
    assert 'def Xform "Collision" (active = false)' not in wrapper
    assert 'physics:approximation = "sdf"' not in wrapper
    assert wrapper.count('def Cube "') == 11
    assert "double3 xformOp:translate = (0.0301682871, 0.0301424648, 0)" in wrapper
    assert "double3 xformOp:scale = (0.12, 0.12, 0.02)" in wrapper
    assert "double3 xformOp:scale = (0.012, 0.12, 0.012)" in wrapper
    assert "double3 xformOp:scale = (0.108, 0.012, 0.012)" in wrapper
    for marker in ("top_01", "top_02", "top_03", "top_04"):
        assert f'def Xform "{marker}"' in source
