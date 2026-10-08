import re

from isaaclab_tutorial.assets import (
    ASSET_ROOT,
    RACK_USD,
    VIAL_USD,
    validate_assets,
)


def test_all_declared_assets_exist():
    assert validate_assets() == []


def test_vial_preserves_visual_mesh_and_measured_collision_dimensions():
    import pytest
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(VIAL_USD))
    body = UsdGeom.Cylinder(stage.GetPrimAtPath("/Vial/body_collider"))
    cap = UsdGeom.Cylinder(stage.GetPrimAtPath("/Vial/cap_collider"))
    bottom = UsdGeom.Sphere(stage.GetPrimAtPath("/Vial/bottom_collider"))
    assert 2 * body.GetRadiusAttr().Get() == pytest.approx(0.0289)
    assert 2 * cap.GetRadiusAttr().Get() == pytest.approx(0.0354)
    assert bottom.GetRadiusAttr().Get() == pytest.approx(body.GetRadiusAttr().Get())
    assert cap.GetHeightAttr().Get() == pytest.approx(0.01277)
    assert UsdGeom.Mesh(stage.GetPrimAtPath("/Vial/Mesh_002"))


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


def test_vial_legacy_mesh_does_not_contribute_implicit_mass():
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(VIAL_USD))
    legacy = stage.GetPrimAtPath("/Vial/collider")
    assert legacy.IsActive()  # Preserve the render mesh and its material binding.
    assert not legacy.HasAPI(UsdPhysics.CollisionAPI)
    colliders = {str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
    assert colliders == {"/Vial/body_collider", "/Vial/cap_collider", "/Vial/bottom_collider"}


def test_success_hole_centers_match_the_physical_rack_openings():
    import torch
    from pxr import Usd, UsdGeom

    from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS

    stage = Usd.Stage.Open(str(RACK_USD))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["guide"], False, True)

    def interval(name, axis):
        box = bounds.ComputeWorldBound(stage.GetPrimAtPath(f"/Rack/Collision/{name}")).ComputeAlignedRange()
        return box.GetMin()[axis], box.GetMax()[axis]

    left, center_x, right = (interval(name, 0) for name in ("TopRailLeft", "TopRailCenterX", "TopRailRight"))
    front, center_y, back = (interval(name, 1) for name in ("TopRailFront", "TopRailCenterY", "TopRailBack"))
    xs = ((left[1] + center_x[0]) / 2, (center_x[1] + right[0]) / 2)
    ys = ((front[1] + center_y[0]) / 2, (center_y[1] + back[0]) / 2)
    actual = torch.tensor([(xs[0], ys[0], 0), (xs[1], ys[0], 0), (xs[1], ys[1], 0), (xs[0], ys[1], 0)])
    torch.testing.assert_close(torch.tensor(RACK_HOLE_CENTERS), actual, atol=0.0005, rtol=0)
