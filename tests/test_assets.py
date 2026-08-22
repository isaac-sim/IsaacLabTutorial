import re
from hashlib import sha256

from so101_vial_place.assets import ASSET_ROOT, RACK_USD, SO101_USD, SO101_VARIANTS, VIAL_USD, validate_assets


def test_all_declared_assets_exist():
    assert validate_assets() == []


def test_so101_variants_and_camera_are_present():
    interface = SO101_USD.read_text()
    assert SO101_VARIANTS == {"Robot": "robot", "Sensor": "sensors", "Physics": "physics"}
    for name, selection in SO101_VARIANTS.items():
        assert f'variantSet "{name}"' in interface
        assert f'"{selection}" (' in interface
    sensor_payload = (SO101_USD.parent / "payloads/Sensor/sensors.usda").read_text()
    assert "so101-camera_mount/SO-101_camera_mount.usda" in sensor_payload
    camera_payload = SO101_USD.parent / "payloads/so101-camera_mount/payloads/base.usda"
    assert 'Camera "wowrobo_2MP_camera"' in camera_payload.read_text()


def test_identified_newton_actuator_layer_is_unmodified():
    physics = SO101_USD.parent / "payloads/Physics/physics.usda"
    # Hash of the layer in the user-supplied so101_new_calib_SysID.zip. This
    # protects both the identified drives and the authored mass properties.
    expected_digest = "859dfddc29fd5e4c0ec753f6842d87d26d3fcd381f5efa9722d68a4789a3ae27"

    assert sha256(physics.read_bytes()).hexdigest() == expected_digest


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


def test_authored_robot_collision_geometry_is_enabled_without_proxies():
    physics = (SO101_USD.parent / "payloads/Physics/physics.usda").read_text()
    instances = (SO101_USD.parent / "payloads/instances.usda").read_text()

    assert "task_fingertip_material" not in physics
    assert "task_fixed_jaw_collision" not in physics
    assert "task_moving_jaw_collision" not in physics
    # Link colliders are single convex hulls. Convex decomposition would run
    # CoACD over high-resolution visual meshes at every process startup.
    assert "convexDecomposition" not in instances
    assert instances.count('physics:approximation = "convexHull"') == 28
    assert instances.count("bool physics:collisionEnabled = 1") == 15
    assert "bool physics:collisionEnabled = 0" not in instances


def test_rack_uses_the_detailed_four_hole_mesh_for_visuals_and_collision():
    wrapper = RACK_USD.read_text()
    source = (RACK_USD.parent / "Vial_rack_simple.usda").read_text()

    assert "@./Vial_rack_simple.usda@</World>" in wrapper
    assert "double3 xformOp:translate = (-0.0298317129, -0.0298575352, 0)" in wrapper
    assert 'over "Mesh"' in wrapper
    assert 'uniform token physics:approximation = "sdf"' in wrapper
    assert 'def Xform "Collision" (active = false)' in wrapper
    for marker in ("top_01", "top_02", "top_03", "top_04"):
        assert f'def Xform "{marker}"' in source
