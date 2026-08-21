import re

from so101_vial_lift.assets import ASSET_ROOT, SO101_USD, SO101_VARIANTS, validate_assets


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


def test_text_usd_dependencies_resolve():
    missing = []
    for usd in ASSET_ROOT.rglob("*.usda"):
        text = usd.read_text(errors="ignore")
        for reference in re.findall(r"@([^@\n]+)@", text):
            if "://" not in reference and not (usd.parent / reference).resolve().exists():
                missing.append(f"{usd}: {reference}")
    assert missing == []


def test_only_two_fingertip_collision_proxies_are_enabled():
    physics = (SO101_USD.parent / "payloads/Physics/physics.usda").read_text()
    instances = (SO101_USD.parent / "payloads/instances.usda").read_text()

    assert 'def Cube "task_fixed_jaw_collision"' in physics
    assert 'def Cube "task_moving_jaw_collision"' in physics
    assert physics.count("bool physics:collisionEnabled = true") == 2
    assert physics.count("rel material:binding:physics = </so101_new_calib/task_fingertip_material>") == 2
    assert 'def Material "task_fingertip_material"' in physics
    assert "bool physics:collisionEnabled = 1" not in instances
