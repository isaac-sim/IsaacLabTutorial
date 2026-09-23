from so101_place_vial.tasks.place_vial.config.so101.env_cfg import SO101InspectionEnvCfg


def test_scene_has_only_the_assets_introduced_so_far():
    cfg = SO101InspectionEnvCfg()
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.sim.dt == 1.0 / 120.0
    assert cfg.decimation == 4
    assert hasattr(cfg.scene, "vial")
    assert cfg.events.vial_mass.params["mass_distribution_params"] == (0.02, 0.02)
