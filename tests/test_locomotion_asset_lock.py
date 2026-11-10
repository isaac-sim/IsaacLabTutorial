# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A familiar asset filename must not admit changed robot geometry or mechanics."""

import pytest

from isaaclab_tutorial.locomotion.assets import verify_robot_asset_file


def test_changed_robot_bytes_are_rejected_even_when_the_filename_matches(tmp_path):
    asset = tmp_path / "microduck_walk.usd"
    asset.write_bytes(b"Different geometry and mass, despite the familiar filename.")
    with pytest.raises(ValueError, match="SHA256"):
        verify_robot_asset_file(asset)
    unknown = tmp_path / "another_robot.usd"
    unknown.write_bytes(b"Unreviewed model")
    with pytest.raises(ValueError, match="locked Microduck"):
        verify_robot_asset_file(unknown)
