# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A repeated or unmatched signal cannot justify a measured sample-age claim."""

from isaaclab_tutorial.locomotion.playground.actuator_trace import unique_sample_ages


def test_sample_age_requires_one_past_match_and_uses_the_control_clock():
    raw = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    observed = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [9.0, 0.0]]
    # The first two samples identify a source. Two identical past values make the third ambiguous;
    # the final value is absent from recorded history. Neither may be assigned the configured maximum.
    assert unique_sample_ages(raw, observed, 0.02, 2) == [0.0, 0.02, None, None]
