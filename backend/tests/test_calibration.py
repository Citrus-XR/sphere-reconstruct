"""INSV の versioned ASCII calibration を検証する。"""

from __future__ import annotations

import pytest

from sphere_reconstruct.domain.camera_system import OmniDistortionModel
from sphere_reconstruct.insta360 import calibration

OFFSET_V3 = (
    "2_"
    "2.000000_4278.300_4277.330_2694.630_2681.840_0.615_0.016_89.937_"
    "0.000000_0.000000_0.000000_0.18366432_2.07332635_-3.27984834_"
    "-0.00005305_0.00065176_10752_5376_113_"
    "2.000000_4296.810_4298.540_8064.920_2686.410_-0.718_0.211_89.840_"
    "-0.000048_0.000131_-0.032273_0.18302010_2.05338216_-3.26668859_"
    "0.00187136_0.00038193_10752_5376_113_197632"
)

OFFSET_V6 = (
    "2_"
    "2.000000_4274.220_4273.920_2693.700_2683.680_0.653_0.053_89.946_"
    "0.000000_0.000000_0.000000_0.22852755_1.56955242_-1.15421379_-3.01855540_"
    "0.00000000_0.00020713_-0.00208586_0.00644459_0.01365752_-0.00300253_"
    "-0.00044836_-0.00027934_-0.01400746_10752_5376_113_"
    "2.000000_4291.970_4293.160_8058.960_2677.980_-1.002_0.477_89.834_"
    "0.000014_-0.000111_-0.032081_0.25025144_1.21299148_0.64438474_-6.10575676_"
    "0.00000000_-0.00154459_0.00110053_-0.01006391_-0.01174382_0.00528728_"
    "0.00366531_0.02918606_0.03056746_10752_5376_113_394240"
)


def _candidate(text: str, offset: int = 0) -> calibration.OffsetCandidate:
    return calibration.OffsetCandidate(
        inst_offset=offset,
        text=text,
        values=tuple(float(item) for item in text.split("_")),
    )


def test_highest_supported_calibration_version_is_selected():
    data = b"\0" + OFFSET_V6.encode() + b"\0" + OFFSET_V3.encode() + b"\0"
    candidates = calibration.find_ascii_calibrations(data)

    chosen = calibration.pick_calibration(candidates)

    assert chosen is not None
    assert chosen.version == 6
    assert len(chosen.values) == 56


def test_duplicate_highest_version_uses_latest_footer_candidate():
    first = _candidate(OFFSET_V6, 100)
    second = _candidate(OFFSET_V6, 200)

    assert calibration.pick_calibration([second, first]) is second


def test_v3_uses_five_parameter_radtan_model():
    result = calibration.parse_ascii_calibration(_candidate(OFFSET_V3))

    assert result.version == 3
    assert result.source == calibration.CalibSource.OFFSET
    assert result.raw["calibration_id"] == 0x30400
    assert result.lenses[0].distortion_model == OmniDistortionModel.RADTAN
    assert result.lenses[0].distortion_parameters == (
        0.18366432,
        2.07332635,
        -3.27984834,
        -0.00005305,
        0.00065176,
    )
    assert result.lenses[1].tz == -0.032273


def test_v6_uses_thirteen_parameter_radtan_pro_model():
    result = calibration.parse_ascii_calibration(_candidate(OFFSET_V6))

    assert result.version == 6
    assert result.raw["calibration_id"] == 0x60400
    assert result.lenses[0].distortion_model == OmniDistortionModel.RADTAN_PRO
    assert result.lenses[0].distortion_parameters == (
        0.22852755,
        1.56955242,
        -1.15421379,
        -3.01855540,
        0.0,
        0.00020713,
        -0.00208586,
        0.00644459,
        0.01365752,
        -0.00300253,
        -0.00044836,
        -0.00027934,
        -0.01400746,
    )
    assert result.lenses[1].fx == 4291.97
    assert result.lenses[1].tz == -0.032081


def test_unsupported_version_is_not_silently_interpreted():
    values = (2.0, *(1.0 for _ in range(32)), float(0x20400))
    candidate = calibration.OffsetCandidate(0, "_".join(str(value) for value in values), values)

    assert calibration.pick_calibration([candidate]) is None
    with pytest.raises(ValueError, match="未対応"):
        calibration.parse_ascii_calibration(candidate)
