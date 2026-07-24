"""calibration の降級鎖 (choose) と ASCII offset_v3 の追加テスト."""

from __future__ import annotations

from sphere_reconstruct.insta360 import calibration as calib


def _valid_dual() -> calib.DualLensCalibration:
    lens = calib.MeiLensCalibration(xi=2.0, fx=4278.3, fy=4277.3, cx=2694.6, cy=2681.8)
    return calib.DualLensCalibration(source=calib.CalibSource.OFFSET_V3, lenses=[lens, lens])


def test_choose_skips_none_and_invalid():
    invalid = calib.DualLensCalibration(source=calib.CalibSource.PB, lenses=[])
    valid = _valid_dual()
    assert calib.choose(None, invalid, valid) is valid


def test_choose_returns_none_when_all_invalid():
    invalid = calib.DualLensCalibration(source=calib.CalibSource.PB, lenses=[])
    assert calib.choose(None, invalid) is None


def test_choose_prefers_first_valid():
    a = _valid_dual()
    b = _valid_dual()
    assert calib.choose(a, b) is a


def test_is_valid_requires_two_lenses_with_positive_focal():
    one = calib.DualLensCalibration(
        source=calib.CalibSource.OFFSET_V3,
        lenses=[calib.MeiLensCalibration(xi=2.0, fx=100, fy=100, cx=50, cy=50)],
    )
    assert not one.is_valid()
    zero_focal = calib.DualLensCalibration(
        source=calib.CalibSource.OFFSET_V3,
        lenses=[
            calib.MeiLensCalibration(xi=2.0, fx=0, fy=100, cx=50, cy=50),
            calib.MeiLensCalibration(xi=2.0, fx=100, fy=100, cx=50, cy=50),
        ],
    )
    assert not zero_focal.is_valid()
