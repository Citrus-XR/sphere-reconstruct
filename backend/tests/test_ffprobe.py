"""frame timing と multi-sensor 同期契約を検証する。"""

import pytest

from sphere_reconstruct.imaging.ffprobe import FrameTiming, validate_synchronized_timings


def _timings(*values: float) -> list[FrameTiming]:
    return [FrameTiming(index=index, pts_sec=value, duration_sec=None) for index, value in enumerate(values)]


def test_synchronized_timings_report_maximum_skew():
    maximum = validate_synchronized_timings(
        _timings(0.0, 0.04, 0.08),
        _timings(0.0001, 0.0402, 0.0801),
        maximum_skew_sec=0.00025,
    )

    assert maximum == pytest.approx(0.0002)


def test_synchronized_timings_reject_missing_or_shifted_frames():
    with pytest.raises(ValueError, match="frame 数"):
        validate_synchronized_timings(
            _timings(0.0, 0.04),
            _timings(0.0),
            maximum_skew_sec=0.001,
        )
    with pytest.raises(ValueError, match="PTS skew"):
        validate_synchronized_timings(
            _timings(0.0, 0.04),
            _timings(0.0, 0.05),
            maximum_skew_sec=0.001,
        )
