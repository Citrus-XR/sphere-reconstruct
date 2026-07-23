"""imu.parse_imu_payload の単体テスト."""

from __future__ import annotations

import struct

from sphere_reconstruct.insta360.imu import (
    IMU_ENTRY_SIZE,
    parse_imu_payload,
    sample_rate_hz,
)


def _make_entry(ts_us: int, gx: float, gy: float, gz: float, ax: float, ay: float, az: float) -> bytes:
    return struct.pack("<Q6d", ts_us, gx, gy, gz, ax, ay, az)


def test_parse_single_entry():
    payload = _make_entry(1_000_000, 0.1, 0.2, 0.3, 9.81, 0.0, 0.0)
    assert len(payload) == IMU_ENTRY_SIZE
    samples = parse_imu_payload(payload)
    assert len(samples) == 1
    s = samples[0]
    assert s.timestamp_us == 1_000_000
    assert s.gyro_xyz == (0.1, 0.2, 0.3)
    assert s.accel_xyz == (9.81, 0.0, 0.0)


def test_parse_multi_entry_and_rate():
    entries = [
        _make_entry(0, 0, 0, 0, 0, 0, 0),
        _make_entry(5_000, 0, 0, 0, 0, 0, 0),      # 5ms -> 200Hz
        _make_entry(10_000, 0, 0, 0, 0, 0, 0),
    ]
    samples = parse_imu_payload(b"".join(entries))
    assert len(samples) == 3
    rate = sample_rate_hz(samples)
    assert rate is not None
    assert 199.0 < rate < 201.0
