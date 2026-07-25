"""imu.parse_imu_payload の単体テスト (accel 先, gyro 後; telemetry-parser 準拠)."""

from __future__ import annotations

import struct

from sphere_reconstruct.insta360.imu import (
    IMU_ENTRY_SIZE,
    parse_imu_payload,
    sample_rate_hz,
)


def _make_entry(ts_us: int, ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> bytes:
    # ディスク上のバイト順: timestamp, accel(x,y,z), gyro(x,y,z).
    return struct.pack("<Q6d", ts_us, ax, ay, az, gx, gy, gz)


def test_parse_single_entry():
    payload = _make_entry(1_000_000, 9.81, 0.0, 0.0, 0.1, 0.2, 0.3)
    assert len(payload) == IMU_ENTRY_SIZE
    (s,) = parse_imu_payload(payload)
    assert s.timestamp_us == 1_000_000
    assert s.accel_xyz == (9.81, 0.0, 0.0)
    assert s.gyro_xyz == (0.1, 0.2, 0.3)


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
