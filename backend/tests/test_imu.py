"""imu.parse_imu_payload の単体テスト (accel 先, gyro 後; telemetry-parser 準拠)."""

from __future__ import annotations

import struct

from sphere_reconstruct.insta360.imu import (
    IMU_ENTRY_SIZE,
    ImuRecording,
    ImuSample,
    parse_imu_payload,
    parse_raw_imu_payload,
    rolling_shutter_motion_by_frame,
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
        _make_entry(5_000, 0, 0, 0, 0, 0, 0),  # 5ms -> 200Hz
        _make_entry(10_000, 0, 0, 0, 0, 0, 0),
    ]
    samples = parse_imu_payload(b"".join(entries))
    assert len(samples) == 3
    rate = sample_rate_hz(samples)
    assert rate is not None
    assert 199.0 < rate < 201.0


def test_parse_raw_entry():
    payload = struct.pack(
        "<Q6H",
        1_000,
        32768 + 10,
        32768 - 20,
        32768 + 30,
        32768 + 1,
        32768 + 2,
        32768 + 3,
    )
    (sample,) = parse_raw_imu_payload(payload)
    assert sample.timestamp_us == 1_000
    assert sample.accel_xyz == (10.0, -20.0, 30.0)
    assert sample.gyro_xyz == (1.0, 2.0, 3.0)


def test_rolling_shutter_motion_uses_recorded_range_and_readout():
    recording = ImuRecording(
        samples=[
            ImuSample(0, (0.0, 0.0, 0.0), (3276.8, 0.0, 0.0)),
            ImuSample(1_000_000, (0.0, 0.0, 0.0), (6553.6, 0.0, 0.0)),
        ],
        timestamps_sec=[0.0, 1.0],
        orientation="yzX",
        camera_type="Generic dual fisheye",
        is_raw=True,
        gyro_range_dps=2000,
        rolling_shutter_time_ms=20.0,
    )

    motion = rolling_shutter_motion_by_frame(recording, [0, 30], fps=30.0)

    assert abs(motion[0] - 4.0) < 1e-9
    assert abs(motion[30] - 8.0) < 1e-9
