"""IMU (Gyro record) parse と重力抽出の検証.

telemetry-parser 準拠のバイト順 (accel 先, gyro 後, 各 f64) と, 平均加速度からの
重力方向計算を合成データで確認する.
"""

from __future__ import annotations

import struct

from sphere_reconstruct.insta360.imu import (
    IMU_ENTRY_SIZE,
    extract_gravity,
    parse_imu_payload,
)


def _entry(ts: int, accel: tuple[float, float, float], gyro: tuple[float, float, float]) -> bytes:
    return struct.pack("<Q6d", ts, *accel, *gyro)


def test_parse_reads_accel_first_then_gyro() -> None:
    payload = _entry(1000, (0.0, 0.0, 1.0), (0.1, 0.2, 0.3))
    assert len(payload) == IMU_ENTRY_SIZE
    (s,) = parse_imu_payload(payload)
    assert s.timestamp_us == 1000
    assert s.accel_xyz == (0.0, 0.0, 1.0)
    assert s.gyro_xyz == (0.1, 0.2, 0.3)


def test_parse_rejects_bad_length() -> None:
    try:
        parse_imu_payload(b"\x00" * 55)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


class _FakeFooter:
    """iter_trailer_records が返すべき (id, format, data) を差し込むだけのダミー."""

    def __init__(self, records):
        self._records = records


def test_extract_gravity_averages_accel(monkeypatch) -> None:
    # 重力が概ね +Z, ノイズが平均で打ち消される合成 Gyro record.
    samples = b"".join(
        _entry(i, accel, (0.0, 0.0, 0.0))
        for i, accel in enumerate([(0.1, 0.0, 0.98), (-0.1, 0.0, 1.02)] * 50)
    )
    from sphere_reconstruct.insta360 import metadata as md

    monkeypatch.setattr(md, "iter_trailer_records", lambda footer: iter([(3, 0, samples)]))
    g = extract_gravity(_FakeFooter(None))
    assert g is not None
    assert g.sample_count == 100
    # 平均方向はほぼ +Z の単位ベクトル.
    gx, gy, gz = g.gravity_imu
    assert abs(gx) < 0.15 and abs(gy) < 1e-6 and gz > 0.98
    assert abs((gx * gx + gy * gy + gz * gz) ** 0.5 - 1.0) < 1e-9
    assert 0.99 < g.mean_magnitude < 1.03


def test_extract_gravity_none_without_gyro_record(monkeypatch) -> None:
    from sphere_reconstruct.insta360 import metadata as md

    monkeypatch.setattr(md, "iter_trailer_records", lambda footer: iter([(4, 0, b"\x00" * 56)]))
    assert extract_gravity(_FakeFooter(None)) is None


def test_extract_gravity_raw_u16_format(monkeypatch) -> None:
    # X5 実機の raw(u16) 形式: 20B/サンプル, 値 = u16 - 32768, accel 先.
    def _raw(accel):  # accel は -32768..32767 の生値
        return struct.pack("<Q6H", 0, *(int(a + 32768) for a in accel), 0, 0, 0)

    data = b"".join(_raw(a) for a in [(0, -2048, 0)] * 40)  # 重力が -Y (下向き軸)
    assert len(data) % 20 == 0
    from sphere_reconstruct.insta360 import metadata as md

    monkeypatch.setattr(md, "iter_trailer_records", lambda footer: iter([(3, 0, data)]))
    g = extract_gravity(_FakeFooter(None))
    assert g is not None and g.sample_count == 40
    assert g.gravity_imu == (0.0, -1.0, 0.0)
