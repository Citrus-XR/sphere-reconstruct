"""IMU レコード (record type 0x0300 = Gyro) の解釈と重力方向の抽出.

Insta360 の inst box 末尾 record 群のうち id=3 (Gyro) が IMU. 非 raw (f64) 形式の
1 サンプルは 56 バイト:

    timestamp: uint64 LE
    accel  x/y/z: 3 * float64   (単位 g, 静止時ノルム ~1)
    gyro   x/y/z: 3 * float64

バイト順は telemetry-parser (AdrianEddy/telemetry-parser, gyroflow が使う事実上の
リファレンス) に合わせ **accel が先, gyro が後** とする. サンプリングは ~200Hz.

軸の向きと IMU->camera 回転は個体差 / 機種差がありうる. ここでは生値と, 全サンプル
平均から求めた「重力方向 (IMU 座標)」だけを返す. camera / world への変換は上位で行う.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


IMU_RECORD_TYPE = 0x0300  # inst box record id=3 (Gyro). 0x0300 は旧観測メモの表記.
GYRO_RECORD_ID = 3
IMU_ENTRY_SIZE = 56       # 非 raw (f64) 形式: u64 ts + 3*f64 accel + 3*f64 gyro.
IMU_ENTRY_SIZE_RAW = 20   # raw (u16) 形式: u64 ts + 3*u16 accel + 3*u16 gyro (値は -32768 オフセット).


@dataclass
class ImuSample:
    timestamp_us: int
    accel_xyz: tuple[float, float, float]  # 単位 g
    gyro_xyz: tuple[float, float, float]


@dataclass
class GravityResult:
    """全 IMU サンプル平均から求めた重力方向 (IMU 座標系, 単位ベクトル)."""

    gravity_imu: tuple[float, float, float]  # 単位ベクトル (加速度計の平均方向 = 上向き)
    sample_count: int
    mean_magnitude: float  # 平均加速度のノルム (~1 なら単位 g, ~9.8 なら m/s^2)


def parse_imu_payload(payload: bytes) -> list[ImuSample]:
    """Gyro record の payload (連続する 56 バイトエントリ) を parse する (accel 先, gyro 後)."""
    if len(payload) == 0 or len(payload) % IMU_ENTRY_SIZE != 0:
        raise ValueError(
            f"IMU payload length {len(payload)} is not a positive multiple of {IMU_ENTRY_SIZE}"
        )
    out: list[ImuSample] = []
    for off in range(0, len(payload), IMU_ENTRY_SIZE):
        ts = struct.unpack_from("<Q", payload, off)[0]
        ax, ay, az, gx, gy, gz = struct.unpack_from("<6d", payload, off + 8)
        out.append(
            ImuSample(timestamp_us=ts, accel_xyz=(ax, ay, az), gyro_xyz=(gx, gy, gz))
        )
    return out


def extract_gravity(footer) -> GravityResult | None:
    """inst box の Gyro record を探し, accel の平均から重力方向 (IMU 座標) を返す.

    静止 or 平均的に無加速な区間では accel の平均 ≈ 重力の反力 = 上向き. 動きが激しい
    素材では並進加速度が混じるが, 長時間平均で概ね重力に収束する. record が見つからない /
    形式が想定外なら None (呼び出し側で「重力対齐なし」に降級).
    """
    from . import metadata as _md  # noqa: PLC0415

    for rid, _fmt, data in _md.iter_trailer_records(footer):
        if rid != GYRO_RECORD_ID or len(data) == 0:
            continue
        # 形式判定: f64(56B) を優先, 無ければ raw u16(20B). どちらでも accel が先, gyro が後.
        if len(data) % IMU_ENTRY_SIZE == 0:
            entry, raw = IMU_ENTRY_SIZE, False
        elif len(data) % IMU_ENTRY_SIZE_RAW == 0:
            entry, raw = IMU_ENTRY_SIZE_RAW, True
        else:
            continue
        sx = sy = sz = 0.0
        mag = 0.0
        count = 0
        for off in range(0, len(data), entry):
            if raw:
                ax, ay, az = (float(v) - 32768.0 for v in struct.unpack_from("<3H", data, off + 8))
            else:
                ax, ay, az = struct.unpack_from("<3d", data, off + 8)
            sx += ax
            sy += ay
            sz += az
            mag += (ax * ax + ay * ay + az * az) ** 0.5
            count += 1
        if count == 0:
            continue
        mx, my, mz = sx / count, sy / count, sz / count
        norm = (mx * mx + my * my + mz * mz) ** 0.5
        if norm < 1e-6:
            continue  # 平均がほぼ 0 (向きが定まらない) → 使えない.
        return GravityResult(
            gravity_imu=(mx / norm, my / norm, mz / norm),
            sample_count=count,
            mean_magnitude=mag / count,
        )
    return None


def sample_rate_hz(samples: list[ImuSample]) -> float | None:
    """連続する timestamp から素朴に推定 (μs 前提)."""
    if len(samples) < 2:
        return None
    dt_us = samples[-1].timestamp_us - samples[0].timestamp_us
    if dt_us <= 0:
        return None
    return (len(samples) - 1) / (dt_us / 1_000_000.0)
