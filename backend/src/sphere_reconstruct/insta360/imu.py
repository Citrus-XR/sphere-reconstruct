"""IMU レコード (record type 0x0300) の解釈.

観察例 (insv-stitch の x5_pipeline.md 記述より, X5 の 1 サンプルでの観測):

  record id: 0x0300
  record size: 56 bytes
  payload:
    timestamp:   uint64 little-endian
    gyro   x/y/z: 3 * float64
    accel  x/y/z: 3 * float64

サンプリングレートは約 200Hz と観測されているが, ファーム差の可能性があるので
コード側では固定値としない.

軸の向きと IMU->camera の回転行列は個体差 / 機種差の可能性がある. ここでは
生値だけを返し, カメラ座標系への変換は calibration.py 側で行う.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


IMU_RECORD_TYPE = 0x0300
IMU_ENTRY_SIZE = 56


@dataclass
class ImuSample:
    timestamp_us: int  # マイクロ秒基準と仮定 (要検証)
    gyro_xyz: tuple[float, float, float]  # rad/s と仮定 (要検証)
    accel_xyz: tuple[float, float, float]  # m/s^2 と仮定 (要検証)


def parse_imu_payload(payload: bytes) -> list[ImuSample]:
    """IMU レコードの payload (連続する 56 バイトエントリの配列) を parse する."""
    if len(payload) % IMU_ENTRY_SIZE != 0:
        raise ValueError(
            f"IMU payload length {len(payload)} is not a multiple of {IMU_ENTRY_SIZE}"
        )
    out: list[ImuSample] = []
    for off in range(0, len(payload), IMU_ENTRY_SIZE):
        ts = struct.unpack_from("<Q", payload, off)[0]
        gx, gy, gz, ax, ay, az = struct.unpack_from("<6d", payload, off + 8)
        out.append(
            ImuSample(
                timestamp_us=ts,
                gyro_xyz=(gx, gy, gz),
                accel_xyz=(ax, ay, az),
            )
        )
    return out


def sample_rate_hz(samples: list[ImuSample]) -> float | None:
    """連続する timestamp から素朴に推定. 単位が μs 前提."""
    if len(samples) < 2:
        return None
    dt_us = samples[-1].timestamp_us - samples[0].timestamp_us
    if dt_us <= 0:
        return None
    return (len(samples) - 1) / (dt_us / 1_000_000.0)


def integrate_rotation(samples: list[ImuSample], t_start_us: int, t_end_us: int) -> float:
    """[t_start, t_end] の間の gyro を積分し, 総回転角 (ラジアン) を返す.

    空間抽出のフレーム間隔判定に使う. gyro の各軸を独立に積分し, 角速度ベクトルの
    ノルムを台形則で積分する (小角近似での総回転量). サンプルが範囲外なら 0.

    gyro の単位は rad/s 前提 (imu.py の観測メモ参照). 単位が deg/s だった場合は
    呼び出し側で換算するか, ここを調整する.
    """
    if t_end_us <= t_start_us:
        return 0.0
    # 範囲内のサンプルを時刻順に取る.
    inrange = [s for s in samples if t_start_us <= s.timestamp_us <= t_end_us]
    if len(inrange) < 2:
        return 0.0
    inrange.sort(key=lambda s: s.timestamp_us)
    total = 0.0
    for a, b in zip(inrange[:-1], inrange[1:], strict=False):
        dt = (b.timestamp_us - a.timestamp_us) / 1_000_000.0
        if dt <= 0:
            continue
        wa = _norm3(a.gyro_xyz)
        wb = _norm3(b.gyro_xyz)
        total += 0.5 * (wa + wb) * dt  # 台形則
    return total


def _norm3(v: tuple[float, float, float]) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
