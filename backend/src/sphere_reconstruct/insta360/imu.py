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
IMU_ENTRY_SIZE = 56  # 非 raw (f64) 形式: u64 ts + 3*f64 accel + 3*f64 gyro.
IMU_ENTRY_SIZE_RAW = 20  # raw (u16) 形式: u64 ts + 3*u16 accel + 3*u16 gyro (値は -32768 オフセット).


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


@dataclass
class ImuRecording:
    samples: list[ImuSample]
    timestamps_sec: list[float]
    orientation: str
    camera_type: str
    is_raw: bool


def parse_imu_payload(payload: bytes) -> list[ImuSample]:
    """Gyro record の payload (連続する 56 バイトエントリ) を parse する (accel 先, gyro 後)."""
    if len(payload) == 0 or len(payload) % IMU_ENTRY_SIZE != 0:
        raise ValueError(f"IMU payload length {len(payload)} is not a positive multiple of {IMU_ENTRY_SIZE}")
    out: list[ImuSample] = []
    for off in range(0, len(payload), IMU_ENTRY_SIZE):
        ts = struct.unpack_from("<Q", payload, off)[0]
        ax, ay, az, gx, gy, gz = struct.unpack_from("<6d", payload, off + 8)
        out.append(ImuSample(timestamp_us=ts, accel_xyz=(ax, ay, az), gyro_xyz=(gx, gy, gz)))
    return out


def parse_raw_imu_payload(payload: bytes) -> list[ImuSample]:
    """X5 raw gyro record (20B/entry) を signed 値へ戻して parse する."""
    if len(payload) == 0 or len(payload) % IMU_ENTRY_SIZE_RAW != 0:
        raise ValueError(
            f"raw IMU payload length {len(payload)} is not a positive multiple of {IMU_ENTRY_SIZE_RAW}"
        )
    samples: list[ImuSample] = []
    for offset in range(0, len(payload), IMU_ENTRY_SIZE_RAW):
        timestamp = struct.unpack_from("<Q", payload, offset)[0]
        values = tuple(float(value) - 32768.0 for value in struct.unpack_from("<6H", payload, offset + 8))
        samples.append(
            ImuSample(
                timestamp_us=timestamp,
                accel_xyz=(values[0], values[1], values[2]),
                gyro_xyz=(values[3], values[4], values[5]),
            )
        )
    return samples


def extract_imu_samples(footer) -> list[ImuSample]:
    """footer の gyro record を時刻付きサンプル列として返す."""
    from . import metadata as _md  # noqa: PLC0415

    extra = _md.read_extra_metadata(footer)
    for record_id, _record_format, data in _md.iter_trailer_records(footer):
        if record_id != GYRO_RECORD_ID or not data:
            continue
        if extra is not None:
            return parse_raw_imu_payload(data) if extra.is_raw_gyro else parse_imu_payload(data)
        raw_possible = len(data) % IMU_ENTRY_SIZE_RAW == 0
        float_possible = len(data) % IMU_ENTRY_SIZE == 0
        if raw_possible and not float_possible:
            return parse_raw_imu_payload(data)
        if float_possible and not raw_possible:
            return parse_imu_payload(data)
        if raw_possible and float_possible:
            return _select_ambiguous_format(data)
        raise ValueError(f"unsupported IMU record size: {len(data)} bytes")
    return []


def read_imu_samples(path) -> list[ImuSample]:
    """INSV から時刻付き IMU サンプルを読み出す."""
    from pathlib import Path

    from . import insv, metadata  # noqa: PLC0415

    source = Path(path)
    layout = insv.layout(source)
    if layout.footer_offset is None:
        return []
    footer = metadata.read_footer(source, layout.footer_offset)
    return extract_imu_samples(footer)


def read_imu_recording(path) -> ImuRecording | None:
    from pathlib import Path

    from . import insv, metadata  # noqa: PLC0415

    source = Path(path)
    layout = insv.layout(source)
    if layout.footer_offset is None:
        return None
    footer = metadata.read_footer(source, layout.footer_offset)
    extra = metadata.read_extra_metadata(footer)
    samples = extract_imu_samples(footer)
    if not samples:
        return None
    if extra is None:
        raise ValueError("Insta360 metadata record is required for IMU time alignment")

    if extra.is_raw_gyro:
        timestamps = [
            (sample.timestamp_us - extra.first_frame_timestamp) / 1_000_000.0
            - (extra.gyro_timestamp / 1_000.0 if extra.has_gyro_timestamp else 0.0)
            for sample in samples
        ]
    else:
        timestamps = [
            (sample.timestamp_us - extra.first_frame_timestamp) / 1_000.0
            - (extra.gyro_timestamp / 1_000.0 if extra.has_gyro_timestamp else 0.0)
            for sample in samples
        ]
    orientation = "yzX" if extra.camera_type == "Insta360 X5" else "Xyz"
    return ImuRecording(
        samples=samples,
        timestamps_sec=timestamps,
        orientation=orientation,
        camera_type=extra.camera_type,
        is_raw=extra.is_raw_gyro,
    )


def _select_ambiguous_format(payload: bytes) -> list[ImuSample]:
    candidates = [
        ("raw", parse_raw_imu_payload(payload)),
        ("float", parse_imu_payload(payload)),
    ]

    def score(kind: str, samples: list[ImuSample]) -> tuple[float, float, int]:
        monotonic = sum(
            current.timestamp_us > previous.timestamp_us
            for previous, current in zip(samples, samples[1:], strict=False)
        )
        rate = sample_rate_hz(samples) or 0.0
        plausible_rate = 1.0 if 50.0 <= rate <= 2_000.0 else 0.0
        magnitudes = [sum(value * value for value in sample.accel_xyz) ** 0.5 for sample in samples]
        median_magnitude = sorted(magnitudes)[len(magnitudes) // 2]
        if kind == "raw":
            plausible_magnitude = 1.0 if 10.0 <= median_magnitude <= 20_000.0 else 0.0
        else:
            plausible_magnitude = 1.0 if 0.05 <= median_magnitude <= 100.0 else 0.0
        return plausible_rate, plausible_magnitude, monotonic

    scored = sorted(
        ((score(kind, samples), samples) for kind, samples in candidates),
        key=lambda item: item[0],
    )
    if scored[0][0] == scored[1][0]:
        raise ValueError("ambiguous IMU record format; metadata raw flag is required")
    return scored[-1][1]


def extract_gravity(footer) -> GravityResult | None:
    """inst box の Gyro record を探し, accel の平均から重力方向 (IMU 座標) を返す.

    静止 or 平均的に無加速な区間では accel の平均 ≈ 重力の反力 = 上向き. 動きが激しい
    素材では並進加速度が混じるが, 長時間平均で概ね重力に収束する. record が見つからない /
    形式が想定外なら None (呼び出し側で「重力対齐なし」に降級).
    """
    samples = extract_imu_samples(footer)
    if samples:
        sx = sy = sz = 0.0
        mag = 0.0
        for sample in samples:
            ax, ay, az = sample.accel_xyz
            sx += ax
            sy += ay
            sz += az
            mag += (ax * ax + ay * ay + az * az) ** 0.5
        count = len(samples)
        mx, my, mz = sx / count, sy / count, sz / count
        norm = (mx * mx + my * my + mz * mz) ** 0.5
        if norm < 1e-6:
            return None
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
