"""INSV footer の versioned dual-fisheye calibration を解釈する。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..domain.camera_system import OmniDistortionModel


class CalibSource(StrEnum):
    OFFSET = "insta360_offset"


@dataclass(frozen=True)
class OmniLensCalibration:
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    yaw: float
    pitch: float
    roll: float
    tx: float
    ty: float
    tz: float
    distortion_model: OmniDistortionModel
    distortion_parameters: tuple[float, ...]
    ref_image_width: int
    ref_image_height: int
    lens_flags: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "xi": self.xi,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
            "distortion_model": self.distortion_model.value,
            "distortion_parameters": list(self.distortion_parameters),
            "ref_image_width": self.ref_image_width,
            "ref_image_height": self.ref_image_height,
            "lens_flags": self.lens_flags,
        }


@dataclass(frozen=True)
class DualLensCalibration:
    source: CalibSource
    version: int
    lenses: tuple[OmniLensCalibration, OmniLensCalibration]
    raw: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return all(lens.fx > 0.0 and lens.fy > 0.0 and lens.xi > 0.0 for lens in self.lenses)


@dataclass(frozen=True)
class OffsetCandidate:
    inst_offset: int
    text: str
    values: tuple[float, ...]

    @property
    def calibration_id(self) -> int:
        return int(self.values[-1]) if self.values else 0

    @property
    def version(self) -> int:
        return self.calibration_id >> 16


_ASCII_CALIBRATION = re.compile(rb"-?[0-9]+(?:\.[0-9]+)?(?:_-?[0-9]+(?:\.[0-9]+)?){5,}")
_LENS_PARAMETER_COUNTS = {3: 19, 6: 27}


def find_ascii_calibrations(inst_bytes: bytes) -> list[OffsetCandidate]:
    candidates = []
    for match in _ASCII_CALIBRATION.finditer(inst_bytes):
        text = match.group().decode("ascii", "strict")
        try:
            values = tuple(float(part) for part in text.split("_"))
        except ValueError:
            continue
        candidates.append(OffsetCandidate(match.start(), text, values))
    return candidates


def supported_candidate(candidate: OffsetCandidate) -> bool:
    lens_parameter_count = _LENS_PARAMETER_COUNTS.get(candidate.version)
    if lens_parameter_count is None or len(candidate.values) < 2:
        return False
    sensor_count = int(candidate.values[0])
    return sensor_count == 2 and len(candidate.values) == 2 + sensor_count * lens_parameter_count


def pick_calibration(candidates: list[OffsetCandidate]) -> OffsetCandidate | None:
    supported = [candidate for candidate in candidates if supported_candidate(candidate)]
    return max(supported, key=lambda candidate: (candidate.version, candidate.inst_offset), default=None)


def parse_ascii_calibration(candidate: OffsetCandidate) -> DualLensCalibration:
    if not supported_candidate(candidate):
        versions = ", ".join(str(version) for version in sorted(_LENS_PARAMETER_COUNTS))
        raise ValueError(
            f"未対応の Insta360 calibration です: version={candidate.version}, "
            f"items={len(candidate.values)}; supported versions: {versions}"
        )
    count = _LENS_PARAMETER_COUNTS[candidate.version]
    first = _parse_lens(candidate.values[1 : 1 + count], candidate.version)
    second = _parse_lens(candidate.values[1 + count : 1 + count * 2], candidate.version)
    return DualLensCalibration(
        source=CalibSource.OFFSET,
        version=candidate.version,
        lenses=(first, second),
        raw={
            "kind": "ascii_underscore",
            "calibration_id": candidate.calibration_id,
            "text": candidate.text,
            "inst_offset": candidate.inst_offset,
        },
    )


def _parse_lens(items: tuple[float, ...], version: int) -> OmniLensCalibration:
    common = items[:11]
    if version == 3:
        distortion_model = OmniDistortionModel.RADTAN
        distortion = items[11:16]
        trailer = items[16:]
    elif version == 6:
        distortion_model = OmniDistortionModel.RADTAN_PRO
        distortion = items[11:24]
        trailer = items[24:]
    else:
        raise ValueError(f"未対応の Insta360 calibration version です: {version}")
    if len(common) != 11 or len(trailer) != 3:
        raise ValueError(f"calibration lens block が不正です: version={version}, items={len(items)}")
    xi, fx, fy, cx, cy, yaw, pitch, roll, tx, ty, tz = common
    ref_width, ref_height, lens_flags = trailer
    return OmniLensCalibration(
        xi=xi,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        tx=tx,
        ty=ty,
        tz=tz,
        distortion_model=distortion_model,
        distortion_parameters=tuple(distortion),
        ref_image_width=int(ref_width),
        ref_image_height=int(ref_height),
        lens_flags=int(lens_flags),
    )
