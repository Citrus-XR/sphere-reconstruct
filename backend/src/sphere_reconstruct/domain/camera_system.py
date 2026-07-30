"""カメラ固有メタデータから独立した校正済み camera system。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CANONICAL_COORDINATE_SYSTEM = {
    "handedness": "right",
    "x_axis": "right",
    "y_axis": "down",
    "z_axis": "forward",
    "length_unit": "meter",
    "pixel_origin": "top_left_pixel_center_0",
}


@dataclass(frozen=True)
class MeiIntrinsics:
    """単一センサー画像のローカル座標で表した MEI 内部パラメータ。"""

    width: int
    height: int
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"MEI の参照サイズが不正です: {self.width}x{self.height}")
        if self.xi <= 0.0 or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("MEI の xi/fx/fy は正数でなければなりません")
        if not all(
            math.isfinite(value)
            for value in (
                self.xi,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.k1,
                self.k2,
                self.k3,
                self.p1,
                self.p2,
            )
        ):
            raise ValueError("MEI parameter に non-finite value があります")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MeiIntrinsics:
        if value["model"] != "mei":
            raise ValueError(f"未対応の投影モデルです: {value['model']}")
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            xi=float(value["xi"]),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
            k1=float(value["k1"]),
            k2=float(value["k2"]),
            k3=float(value["k3"]),
            p1=float(value["p1"]),
            p2=float(value["p2"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"model": "mei", **self.__dict__}

    def scaled(self, width: int, height: int) -> MeiIntrinsics:
        if width <= 0 or height <= 0:
            raise ValueError(f"変換先サイズが不正です: {width}x{height}")
        scale_x = width / self.width
        scale_y = height / self.height
        return MeiIntrinsics(
            width=width,
            height=height,
            xi=self.xi,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            p1=self.p1,
            p2=self.p2,
        )


@dataclass(frozen=True)
class SensorExtrinsic:
    """COLMAP と同じ ``cam_from_rig`` 剛体変換。"""

    rotation_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = (*self.rotation_wxyz, *self.translation_xyz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("cam_from_rig に non-finite value があります")
        norm = math.sqrt(sum(value * value for value in self.rotation_wxyz))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"cam_from_rig quaternion が unit ではありません: {norm}")

    @classmethod
    def identity(cls) -> SensorExtrinsic:
        return cls((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SensorExtrinsic:
        rotation = tuple(float(item) for item in value["rotation_wxyz"])
        translation = tuple(float(item) for item in value["translation_xyz"])
        if len(rotation) != 4 or len(translation) != 3:
            raise ValueError("cam_from_rig は quaternion 4 要素と translation 3 要素が必要です")
        return cls(rotation, translation)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "rotation_wxyz": list(self.rotation_wxyz),
            "translation_xyz": list(self.translation_xyz),
        }


@dataclass(frozen=True)
class CalibrationImageTransform:
    """Metadata の sensor-local reference image から校正適用 image への crop。"""

    reference_width: int
    reference_height: int
    crop_x: float
    crop_y: float
    crop_width: int
    crop_height: int

    def __post_init__(self) -> None:
        if min(self.reference_width, self.reference_height, self.crop_width, self.crop_height) <= 0:
            raise ValueError("calibration image transform の size は正数でなければなりません")
        if not math.isfinite(self.crop_x) or not math.isfinite(self.crop_y):
            raise ValueError("calibration image transform の crop origin が non-finite です")
        if self.crop_x < 0.0 or self.crop_y < 0.0:
            raise ValueError("calibration image transform の crop origin は 0 以上でなければなりません")
        if (
            self.crop_x + self.crop_width > self.reference_width
            or self.crop_y + self.crop_height > self.reference_height
        ):
            raise ValueError("calibration image transform の crop が reference image を超えています")

    @classmethod
    def identity(cls, width: int, height: int) -> CalibrationImageTransform:
        return cls(width, height, 0.0, 0.0, width, height)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CalibrationImageTransform:
        return cls(
            reference_width=int(value["reference_width"]),
            reference_height=int(value["reference_height"]),
            crop_x=float(value["crop_x"]),
            crop_y=float(value["crop_y"]),
            crop_width=int(value["crop_width"]),
            crop_height=int(value["crop_height"]),
        )

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


class ShutterType(StrEnum):
    GLOBAL = "global"
    ROLLING = "rolling"
    UNKNOWN = "unknown"


class ScanDirection(StrEnum):
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    UNKNOWN = "unknown"


class TimestampReference(StrEnum):
    START = "start"
    CENTER = "center"
    END = "end"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ShutterCalibration:
    type: ShutterType
    readout_time_ms: float | None
    scan_direction: ScanDirection
    timestamp_reference: TimestampReference

    def __post_init__(self) -> None:
        if self.type == ShutterType.ROLLING:
            if (
                self.readout_time_ms is None
                or not math.isfinite(self.readout_time_ms)
                or self.readout_time_ms <= 0.0
            ):
                raise ValueError("rolling shutter には正の readout time が必要です")
        elif self.readout_time_ms is not None:
            raise ValueError("rolling 以外の shutter に readout time は指定できません")

    @classmethod
    def global_shutter(cls) -> ShutterCalibration:
        return cls(
            type=ShutterType.GLOBAL,
            readout_time_ms=None,
            scan_direction=ScanDirection.UNKNOWN,
            timestamp_reference=TimestampReference.UNKNOWN,
        )

    @classmethod
    def unknown(cls) -> ShutterCalibration:
        return cls(
            type=ShutterType.UNKNOWN,
            readout_time_ms=None,
            scan_direction=ScanDirection.UNKNOWN,
            timestamp_reference=TimestampReference.UNKNOWN,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ShutterCalibration:
        readout = value["readout_time_ms"]
        return cls(
            type=ShutterType(value["type"]),
            readout_time_ms=float(readout) if readout is not None else None,
            scan_direction=ScanDirection(value["scan_direction"]),
            timestamp_reference=TimestampReference(value["timestamp_reference"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "readout_time_ms": self.readout_time_ms,
            "scan_direction": self.scan_direction.value,
            "timestamp_reference": self.timestamp_reference.value,
        }


@dataclass(frozen=True)
class CalibratedSensor:
    id: str
    image_key: str
    intrinsics: MeiIntrinsics
    calibration_image_transform: CalibrationImageTransform
    cam_from_rig: SensorExtrinsic
    shutter: ShutterCalibration

    def __post_init__(self) -> None:
        if not self.id or not self.image_key:
            raise ValueError("sensor id と image_key は空にできません")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CalibratedSensor:
        return cls(
            id=str(value["id"]),
            image_key=str(value["image_key"]),
            intrinsics=MeiIntrinsics.from_dict(value["projection"]),
            calibration_image_transform=CalibrationImageTransform.from_dict(
                value["calibration_image_transform"]
            ),
            cam_from_rig=SensorExtrinsic.from_dict(value["cam_from_rig"]),
            shutter=ShutterCalibration.from_dict(value["shutter"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_key": self.image_key,
            "projection": self.intrinsics.to_dict(),
            "calibration_image_transform": self.calibration_image_transform.to_dict(),
            "cam_from_rig": self.cam_from_rig.to_dict(),
            "shutter": self.shutter.to_dict(),
        }


@dataclass(frozen=True)
class CalibratedCameraSystem:
    """adapter が出力し、後段が唯一の校正入力として読む契約。"""

    calibration_source: str
    reference_sensor_id: str
    sensors: tuple[CalibratedSensor, ...]

    def __post_init__(self) -> None:
        sensor_ids = [sensor.id for sensor in self.sensors]
        image_keys = [sensor.image_key for sensor in self.sensors]
        if not self.calibration_source:
            raise ValueError("calibration source が空です")
        if not sensor_ids:
            raise ValueError("camera system に sensor がありません")
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("camera system の sensor ID が重複しています")
        if len(image_keys) != len(set(image_keys)):
            raise ValueError("camera system の image_key が重複しています")
        if self.reference_sensor_id not in sensor_ids:
            raise ValueError("reference sensor が camera system に存在しません")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CalibratedCameraSystem:
        if int(value["version"]) != 1:
            raise ValueError(f"未対応の camera system version です: {value['version']}")
        if value["coordinate_system"] != CANONICAL_COORDINATE_SYSTEM:
            raise ValueError("camera system の座標系が canonical contract と一致しません")
        return cls(
            calibration_source=str(value["calibration_source"]),
            reference_sensor_id=str(value["reference_sensor_id"]),
            sensors=tuple(CalibratedSensor.from_dict(sensor) for sensor in value["sensors"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "coordinate_system": dict(CANONICAL_COORDINATE_SYSTEM),
            "calibration_source": self.calibration_source,
            "reference_sensor_id": self.reference_sensor_id,
            "sensors": [sensor.to_dict() for sensor in self.sensors],
        }

    @property
    def maximum_rolling_shutter_readout_ms(self) -> float:
        return max(
            (
                sensor.shutter.readout_time_ms or 0.0
                for sensor in self.sensors
                if sensor.shutter.type == ShutterType.ROLLING
            ),
            default=0.0,
        )
