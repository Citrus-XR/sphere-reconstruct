"""校正済み multi-sensor rig の COLMAP 設定生成。"""

from __future__ import annotations

from ..domain.camera_system import SensorExtrinsic


def build_rig_config(
    camera_model_name: str,
    camera_params_by_sensor: dict[str, list[float]],
    sensor_extrinsics: dict[str, SensorExtrinsic],
    *,
    prefix: str = "",
) -> list[dict]:
    """Adapter が正規化した外参から rig_configurator 入力を作る。"""
    if not sensor_extrinsics:
        raise ValueError("rig に sensor がありません")
    if camera_params_by_sensor.keys() != sensor_extrinsics.keys():
        raise ValueError("camera parameter と sensor extrinsic の sensor ID が一致しません")
    cameras = []
    for index, (sensor_id, extrinsic) in enumerate(sensor_extrinsics.items()):
        camera = {
            "image_prefix": f"{prefix}{sensor_id}/",
            "camera_model_name": camera_model_name,
            "camera_params": list(camera_params_by_sensor[sensor_id]),
        }
        if index == 0:
            camera["ref_sensor"] = True
        else:
            camera["cam_from_rig_rotation"] = list(extrinsic.rotation_wxyz)
            camera["cam_from_rig_translation"] = list(extrinsic.translation_xyz)
        cameras.append(camera)
    return [{"cameras": cameras}]
