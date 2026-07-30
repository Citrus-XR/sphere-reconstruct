"""Stock LFStudio の camera-model 不整合を export 時に隔離する。"""

from __future__ import annotations

from dataclasses import replace

from .model import Reconstruction
from .training_crop import CropPlan


def apply_stock_camera_workarounds(
    reconstruction: Reconstruction,
    catalog: dict,
    crop_plan: CropPlan | None,
) -> tuple[Reconstruction, dict]:
    """THIN_PRISM を internally consistent な OPENCV_FISHEYE approximation へ置換する。"""
    records = {record["name"]: record for record in catalog["images"]}
    groups = {group["id"]: group for group in catalog.get("camera_groups", [])}
    compatibility_by_camera: dict[int, dict] = {}
    group_by_camera: dict[int, str] = {}
    for image in reconstruction.images.values():
        record = records[image.name]
        group_id = record.get("camera_group_id")
        if group_id is None or group_id not in groups:
            continue
        group = groups[group_id]
        compatibility = (group.get("consumer_compatibility") or {}).get("lichtfeld_stock")
        if compatibility is None:
            continue
        previous_group = group_by_camera.setdefault(image.camera_id, group_id)
        if previous_group != group_id:
            raise ValueError(
                f"camera {image.camera_id} が複数の compatibility group を参照しています: "
                f"{previous_group}, {group_id}"
            )
        compatibility_by_camera[image.camera_id] = compatibility

    cameras = dict(reconstruction.cameras)
    reports = []
    for camera_id, compatibility in compatibility_by_camera.items():
        camera = cameras[camera_id]
        if camera.model != "THIN_PRISM_FISHEYE":
            raise ValueError(
                f"camera {camera_id} の compatibility source model が不正です: {camera.model}"
            )
        if compatibility["camera_model"] != "OPENCV_FISHEYE":
            raise ValueError(
                f"camera {camera_id} の compatibility target model が不正です: "
                f"{compatibility['camera_model']}"
            )
        params = [float(value) for value in compatibility["camera_params"]]
        rect = crop_plan.cameras[camera_id] if crop_plan is not None else None
        if rect is not None:
            params[2] -= rect.left
            params[3] -= rect.top
        cameras[camera_id] = replace(
            camera,
            model=str(compatibility["camera_model"]),
            params=params,
            model_id=5,
        )
        reports.append(
            {
                "camera_id": camera_id,
                "source_model": camera.model,
                "export_model": compatibility["camera_model"],
                "rms_error_px": float(compatibility["rms_error_px"]),
                "maximum_error_px": float(compatibility["maximum_error_px"]),
                "reason": compatibility["reason"],
            }
        )
    return (
        Reconstruction(
            cameras=cameras,
            images=reconstruction.images,
            points3D=reconstruction.points3D,
        ),
        {
            "applied": bool(reports),
            "consumer": "lichtfeld_stock",
            "camera_reports": reports,
        },
    )
