"""Raw fisheye を SfM / trainer 共通の OPENCV_FISHEYE grid へ一回だけ再投影する。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from ..domain.artifacts import FileRef, StageManifest
from ..domain.camera_system import OmniIntrinsics
from ..domain.pipeline_state import StageName
from ..imaging import fisheye_camera, valid_region
from ..imaging.projection import project_omni, unproject_omni
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest


@register
class RectifyFisheye(Stage):
    name = StageName.RECTIFY_FISHEYE
    impl_version = "2.1"

    def normalize_params(self, raw: dict) -> dict:
        projection_contract = str(raw.get("projection_contract", "radtan_pro_v2"))
        if projection_contract != "radtan_pro_v2":
            raise ValueError(f"unsupported projection contract: {projection_contract}")
        return {"projection_contract": projection_contract}

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "prepare_images.json",
            ctx.project_dir / "prepare_images" / "image_catalog.json",
            ctx.project_dir / "prepare_images" / "rig_config.json",
        ]
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.is_file()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        source_catalog_path = ctx.project_dir / "prepare_images" / "image_catalog.json"
        if not source_catalog_path.is_file():
            raise RuntimeError("prepare_images を先に実行してください")
        source_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
        catalog = json.loads(json.dumps(source_catalog))
        groups = {group["id"]: group for group in catalog["camera_groups"]}
        source_records = {record["name"]: record for record in source_catalog["images"]}
        target_records = {record["name"]: record for record in catalog["images"]}
        rectified_groups = {
            group_id: group["rectification"]
            for group_id, group in groups.items()
            if (group.get("rectification") or {}).get("required") is True
        }

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        outputs: list[FileRef] = []
        reports = []
        quality_reports = []
        name_mapping: dict[str, str] = {}
        target_by_source_sensor: dict[tuple[str, str], tuple[str, list[float]]] = {}
        total_images = sum(
            len(groups[group_id]["image_names"]) for group_id in rectified_groups
        )
        completed_images = 0
        image_span = ProgressSpan(ctx.progress, 0.05, 0.95)

        cv2 = _cv2()
        cv2.setNumThreads(1)
        for group_number, (group_id, rectification) in enumerate(rectified_groups.items(), 1):
            group = groups[group_id]
            image_names = list(group["image_names"])
            if not image_names:
                raise ValueError(f"rectification group に image がありません: {group_id}")
            records = [source_records[name] for name in image_names]
            first = records[0]
            width, height = int(first["width"]), int(first["height"])
            if any((int(record["width"]), int(record["height"])) != (width, height) for record in records):
                raise ValueError(f"rectification group の image size が一致しません: {group_id}")
            source_projection = OmniIntrinsics.from_dict(rectification["source_projection"])
            if (source_projection.width, source_projection.height) != (width, height):
                raise ValueError(
                    f"rectification source projection size が image と一致しません: "
                    f"{source_projection.width}x{source_projection.height} != {width}x{height}"
                )
            target_model = str(rectification["target_camera_model"])
            target_params = [float(value) for value in rectification["target_camera_params"]]
            ctx.progress.tick(
                image_span.value(completed_images / max(1, total_images)),
                message=f"build fisheye remap {group_number}/{len(rectified_groups)}",
                key="log.rectify_build_map",
                args={"cur": group_number, "tot": len(rectified_groups)},
            )
            map_x, map_y, geometric_validity = _build_remap(
                source_projection,
                target_model,
                target_params,
                width,
                height,
                float(first["valid_region"]["max_theta_rad"]),
            )
            source_validity = valid_region.render_mask(first["valid_region"], width, height)
            target_validity = cv2.remap(
                source_validity,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            target_validity = ((target_validity > 0) & geometric_validity).astype(np.uint8)
            source_id = str(first["source_id"])
            sensor_id = str(first["sensor_id"])
            validity_path = (
                ctx.stage_out_dir / "validity" / source_id / f"{sensor_id}.png"
            )
            validity_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target_validity * 255, mode="L").save(
                validity_path,
                format="PNG",
                compress_level=6,
            )
            outputs.append(_file_ref(validity_path, ctx, "image/png"))

            jobs = []
            with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 1))) as executor:
                for record in records:
                    source_path = ctx.project_dir / record["path"]
                    target_name = str(PurePosixPath(record["name"]).with_suffix(".png"))
                    target_path = ctx.stage_out_dir / "images" / Path(target_name)
                    jobs.append(
                        (
                            record,
                            target_name,
                            target_path,
                            executor.submit(
                                _rectify_image,
                                source_path,
                                target_path,
                                map_x,
                                map_y,
                                geometric_validity,
                            ),
                        )
                    )
                for record, target_name, target_path, future in jobs:
                    source_bytes, target_bytes = future.result()
                    name_mapping[record["name"]] = target_name
                    target_record = target_records[record["name"]]
                    target_record.update(
                        {
                            "name": target_name,
                            "path": _final_relpath(target_path, ctx),
                            "valid_mask_path": _final_relpath(validity_path, ctx),
                            "valid_region": {
                                "kind": "fisheye",
                                "camera_model": target_model,
                                "params": target_params,
                                "max_theta_rad": float(record["valid_region"]["max_theta_rad"]),
                            },
                        }
                    )
                    outputs.append(_file_ref(target_path, ctx, "image/png"))
                    completed_images += 1
                    image_span.tick(
                        completed_images / max(1, total_images),
                        message=f"rectify image {completed_images}/{total_images}",
                        key="log.rectify_image_progress",
                        args={"cur": completed_images, "tot": total_images},
                    )
                    reports.append(
                        {
                            "name": target_name,
                            "source_bytes": source_bytes,
                            "target_bytes": target_bytes,
                        }
                    )
            group["camera_model"] = target_model
            group["camera_params"] = target_params
            group["image_names"] = [name_mapping[name] for name in image_names]
            group["rectification_result"] = {
                "applied": True,
                "source_model": "OMNI",
                "target_model": target_model,
                "interpolation": "lanczos4",
                "image_format": "png",
                "valid_pixels": int(target_validity.sum()),
                "total_pixels": width * height,
            }
            quality = _roundtrip_quality(
                ctx.project_dir / records[0]["path"],
                jobs[0][2],
                source_projection,
                target_model,
                target_params,
                source_validity,
            )
            group["rectification_result"]["roundtrip_quality"] = quality
            quality_reports.append({"group_id": group_id, **quality})
            target_by_source_sensor[(source_id, sensor_id)] = (target_model, target_params)

        catalog["version"] = 3
        catalog["images"] = [
            target_records[record["name"]] for record in source_catalog["images"]
        ]
        catalog["rectification"] = {
            "mandatory_for_raw_fisheye": True,
            "rectified_groups": len(rectified_groups),
            "rectified_images": completed_images,
            "passthrough_images": len(catalog["images"]) - completed_images,
            "interpolation": "lanczos4",
            "image_format": "png",
        }
        for calibration in catalog.get("source_calibrations", []):
            calibration["rectified_for_consumers"] = any(
                source_id == calibration["source_id"]
                for source_id, _sensor_id in target_by_source_sensor
            )
        rig_path = _write_rig_config(ctx, catalog, target_by_source_sensor)
        if rig_path is not None:
            outputs.append(_file_ref(rig_path, ctx, "application/json"))
            catalog["rig_config_path"] = "rig_config.json"
        catalog_path = ctx.stage_out_dir / "image_catalog.json"
        catalog_path.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs.append(_file_ref(catalog_path, ctx, "application/json"))
        manifest.outputs = outputs
        manifest.extra = {
            **catalog["rectification"],
            "source_bytes": sum(report["source_bytes"] for report in reports),
            "target_bytes": sum(report["target_bytes"] for report in reports),
            "camera_models": sorted({group["camera_model"] for group in groups.values()}),
            "roundtrip_quality": quality_reports,
        }
        ctx.progress.info(
            f"rectify_fisheye done: {completed_images} images",
            progress=0.99,
            key="log.rectify_done",
            args={"count": completed_images},
        )
        return manifest


def _build_remap(
    source: OmniIntrinsics,
    target_model: str,
    target_params: list[float],
    width: int,
    height: int,
    maximum_theta_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fisheye_camera.validate_forward_hemisphere(
        target_model,
        target_params,
        maximum_theta_rad,
    )
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    valid = np.empty((height, width), dtype=bool)
    pixel_x = np.arange(width, dtype=np.float64)
    for start in range(0, height, 128):
        end = min(height, start + 128)
        grid_x, grid_y = np.meshgrid(pixel_x, np.arange(start, end, dtype=np.float64))
        target_pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rays = fisheye_camera.pixels_to_camera_rays(
                target_model,
                target_params,
                target_pixels,
            )
        source_pixels, source_valid = project_omni(rays, source)
        target_theta = np.arccos(np.clip(rays[:, 2], -1.0, 1.0))
        target_valid = target_theta < maximum_theta_rad
        block_valid = source_valid & target_valid & np.all(np.isfinite(source_pixels), axis=1)
        block_x = source_pixels[:, 0].reshape((end - start, width)).astype(np.float32)
        block_y = source_pixels[:, 1].reshape((end - start, width)).astype(np.float32)
        block_valid_2d = block_valid.reshape((end - start, width))
        map_x[start:end] = np.where(block_valid_2d, block_x, -1.0)
        map_y[start:end] = np.where(block_valid_2d, block_y, -1.0)
        valid[start:end] = block_valid_2d
    return map_x, map_y, valid


def _rectify_image(
    source: Path,
    target: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    validity: np.ndarray,
) -> tuple[int, int]:
    cv2 = _cv2()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    if image.shape[:2] != map_x.shape:
        raise ValueError(
            f"rectification image size が map と一致しません: {image.shape[1]}x{image.shape[0]}"
        )
    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rectified[validity == 0] = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), rectified, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"rectified PNG を保存できません: {target}")
    return source.stat().st_size, target.stat().st_size


def _roundtrip_quality(
    source_path: Path,
    rectified_path: Path,
    source_projection: OmniIntrinsics,
    target_model: str,
    target_params: list[float],
    source_validity: np.ndarray,
) -> dict:
    cv2 = _cv2()
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    rectified = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
    if source is None or rectified is None:
        raise RuntimeError("rectification roundtrip quality の画像を読み込めません")
    height, width = source.shape[:2]
    step = max(1, max(width, height) // 960)
    sample_x = np.arange(0, width, step, dtype=np.float64)
    sample_y = np.arange(0, height, step, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    source_pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    rays, source_rays_valid = unproject_omni(source_pixels, source_projection)
    target_pixels = fisheye_camera.camera_rays_to_pixels(target_model, target_params, rays)
    target_in_frame = (
        (target_pixels[:, 0] >= 0.0)
        & (target_pixels[:, 0] <= width - 1)
        & (target_pixels[:, 1] >= 0.0)
        & (target_pixels[:, 1] <= height - 1)
    )
    sampled_rectified = cv2.remap(
        rectified,
        target_pixels[:, 0].reshape(grid_x.shape).astype(np.float32),
        target_pixels[:, 1].reshape(grid_y.shape).astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).reshape((-1, 3))
    sampled_source = source[::step, ::step].reshape((-1, 3))
    mask = (
        source_rays_valid
        & target_in_frame
        & (source_validity[::step, ::step].ravel() > 0)
    )
    if not mask.any():
        raise RuntimeError("rectification roundtrip quality に valid sample がありません")
    delta = sampled_rectified[mask].astype(np.float64) - sampled_source[mask].astype(np.float64)
    mse = float(np.mean(delta * delta))
    return {
        "sample_step": step,
        "samples": int(mask.sum()),
        "mae_8bit": float(np.mean(np.abs(delta))),
        "psnr_db": float(10.0 * np.log10(255.0 * 255.0 / max(mse, 1e-12))),
    }


def _write_rig_config(
    ctx: StageContext,
    catalog: dict,
    target_by_source_sensor: dict[tuple[str, str], tuple[str, list[float]]],
) -> Path | None:
    source = ctx.project_dir / "prepare_images" / "rig_config.json"
    if not source.is_file():
        return None
    rigs = json.loads(source.read_text(encoding="utf-8"))
    for rig in rigs:
        for camera in rig["cameras"]:
            prefix = PurePosixPath(str(camera["image_prefix"]).replace("\\", "/"))
            parts = prefix.parts
            if len(parts) < 3 or parts[0] != "sources":
                continue
            key = (parts[1], parts[2])
            target = target_by_source_sensor.get(key)
            if target is not None:
                camera["camera_model_name"] = target[0]
                camera["camera_params"] = target[1]
    destination = ctx.stage_out_dir / "rig_config.json"
    destination.write_text(json.dumps(rigs, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


def _final_relpath(path: Path, ctx: StageContext) -> str:
    return str(Path(ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")) / path.relative_to(ctx.stage_out_dir))


def _file_ref(path: Path, ctx: StageContext, mime: str) -> FileRef:
    return FileRef(
        path=_final_relpath(path, ctx),
        size=path.stat().st_size,
        sha256=sha256_file(path) if path.suffix == ".json" else "",
        mime=mime,
    )
