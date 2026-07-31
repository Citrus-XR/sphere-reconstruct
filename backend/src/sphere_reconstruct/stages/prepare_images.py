"""全 source を COLMAP 用の canonical image catalog へ変換する。"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..colmap import calibrated_rig
from ..colmap import rig as colmap_rig
from ..domain.artifacts import FileRef, StageManifest
from ..domain.camera_system import CalibratedCameraSystem, SensorExtrinsic
from ..domain.pipeline_state import StageName
from ..domain.source import Projection, SourceRole
from ..imaging import fisheye_region, projection, rendering
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest

EXIF_MAKE = 271
EXIF_MODEL = 272
EXIF_FOCAL_35MM = 41989
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


@register
class PrepareImages(Stage):
    name = StageName.PREPARE_IMAGES
    impl_version = "3.2"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "inspect_source.json",
            ctx.project_dir / "manifests" / "extract_frames.json",
            ctx.project_dir / "inspect_source" / "sources.json",
            ctx.project_dir / "extract_frames" / "manifest_frames.json",
            fisheye_region.region_path(ctx.project_dir),
            *sorted((ctx.project_dir / "inspect_source" / "sources").glob("*/camera_system.json")),
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

    def normalize_params(self, raw: dict) -> dict:
        mode = str(raw.get("reconstruction_mode", "native_fisheye"))
        if mode not in {"native_fisheye", "equirectangular", "pinhole_rig"}:
            raise ValueError(f"unsupported reconstruction mode: {mode}")
        return {
            "reconstruction_mode": mode,
            "size": int(raw.get("size", 1024)),
            "fov_deg": float(raw.get("fov_deg", 90.0)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        frames_path = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        inspections_path = ctx.project_dir / "inspect_source" / "sources.json"
        if not frames_path.is_file() or not inspections_path.is_file():
            raise RuntimeError("inspect_source and extract_frames must run before prepare_images")
        frames_document = json.loads(frames_path.read_text(encoding="utf-8"))
        inspection_document = json.loads(inspections_path.read_text(encoding="utf-8"))
        if int(inspection_document["version"]) != 3:
            raise RuntimeError("source inspection artifact を現在の adapter contract で再生成してください")
        inspections = {
            source["id"]: source
            for source in inspection_document["sources"]
        }

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        images: list[dict] = []
        groups: list[dict] = []
        rigs: list[dict] = []
        outputs: list[FileRef] = []
        source_calibrations: list[dict] = []
        source_documents = frames_document["sources"]
        for source_number, source in enumerate(source_documents):
            source_span = ProgressSpan(
                ctx.progress,
                0.98 * source_number / max(1, len(source_documents)),
                0.98 * (source_number + 1) / max(1, len(source_documents)),
            )
            ctx.progress.info(
                f"preparing images: {source['label']}",
                progress=source_span.low,
                key="log.prepare_source_start",
                args={
                    "source": source["label"],
                    "cur": source_number + 1,
                    "tot": len(source_documents),
                },
            )
            inspection = inspections[source["id"]]
            projection_name = Projection(source["projection"])
            if projection_name == Projection.DUAL_FISHEYE:
                if ctx.params["reconstruction_mode"] == "pinhole_rig":
                    result = self._prepare_dual_fisheye_pinhole(ctx, source, inspection, source_span)
                else:
                    result = self._prepare_dual_fisheye_native(ctx, source, inspection, source_span)
            elif projection_name == Projection.EQUIRECTANGULAR:
                if ctx.params["reconstruction_mode"] == "pinhole_rig":
                    result = self._prepare_equirectangular_pinhole(ctx, source, source_span)
                else:
                    result = self._prepare_equirectangular_native(ctx, source, source_span)
            else:
                result = self._prepare_perspective(ctx, source, source_span)
            images.extend(result["images"])
            groups.extend(result["camera_groups"])
            rigs.extend(result["rigs"])
            outputs.extend(result["outputs"])
            if result.get("calibration") is not None:
                source_calibrations.append({"source_id": source["id"], **result["calibration"]})

        primary = next(
            (source for source in frames_document["sources"] if source["role"] == SourceRole.PRIMARY.value),
            None,
        )
        if primary is None:
            raise RuntimeError("primary source is missing")
        rig_path = None
        if rigs:
            rig_file = ctx.stage_out_dir / "rig_config.json"
            rig_file.write_text(json.dumps(rigs, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(_file_ref(rig_file, ctx, "application/json"))
            rig_path = "rig_config.json"
        catalog = {
            "version": 2,
            "reconstruction_mode": ctx.params["reconstruction_mode"],
            "primary_source_id": primary["id"],
            "sources": [
                {
                    key: source[key]
                    for key in ("id", "label", "role", "adapter", "media_kind", "projection", "count")
                }
                for source in frames_document["sources"]
            ],
            "camera_groups": groups,
            "rig_config_path": rig_path,
            "images": images,
            "source_calibrations": source_calibrations,
        }
        catalog_path = ctx.stage_out_dir / "image_catalog.json"
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(_file_ref(catalog_path, ctx, "application/json"))
        manifest.outputs = outputs
        manifest.extra = {
            "sources": len(catalog["sources"]),
            "images": len(images),
            "camera_groups": len(groups),
            "rigs": len(rigs),
            "camera_models": sorted({group["camera_model"] for group in groups}),
            "perspective_images": sum(image["projection"] == "perspective" for image in images),
            "spherical_images": sum(image["projection"] != "perspective" for image in images),
            "source_calibrations": source_calibrations,
        }
        ctx.progress.info(
            f"prepare_images done: {len(images)} images, {len(groups)} camera groups",
            progress=0.99,
            key="log.prepare_done",
            args={"images": len(images), "cameras": len(groups)},
        )
        return manifest

    def _prepare_dual_fisheye_native(
        self,
        ctx: StageContext,
        source: dict,
        inspection: dict,
        progress_span: ProgressSpan,
    ) -> dict:
        width, height = int(source["width"]), int(source["height"])
        system = _load_camera_system(ctx.project_dir, inspection)
        if len(system.sensors) != 2:
            raise RuntimeError(
                f"source {source['label']}: dual-fisheye には 2 sensor が必要です"
            )
        intrinsics = [sensor.intrinsics.scaled(width, height) for sensor in system.sensors]
        approximations = [projection.approximate_thin_prism_fisheye(intr) for intr in intrinsics]
        compatible_approximations = [
            projection.approximate_opencv_fisheye(intr) for intr in intrinsics
        ]
        if any(approximation.maximum_error_px > 1.0 for approximation in approximations):
            raise RuntimeError(
                f"source {source['label']}: MEI → THIN_PRISM_FISHEYE の近似誤差が 1 px を超えました。"
                "この校正を native mode の default にできません"
            )
        sensors = tuple(sensor.id for sensor in system.sensors)
        params_by_sensor = {
            sensor: list(approximations[index].params) for index, sensor in enumerate(sensors)
        }
        prefix = f"sources/{source['id']}/"
        group_ids = {sensor: f"{source['id']}:native-fisheye:{sensor}" for sensor in sensors}
        region = fisheye_region.load_region(ctx.project_dir, source["id"])
        maximum_theta_rad = math.pi / 2 * 0.995
        images = []
        image_names = {sensor: [] for sensor in sensors}
        for frame in source["frames"]:
            for lens, sensor in enumerate(sensors):
                name = f"{prefix}{sensor}/frame_{frame['index']:06d}.jpg"
                image_names[sensor].append(name)
                images.append(
                    _catalog_image(
                        source,
                        frame,
                        name=name,
                        path=frame[system.sensors[lens].image_key],
                        width=width,
                        height=height,
                        camera_group_id=group_ids[sensor],
                        sensor_id=sensor,
                        projection_name="dual_fisheye",
                        valid_region={
                            "kind": "fisheye",
                            "camera_model": approximations[lens].camera_model,
                            "params": list(approximations[lens].params),
                            "max_theta_rad": maximum_theta_rad,
                            "physical_circle": region[sensor],
                        },
                    )
                )
        sensor_extrinsics = {
            sensor.id: sensor.cam_from_rig for sensor in system.sensors
        }
        progress_span.tick(
            1.0,
            message=f"{source['label']}: {len(images)} native fisheye images catalogued",
            key="log.prepare_source_progress",
            args={"source": source["label"], "cur": len(images), "tot": len(images)},
        )
        return {
            "images": images,
            "camera_groups": [
                _camera_group(
                    group_ids[sensor],
                    source["id"],
                    approximations[index].camera_model,
                    params_by_sensor[sensor],
                    image_names[sensor],
                    single_camera=True,
                    single_camera_per_folder=False,
                    refine_intrinsics=False,
                    rectification={
                        "required": True,
                        "source_projection": intrinsics[index].to_dict(),
                        "target_camera_model": compatible_approximations[index].camera_model,
                        "target_camera_params": list(compatible_approximations[index].params),
                        "rms_error_px_before_resampling": compatible_approximations[
                            index
                        ].rms_error_px,
                        "maximum_error_px_before_resampling": compatible_approximations[
                            index
                        ].maximum_error_px,
                    },
                )
                for index, sensor in enumerate(sensors)
            ],
            "rigs": calibrated_rig.build_rig_config(
                approximations[0].camera_model,
                params_by_sensor,
                sensor_extrinsics,
                prefix=prefix,
            ),
            "outputs": [],
            "calibration": {
                "method": f"{system.calibration_source}_mei_to_thin_prism_fisheye",
                "forward_hemisphere_only": True,
                "rolling_shutter_time_ms": system.maximum_rolling_shutter_readout_ms,
                "rolling_shutter_correction": "risk_filtered",
                "sensor_extrinsics": {
                    sensor: {
                        "rotation_wxyz": list(extrinsic.rotation_wxyz),
                        "translation_xyz": list(extrinsic.translation_xyz),
                    }
                    for sensor, extrinsic in sensor_extrinsics.items()
                },
                "lenses": [
                    {
                        "sensor": sensor,
                        "calibration_image_transform": system.sensors[
                            index
                        ].calibration_image_transform.to_dict(),
                        "rms_error_px": approximations[index].rms_error_px,
                        "maximum_error_px": approximations[index].maximum_error_px,
                        "colmap_rms_error_px": approximations[index].colmap_rms_error_px,
                        "colmap_maximum_error_px": approximations[index].colmap_maximum_error_px,
                        "lichtfeld_rms_error_px": approximations[index].lichtfeld_rms_error_px,
                        "lichtfeld_maximum_error_px": approximations[
                            index
                        ].lichtfeld_maximum_error_px,
                        "forward_radius_px": approximations[index].forward_radius_px,
                        "forward_theta_limit_deg": math.degrees(maximum_theta_rad),
                        "physical_valid_radius_ratio": float(region[sensor]["r"]),
                        "rectification_target": {
                            "camera_model": compatible_approximations[index].camera_model,
                            "rms_error_px": compatible_approximations[index].rms_error_px,
                            "maximum_error_px": compatible_approximations[index].maximum_error_px,
                        },
                    }
                    for index, sensor in enumerate(sensors)
                ],
            },
        }

    def _prepare_dual_fisheye_pinhole(
        self,
        ctx: StageContext,
        source: dict,
        inspection: dict,
        progress_span: ProgressSpan,
    ) -> dict:
        system = _load_camera_system(ctx.project_dir, inspection)
        intrinsics = [
            sensor.intrinsics.scaled(int(source["width"]), int(source["height"]))
            for sensor in system.sensors
        ]
        rotations = [_quaternion_to_rotation(sensor.cam_from_rig) for sensor in system.sensors]
        centers = [_camera_center(sensor.cam_from_rig) for sensor in system.sensors]
        return self._render_pinhole_views(
            ctx,
            source,
            lenses=[
                {"index": index, "tx": center[0], "ty": center[1], "tz": center[2]}
                for index, center in enumerate(centers)
            ],
            render=lambda frame, view, lens: rendering.render_pinhole(
                ctx.project_dir / frame[system.sensors[lens].image_key],
                view,
                intrinsics[lens],
                extra_rotation=rotations[lens],
            ),
            progress_span=progress_span,
        )

    def _prepare_equirectangular_native(
        self, ctx: StageContext, source: dict, progress_span: ProgressSpan
    ) -> dict:
        group_id = f"{source['id']}:equirectangular"
        images = []
        names = []
        for frame_number, frame in enumerate(source["frames"], 1):
            input_path = _frame_image_path(ctx.project_dir, frame)
            output_path = ctx.stage_out_dir / "sources" / source["id"] / f"frame_{frame['index']:06d}.jpg"
            width, height, _metadata = _write_prepared_jpeg(
                input_path,
                output_path,
                normalize_exif=source["media_kind"] == "images",
            )
            name = f"sources/{source['id']}/frame_{frame['index']:06d}.jpg"
            names.append(name)
            images.append(
                _catalog_image(
                    source,
                    frame,
                    name=name,
                    path=_final_relpath(output_path, ctx),
                    width=width,
                    height=height,
                    camera_group_id=group_id,
                    sensor_id="main",
                    projection_name="equirectangular",
                    valid_region={"kind": "full"},
                )
            )
            progress_span.tick(
                frame_number / max(1, len(source["frames"])),
                message=f"{source['label']}: prepared image {frame_number}/{len(source['frames'])}",
                key="log.prepare_source_progress",
                args={"source": source["label"], "cur": frame_number, "tot": len(source["frames"])},
            )
        width, height = images[0]["width"], images[0]["height"]
        return {
            "images": images,
            "camera_groups": [
                _camera_group(
                    group_id,
                    source["id"],
                    "EQUIRECTANGULAR",
                    [float(width), float(height)],
                    names,
                    single_camera=True,
                    single_camera_per_folder=False,
                    refine_intrinsics=False,
                )
            ],
            "rigs": [],
            "outputs": [
                _file_ref(path, ctx, "image/jpeg")
                for path in sorted((ctx.stage_out_dir / "sources" / source["id"]).glob("*.jpg"))
            ],
        }

    def _prepare_equirectangular_pinhole(
        self, ctx: StageContext, source: dict, progress_span: ProgressSpan
    ) -> dict:
        return self._render_pinhole_views(
            ctx,
            source,
            lenses=[{"index": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0}],
            render=lambda frame, view, _lens: rendering.render_perspective_from_equirect(
                _frame_image_path(ctx.project_dir, frame), view
            ),
            progress_span=progress_span,
        )

    def _render_pinhole_views(
        self,
        ctx: StageContext,
        source: dict,
        *,
        lenses: list[dict],
        render,
        progress_span: ProgressSpan,
    ) -> dict:
        views = projection.cubemap_views(size=ctx.params["size"], fov_deg=ctx.params["fov_deg"])
        prefix = f"sources/{source['id']}/"
        group_id = f"{source['id']}:pinhole-rig"
        images = []
        names = []
        outputs = []
        total = len(source["frames"]) * len(views) * len(lenses)
        completed = 0
        for frame in source["frames"]:
            for view in views:
                for lens in lenses:
                    lens_index = lens["index"]
                    output_path = (
                        ctx.stage_out_dir
                        / "sources"
                        / source["id"]
                        / f"frame_{frame['index']:06d}"
                        / f"{view.name}_lens{lens_index}.jpg"
                    )
                    image, _statistics = render(frame, view, lens_index)
                    rendering.write_jpeg(image, output_path, quality=92)
                    name = f"{prefix}{view.name}_lens{lens_index}/frame_{frame['index']:06d}.jpg"
                    names.append(name)
                    images.append(
                        _catalog_image(
                            source,
                            frame,
                            name=name,
                            path=_final_relpath(output_path, ctx),
                            width=view.width,
                            height=view.height,
                            camera_group_id=group_id,
                            sensor_id=f"{view.name}_lens{lens_index}",
                            projection_name="perspective",
                            valid_region={"kind": "full"},
                        )
                    )
                    outputs.append(_file_ref(output_path, ctx, "image/jpeg"))
                    completed += 1
                    progress_span.tick(
                        completed / max(1, total),
                        message=f"{source['label']}: rendered view {completed}/{total}",
                        key="log.prepare_source_progress",
                        args={"source": source["label"], "cur": completed, "tot": total},
                    )
        focal = (ctx.params["size"] / 2.0) / math.tan(math.radians(ctx.params["fov_deg"]) / 2.0)
        params = [focal, focal, ctx.params["size"] / 2.0, ctx.params["size"] / 2.0]
        cameras = colmap_rig.compute_rig_cameras(views, lenses, prefix=prefix)
        return {
            "images": images,
            "camera_groups": [
                _camera_group(
                    group_id,
                    source["id"],
                    "PINHOLE",
                    params,
                    names,
                    single_camera=False,
                    single_camera_per_folder=True,
                    refine_intrinsics=False,
                )
            ],
            "rigs": colmap_rig.build_rig_config(cameras, params),
            "outputs": outputs,
        }

    def _prepare_perspective(self, ctx: StageContext, source: dict, progress_span: ProgressSpan) -> dict:
        groups_by_signature: dict[tuple, dict] = {}
        images = []
        outputs = []
        for frame_number, frame in enumerate(source["frames"], 1):
            input_path = _frame_image_path(ctx.project_dir, frame)
            temporary = ctx.stage_out_dir / "sources" / source["id"] / f"frame_{frame['index']:06d}.jpg"
            width, height, metadata = _write_prepared_jpeg(
                input_path,
                temporary,
                normalize_exif=source["media_kind"] == "images",
            )
            signature = (
                width,
                height,
                metadata.get("make", ""),
                metadata.get("model", ""),
                metadata.get("focal_35mm"),
            )
            group = groups_by_signature.get(signature)
            if group is None:
                group_index = len(groups_by_signature)
                group = {
                    "id": f"{source['id']}:perspective:{group_index}",
                    "folder": f"camera_{group_index:02d}",
                    "width": width,
                    "height": height,
                    "focal_35mm": metadata.get("focal_35mm"),
                    "names": [],
                }
                groups_by_signature[signature] = group
            final_path = temporary.with_name(f"{group['folder']}_{temporary.name}")
            temporary.replace(final_path)
            name = f"sources/{source['id']}/{group['folder']}/frame_{frame['index']:06d}.jpg"
            group["names"].append(name)
            images.append(
                _catalog_image(
                    source,
                    frame,
                    name=name,
                    path=_final_relpath(final_path, ctx),
                    width=width,
                    height=height,
                    camera_group_id=group["id"],
                    sensor_id="main",
                    projection_name="perspective",
                    valid_region={"kind": "full"},
                )
            )
            outputs.append(_file_ref(final_path, ctx, "image/jpeg"))
            progress_span.tick(
                frame_number / max(1, len(source["frames"])),
                message=f"{source['label']}: prepared image {frame_number}/{len(source['frames'])}",
                key="log.prepare_source_progress",
                args={"source": source["label"], "cur": frame_number, "tot": len(source["frames"])},
            )
        camera_groups = []
        for group in groups_by_signature.values():
            diagonal = math.hypot(group["width"], group["height"])
            focal = (
                diagonal * float(group["focal_35mm"]) / FULL_FRAME_DIAGONAL_MM
                if group["focal_35mm"]
                else 1.2 * max(group["width"], group["height"])
            )
            camera_groups.append(
                _camera_group(
                    group["id"],
                    source["id"],
                    "SIMPLE_RADIAL",
                    [focal, group["width"] / 2.0, group["height"] / 2.0, 0.0],
                    group["names"],
                    single_camera=True,
                    single_camera_per_folder=False,
                    refine_intrinsics=True,
                )
            )
        return {"images": images, "camera_groups": camera_groups, "rigs": [], "outputs": outputs}


def _frame_image_path(project_dir: Path, frame: dict) -> Path:
    if "image" in frame:
        return project_dir / frame["image"]
    if "image_source" in frame:
        return Path(frame["image_source"])
    raise KeyError("frame has no image path")


def _load_camera_system(project_dir: Path, inspection: dict) -> CalibratedCameraSystem:
    relative = inspection.get("camera_system_path")
    if not relative:
        raise RuntimeError(
            f"source {inspection['label']}: camera adapter が校正済み camera system を出力していません"
        )
    path = project_dir / "inspect_source" / relative
    return CalibratedCameraSystem.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _quaternion_to_rotation(extrinsic: SensorExtrinsic) -> np.ndarray:
    w, x, y, z = extrinsic.rotation_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("cam_from_rig quaternion の norm が 0 です")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _camera_center(extrinsic: SensorExtrinsic) -> tuple[float, float, float]:
    rotation = _quaternion_to_rotation(extrinsic)
    translation = np.asarray(extrinsic.translation_xyz, dtype=np.float64)
    center = -rotation.T @ translation
    return tuple(float(value) for value in center)


def _write_prepared_jpeg(source: Path, destination: Path, *, normalize_exif: bool) -> tuple[int, int, dict]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        exif = image.getexif()
        metadata = {
            "make": str(exif.get(EXIF_MAKE, "")),
            "model": str(exif.get(EXIF_MODEL, "")),
            "focal_35mm": int(exif[EXIF_FOCAL_35MM]) if exif.get(EXIF_FOCAL_35MM) else None,
        }
        if normalize_exif:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            oriented.save(destination, format="JPEG", quality=95, subsampling=0)
            return oriented.width, oriented.height, metadata
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return image.width, image.height, metadata


def _catalog_image(
    source: dict,
    frame: dict,
    *,
    name: str,
    path: str,
    width: int,
    height: int,
    camera_group_id: str,
    sensor_id: str,
    projection_name: str,
    valid_region: dict,
) -> dict:
    return {
        "name": name,
        "path": path,
        "source_id": source["id"],
        "source_label": source["label"],
        "source_role": source["role"],
        "capture_index": frame["index"],
        "source_index": frame["source_index"],
        "timestamp_sec": frame.get("timestamp_sec"),
        "sensor_id": sensor_id,
        "projection": projection_name,
        "width": width,
        "height": height,
        "camera_group_id": camera_group_id,
        "valid_region": valid_region,
    }


def _camera_group(
    group_id: str,
    source_id: str,
    model: str,
    params: list[float],
    image_names: list[str],
    *,
    single_camera: bool,
    single_camera_per_folder: bool,
    refine_intrinsics: bool,
    rectification: dict | None = None,
) -> dict:
    group = {
        "id": group_id,
        "source_id": source_id,
        "camera_model": model,
        "camera_params": params,
        "single_camera": single_camera,
        "single_camera_per_folder": single_camera_per_folder,
        "refine_intrinsics": refine_intrinsics,
        "image_names": image_names,
    }
    if rectification is not None:
        group["rectification"] = rectification
    return group


def _final_relpath(path: Path, ctx: StageContext) -> str:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_name) / path.relative_to(ctx.stage_out_dir))


def _file_ref(path: Path, ctx: StageContext, mime: str) -> FileRef:
    return FileRef(
        path=_final_relpath(path, ctx),
        size=path.stat().st_size,
        sha256=sha256_file(path) if path.suffix == ".json" else "",
        mime=mime,
    )
