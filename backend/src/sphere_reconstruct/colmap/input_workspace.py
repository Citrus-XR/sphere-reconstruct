"""COLMAP 特徴抽出用 workspace の構築.

入力カメラ固有の処理を特徴抽出・matching・mapper から分離し, 後から別のカメラを追加する
ときは builder を 1 つ追加すればよい構造にする. 出力される ``input_spec.json`` が後段との
契約であり, 後段は INSV や ERP の元 manifest を再解釈しない.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..imaging import fisheye_region
from . import native_rig


@dataclass(frozen=True)
class InputSpec:
    version: int
    reconstruction_mode: str
    image_count: int
    width: int
    height: int
    camera_model: str
    camera_params: list[float]
    single_camera: bool
    single_camera_per_folder: bool
    image_path: str
    mask_path: str | None
    rig_config_path: str | None
    refine_intrinsics: bool
    refine_rig: bool
    multiple_models: bool

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> InputSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


Builder = Callable[[Path, Path, bool], InputSpec]
_BUILDERS: dict[str, Builder] = {}


def register_builder(mode: str):
    def register(builder: Builder) -> Builder:
        _BUILDERS[mode] = builder
        return builder

    return register


def build(project_dir: Path, output_dir: Path, mode: str, use_masks: bool) -> InputSpec:
    try:
        builder = _BUILDERS[mode]
    except KeyError as error:
        raise ValueError(f"unsupported reconstruction mode: {mode}") from error
    spec = builder(project_dir, output_dir, use_masks)
    spec.write(output_dir / "input_spec.json")
    return spec


@register_builder("native_fisheye")
def _build_native_fisheye(project_dir: Path, output_dir: Path, use_masks: bool) -> InputSpec:
    frames_path = project_dir / "extract_frames" / "manifest_frames.json"
    source_path = project_dir / "inspect_source" / "source.json"
    if not frames_path.exists():
        raise RuntimeError("extract_frames must run before feature extraction")
    frames_manifest = json.loads(frames_path.read_text(encoding="utf-8"))
    if frames_manifest.get("kind") != "insv_dual":
        raise RuntimeError("native_fisheye requires dual-lens INSV frames")

    width = int(frames_manifest["width"])
    height = int(frames_manifest["height"])
    frames = frames_manifest["frames"]
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    for frame in frames:
        filename = f"frame_{frame['index']:06d}.jpg"
        for source_key, prefix in (("lens0", "front"), ("lens1", "back")):
            source = project_dir / frame[source_key]
            destination = images_dir / prefix / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(source, destination)

    focal = width / 2.0 / 1.75
    camera_params = [focal, focal, width / 2.0, height / 2.0, 0.0, 0.0, 0.0, 0.0]
    region = fisheye_region.load_region(project_dir)
    generated_masks = _native_generated_masks(project_dir) if use_masks else {}
    for frame in frames:
        filename = f"frame_{frame['index']:06d}.jpg"
        for lens, prefix in ((0, "front"), (1, "back")):
            destination = masks_dir / prefix / f"{filename}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            generated = generated_masks.get((lens, frame["index"]))
            if generated is not None:
                _link_or_copy(generated, destination)
            else:
                center_x, center_y, radius = fisheye_region.circle_px(region[f"lens{lens}"], width, height)
                _write_circle_mask(width, height, center_x, center_y, radius, destination)

    baseline = native_rig.DEFAULT_BASELINE_M
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        calibration = source.get("offset_v3") or {}
        if calibration.get("valid") and calibration.get("lenses"):
            baseline = native_rig.baseline_from_lens_centers(calibration["lenses"])
    rig_config = native_rig.build_physical_rig_config("OPENCV_FISHEYE", camera_params, baseline)
    rig_config_path = output_dir / "rig_config.json"
    rig_config_path.write_text(json.dumps(rig_config, ensure_ascii=False, indent=2), encoding="utf-8")

    return InputSpec(
        version=1,
        reconstruction_mode="native_fisheye",
        image_count=len(frames) * 2,
        width=width,
        height=height,
        camera_model="OPENCV_FISHEYE",
        camera_params=camera_params,
        single_camera=False,
        single_camera_per_folder=True,
        image_path="images",
        mask_path="masks",
        rig_config_path="rig_config.json",
        refine_intrinsics=True,
        refine_rig=False,
        multiple_models=False,
    )


def _native_generated_masks(project_dir: Path) -> dict[tuple[int, int], Path]:
    manifest_path = project_dir / "generate_masks" / "manifest_masks.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "sam3_fisheye_masks":
        return {}
    return {
        (lens["lens"], frame["index"]): project_dir / lens["path"]
        for frame in manifest["frames"]
        for lens in frame["lenses"]
    }


@register_builder("equirectangular")
def _build_equirectangular(project_dir: Path, output_dir: Path, use_masks: bool) -> InputSpec:
    frames_path = project_dir / "extract_frames" / "manifest_frames.json"
    if not frames_path.exists():
        raise RuntimeError("extract_frames must run before feature extraction")
    manifest = json.loads(frames_path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in {"erp_video", "erp_images"}:
        raise RuntimeError("equirectangular mode requires ERP frames")
    frames = manifest.get("frames", [])
    if not frames:
        raise RuntimeError("extract_frames produced no ERP frames")

    first_source = _erp_source(project_dir, frames[0])
    if manifest.get("width") and manifest.get("height"):
        width, height = int(manifest["width"]), int(manifest["height"])
    else:
        import cv2  # noqa: PLC0415

        image = cv2.imread(str(first_source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read ERP image: {first_source}")
        height, width = image.shape[:2]

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    source_masks = _erp_generated_masks(project_dir) if use_masks else {}
    have_masks = False
    for frame in frames:
        source = _erp_source(project_dir, frame)
        filename = f"frame_{frame['index']:06d}{source.suffix.lower()}"
        _link_or_copy(source, images_dir / filename)
        generated = source_masks.get(frame["index"])
        if generated is not None:
            destination = output_dir / "masks" / f"{filename}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(generated, destination)
            have_masks = True

    return InputSpec(
        version=1,
        reconstruction_mode="equirectangular",
        image_count=len(frames),
        width=width,
        height=height,
        camera_model="EQUIRECTANGULAR",
        camera_params=[float(width), float(height)],
        single_camera=True,
        single_camera_per_folder=False,
        image_path="images",
        mask_path="masks" if have_masks else None,
        rig_config_path=None,
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )


def _erp_source(project_dir: Path, frame: dict) -> Path:
    return project_dir / frame["erp"] if "erp" in frame else Path(frame["erp_source"])


def _erp_generated_masks(project_dir: Path) -> dict[int, Path]:
    manifest_path = project_dir / "generate_masks" / "manifest_masks.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "sam3_erp_masks":
        return {}
    return {frame["index"]: project_dir / frame["path"] for frame in manifest["frames"]}


@register_builder("pinhole_rig")
def _build_pinhole_rig(project_dir: Path, output_dir: Path, use_masks: bool) -> InputSpec:
    rig_manifest_path = project_dir / "reproject_views" / "manifest_rig.json"
    if not rig_manifest_path.exists():
        raise RuntimeError("reproject_views must run before pinhole feature extraction")
    manifest = json.loads(rig_manifest_path.read_text(encoding="utf-8"))
    views = manifest.get("views", [])
    lenses = manifest.get("lenses", [])
    if not views or not lenses:
        raise RuntimeError("reprojected rig metadata is empty")

    images_dir = output_dir / "images"
    for frame in manifest["frames"]:
        for view in frame["views"]:
            relative = Path(f"{view['view']}_lens{view['lens']}") / f"frame_{frame['index']:06d}.jpg"
            destination = images_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(project_dir / view["path"], destination)

    have_masks = False
    masks_manifest_path = project_dir / "generate_masks" / "manifest_masks.json"
    if use_masks and masks_manifest_path.exists():
        masks_manifest = json.loads(masks_manifest_path.read_text(encoding="utf-8"))
        if masks_manifest.get("kind") == "sam3_pinhole_masks":
            for frame in masks_manifest["frames"]:
                for view in frame["views"]:
                    relative = (
                        Path(f"{view['view']}_lens{view['lens']}") / f"frame_{frame['index']:06d}.jpg.png"
                    )
                    destination = output_dir / "masks" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(project_dir / view["path"], destination)
                    have_masks = True

    size = int(views[0]["size"])
    field_of_view = float(views[0]["fov_deg"])
    focal = (size / 2.0) / math.tan(math.radians(field_of_view) / 2.0)
    camera_params = [focal, focal, size / 2.0, size / 2.0]

    from . import rig as colmap_rig  # noqa: PLC0415

    cameras = colmap_rig.compute_rig_cameras(views, lenses)
    rig_config = colmap_rig.build_rig_config(cameras, camera_params)
    rig_config_path = output_dir / "rig_config.json"
    rig_config_path.write_text(json.dumps(rig_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return InputSpec(
        version=1,
        reconstruction_mode="pinhole_rig",
        image_count=sum(len(frame["views"]) for frame in manifest["frames"]),
        width=size,
        height=size,
        camera_model="PINHOLE",
        camera_params=camera_params,
        single_camera=False,
        single_camera_per_folder=True,
        image_path="images",
        mask_path="masks" if have_masks else None,
        rig_config_path="rig_config.json",
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )


def _write_circle_mask(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    destination: Path,
) -> None:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(
        mask,
        (round(center_x), round(center_y)),
        round(radius),
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), mask):
        raise RuntimeError(f"failed to write mask: {destination}")


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
