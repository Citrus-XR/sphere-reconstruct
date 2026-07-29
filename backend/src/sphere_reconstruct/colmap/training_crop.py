"""LFStudio 学習用 native fisheye の無効な外周を lossless crop する。

Camera-model valid region を含む JPEG MCU 境界へ外向きに丸めるため、有効 pixel は一つも捨てない。
画像 crop と同時に camera 主点および images.bin の 2D 観測を平行移動し、COLMAP dataset としての
整合性を維持する。JPEG は再圧縮せず jpegtran の係数領域 transform を使う。
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image as PilImage

from ..imaging import valid_region
from .model import Camera, Image, ImagePoint2D, Reconstruction

_PRINCIPAL_POINT_INDICES = {
    "SIMPLE_PINHOLE": (1, 2),
    "PINHOLE": (2, 3),
    "SIMPLE_RADIAL": (1, 2),
    "RADIAL": (1, 2),
    "OPENCV": (2, 3),
    "OPENCV_FISHEYE": (2, 3),
    "FULL_OPENCV": (2, 3),
    "SIMPLE_RADIAL_FISHEYE": (1, 2),
    "RADIAL_FISHEYE": (1, 2),
    "THIN_PRISM_FISHEYE": (2, 3),
    "RAD_TAN_THIN_PRISM_FISHEYE": (2, 3),
    "SIMPLE_DIVISION": (1, 2),
    "DIVISION": (1, 2),
    "SIMPLE_FISHEYE": (1, 2),
    "FISHEYE": (1, 2),
    "EUCM": (2, 3),
}


@dataclass(frozen=True)
class CropRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class CropPlan:
    images: dict[str, CropRect]
    cameras: dict[int, CropRect]
    source_sizes: dict[str, tuple[int, int]]
    source_pixels: int
    cropped_pixels: int
    alignment_px: int

    @property
    def changed(self) -> bool:
        return self.cropped_pixels < self.source_pixels


def resolve_jpegtran(explicit: str = "") -> str | None:
    if explicit:
        candidate = Path(explicit)
        return str(candidate) if candidate.is_file() else None
    return shutil.which("jpegtran")


def build_plan(
    reconstruction: Reconstruction,
    catalog: dict,
    *,
    alignment_px: int = 16,
) -> CropPlan:
    if alignment_px <= 0:
        raise ValueError("alignment_px must be positive")
    records = {record["name"]: record for record in catalog["images"]}
    image_rects: dict[str, CropRect] = {}
    camera_rects: dict[int, CropRect] = {}
    camera_region_keys: dict[int, str] = {}
    source_sizes: dict[str, tuple[int, int]] = {}
    source_pixels = 0
    cropped_pixels = 0
    for image in reconstruction.images.values():
        try:
            record = records[image.name]
        except KeyError as error:
            raise ValueError(f"image catalog is missing registered image: {image.name}") from error
        width, height = int(record["width"]), int(record["height"])
        camera = reconstruction.cameras[image.camera_id]
        if (width, height) != (camera.width, camera.height):
            raise ValueError(
                f"catalog/model dimensions differ for {image.name}: "
                f"{width}x{height} != {camera.width}x{camera.height}"
            )
        region_key = valid_region.cache_key(record["valid_region"], width, height)
        previous_key = camera_region_keys.setdefault(image.camera_id, region_key)
        if previous_key != region_key:
            raise ValueError(f"camera {image.camera_id} has inconsistent valid regions")
        rect = camera_rects.get(image.camera_id)
        if rect is None:
            rect = _record_rect(record, alignment_px)
            camera_rects[image.camera_id] = rect
        image_rects[image.name] = rect
        source_sizes[image.name] = (width, height)
        source_pixels += width * height
        cropped_pixels += rect.pixels
    return CropPlan(
        image_rects,
        camera_rects,
        source_sizes,
        source_pixels,
        cropped_pixels,
        alignment_px,
    )


def crop_reconstruction(reconstruction: Reconstruction, plan: CropPlan) -> Reconstruction:
    cameras: dict[int, Camera] = {}
    for camera_id, camera in reconstruction.cameras.items():
        rect = plan.cameras.get(camera_id, CropRect(0, 0, camera.width, camera.height))
        params = list(camera.params)
        if rect.left or rect.top:
            try:
                x_index, y_index = _PRINCIPAL_POINT_INDICES[camera.model]
            except KeyError as error:
                raise ValueError(f"camera model cannot be cropped safely: {camera.model}") from error
            params[x_index] -= rect.left
            params[y_index] -= rect.top
        cameras[camera_id] = replace(camera, width=rect.width, height=rect.height, params=params)

    images: dict[int, Image] = {}
    for image_id, image in reconstruction.images.items():
        rect = plan.images[image.name]
        points = [
            ImagePoint2D(point.x - rect.left, point.y - rect.top, point.point3D_id)
            for point in image.points2D
        ]
        images[image_id] = replace(image, points2D=points)
    return Reconstruction(cameras=cameras, images=images, points3D=reconstruction.points3D)


def crop_images(
    source_dir: Path,
    destination_dir: Path,
    plan: CropPlan,
    jpegtran: str,
    *,
    progress=None,
) -> dict:
    source_bytes = 0
    cropped_bytes = 0
    names = sorted(plan.images)
    workers = min(16, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_crop_image, source_dir / name, destination_dir / name, plan.images[name], jpegtran): name
            for name in names
        }
        for index, future in enumerate(as_completed(futures), 1):
            before, after = future.result()
            source_bytes += before
            cropped_bytes += after
            if progress is not None:
                progress(index, len(names))
    return {"source_bytes": source_bytes, "cropped_bytes": cropped_bytes}


def crop_masks(
    source_dir: Path,
    destination_dir: Path,
    plan: CropPlan,
    *,
    progress=None,
) -> int:
    names = [name for name in sorted(plan.images) if (source_dir / f"{name}.png").is_file()]
    workers = min(16, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                crop_mask_file,
                source_dir / f"{name}.png",
                destination_dir / f"{name}.png",
                plan.images[name],
                plan.source_sizes[name],
            )
            for name in names
        ]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if progress is not None:
                progress(index, len(names))
    return len(names)


def _record_rect(record: dict, alignment: int) -> CropRect:
    width, height = int(record["width"]), int(record["height"])
    region = record.get("valid_region", {"kind": "full"})
    if region.get("kind") == "full":
        return CropRect(0, 0, width, height)
    valid_left, valid_top, valid_right, valid_bottom = valid_region.bounding_box(
        region, width, height
    )
    left = max(0, math.floor(valid_left / alignment) * alignment)
    top = max(0, math.floor(valid_top / alignment) * alignment)
    right = min(width, math.ceil(valid_right / alignment) * alignment)
    bottom = min(height, math.ceil(valid_bottom / alignment) * alignment)
    if right <= left or bottom <= top:
        raise ValueError(f"invalid fisheye crop rectangle for {record['name']}")
    return CropRect(left, top, right, bottom)


def _crop_image(source: Path, destination: Path, rect: CropRect, jpegtran: str) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    with PilImage.open(source) as image:
        full_image = rect == CropRect(0, 0, *image.size)
    if full_image:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return source_size, destination.stat().st_size
    if source.suffix.lower() == ".png":
        with PilImage.open(source) as image:
            image.crop((rect.left, rect.top, rect.right, rect.bottom)).save(
                destination,
                format="PNG",
                compress_level=6,
            )
        return source_size, destination.stat().st_size
    if source.suffix.lower() not in {".jpg", ".jpeg"}:
        raise RuntimeError(f"lossless fisheye crop does not support image format: {source.suffix}")
    command = [
        jpegtran,
        "-copy",
        "all",
        "-perfect",
        "-crop",
        f"{rect.width}x{rect.height}+{rect.left}+{rect.top}",
        "-outfile",
        str(destination),
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"jpegtran crop failed for {source}: {completed.stderr.strip()}")
    return source_size, destination.stat().st_size


def crop_mask_file(
    source: Path,
    destination: Path,
    rect: CropRect,
    expected_size: tuple[int, int],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with PilImage.open(source) as image:
        if image.size != expected_size:
            raise RuntimeError(
                f"mask dimensions changed for {source}: "
                f"{image.width}x{image.height} != {expected_size[0]}x{expected_size[1]}"
            )
        full_image = rect == CropRect(0, 0, *image.size)
        if not full_image:
            image.crop((rect.left, rect.top, rect.right, rect.bottom)).save(
                destination,
                format="PNG",
                compress_level=6,
            )
    if full_image:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
