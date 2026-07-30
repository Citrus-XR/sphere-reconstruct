"""RoMaV2 dense correspondence を camera-model aware ray triangulation で追加する。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..colmap.model import Camera, ImagePoint2D, Point3D, Reconstruction
from ..colmap.model import Image as ColmapImage
from .geometry import image_rays_world, pixels_to_camera_rays, project_world_points, triangulate_rays
from .matcher import DenseMatches


class DenseMatcher(Protocol):
    def match(self, image_a: Path, image_b: Path, *, count: int) -> DenseMatches: ...


@dataclass(frozen=True)
class DenseInitializationConfig:
    reference_fraction: float = 0.25
    neighbors_per_reference: int = 2
    matches_per_pair: int = 2000
    confidence_threshold: float = 0.2
    reprojection_threshold_px: float = 1.5
    minimum_parallax_deg: float = 0.5
    maximum_ray_gap_ratio: float = 0.002
    voxel_size_ratio: float = 0.0005
    maximum_new_points: int = 200_000
    use_feature_masks: bool = True
    seed: int = 0


@dataclass(frozen=True)
class DenseInitializationResult:
    pairs_considered: int
    pairs_processed: int
    raw_points: int
    kept_points: int
    rejected_confidence: int
    rejected_masks: int
    rejected_geometry: int
    scene_scale: float
    voxel_size: float


@dataclass(frozen=True)
class _View:
    image: ColmapImage
    camera: Camera
    image_path: Path
    mask_path: Path | None
    source_id: str
    sensor_id: str
    capture_index: int


def densify_reconstruction(
    reconstruction: Reconstruction,
    image_catalog: dict,
    image_root: Path,
    mask_root: Path | None,
    matcher: DenseMatcher,
    config: DenseInitializationConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DenseInitializationResult:
    if not 0 < config.reference_fraction <= 1:
        raise ValueError("dense reference_fraction must be within (0, 1]")
    if config.maximum_new_points <= 0:
        raise ValueError("dense maximum_new_points must be positive")
    views = _build_views(reconstruction, image_catalog, image_root, mask_root, config.use_feature_masks)
    pairs = _select_pairs(views, config.reference_fraction, config.neighbors_per_reference)
    scene_scale = _scene_scale(reconstruction)
    maximum_ray_gap = scene_scale * config.maximum_ray_gap_ratio
    candidates: list[tuple[np.ndarray, np.ndarray, float, list[tuple[int, float, float]]]] = []
    rejected_confidence = 0
    rejected_masks = 0
    rejected_geometry = 0
    processed = 0
    rng = np.random.default_rng(config.seed)

    for pair_number, (first, second) in enumerate(pairs, 1):
        matches = matcher.match(first.image_path, second.image_path, count=config.matches_per_pair)
        valid = np.isfinite(matches.pixels_a).all(axis=1) & np.isfinite(matches.pixels_b).all(axis=1)
        confident = matches.confidence >= config.confidence_threshold
        rejected_confidence += int(np.count_nonzero(valid & ~confident))
        valid &= confident
        valid &= _inside_image(matches.pixels_a, first.camera) & _inside_image(matches.pixels_b, second.camera)
        if first.mask_path is not None:
            mask_valid = _sample_mask(first.mask_path, matches.pixels_a)
            rejected_masks += int(np.count_nonzero(valid & ~mask_valid))
            valid &= mask_valid
        if second.mask_path is not None:
            mask_valid = _sample_mask(second.mask_path, matches.pixels_b)
            rejected_masks += int(np.count_nonzero(valid & ~mask_valid))
            valid &= mask_valid
        if not np.any(valid):
            if progress is not None:
                progress(pair_number, len(pairs))
            continue

        pixels_a = matches.pixels_a[valid]
        pixels_b = matches.pixels_b[valid]
        origins_a, directions_a = image_rays_world(
            first.image, pixels_to_camera_rays(first.camera, pixels_a)
        )
        origins_b, directions_b = image_rays_world(
            second.image, pixels_to_camera_rays(second.camera, pixels_b)
        )
        points, depth_a, depth_b, geometry = triangulate_rays(
            origins_a, directions_a, origins_b, directions_b
        )
        ray_gap, parallax = geometry[:, 0], geometry[:, 1]
        projected_a, visible_a = project_world_points(first.camera, first.image, points)
        projected_b, visible_b = project_world_points(second.camera, second.image, points)
        error_a = np.linalg.norm(projected_a - pixels_a, axis=1)
        error_b = np.linalg.norm(projected_b - pixels_b, axis=1)
        errors = np.maximum(error_a, error_b)
        geometry_valid = (
            np.isfinite(points).all(axis=1)
            & (depth_a > 0)
            & (depth_b > 0)
            & (visible_a > 0)
            & (visible_b > 0)
            & (parallax >= config.minimum_parallax_deg)
            & (ray_gap <= maximum_ray_gap)
            & (errors <= config.reprojection_threshold_px)
        )
        rejected_geometry += int(np.count_nonzero(~geometry_valid))
        if np.any(geometry_valid):
            colors = _sample_rgb(first.image_path, pixels_a[geometry_valid])
            for point, color, error, pixel_a, pixel_b in zip(
                points[geometry_valid],
                colors,
                errors[geometry_valid],
                pixels_a[geometry_valid],
                pixels_b[geometry_valid],
                strict=True,
            ):
                candidates.append(
                    (
                        point,
                        color,
                        float(error),
                        [
                            (first.image.image_id, float(pixel_a[0]), float(pixel_a[1])),
                            (second.image.image_id, float(pixel_b[0]), float(pixel_b[1])),
                        ],
                    )
                )
        processed += 1
        if len(candidates) >= config.maximum_new_points * 2:
            break
        if progress is not None:
            progress(pair_number, len(pairs))

    voxel_size = max(scene_scale * config.voxel_size_ratio, 1e-9)
    selected = _voxel_select(candidates, voxel_size)
    if len(selected) > config.maximum_new_points:
        indices = np.sort(rng.choice(len(selected), config.maximum_new_points, replace=False))
        selected = [selected[index] for index in indices]
    _append_points(reconstruction, selected)
    return DenseInitializationResult(
        pairs_considered=len(pairs),
        pairs_processed=processed,
        raw_points=len(candidates),
        kept_points=len(selected),
        rejected_confidence=rejected_confidence,
        rejected_masks=rejected_masks,
        rejected_geometry=rejected_geometry,
        scene_scale=scene_scale,
        voxel_size=voxel_size,
    )


def _build_views(
    reconstruction: Reconstruction,
    catalog: dict,
    image_root: Path,
    mask_root: Path | None,
    use_masks: bool,
) -> list[_View]:
    records = {str(record["name"]).replace("\\", "/"): record for record in catalog["images"]}
    views = []
    for image in reconstruction.images.values():
        name = image.name.replace("\\", "/")
        record = records.get(name)
        if record is None:
            continue
        camera = reconstruction.cameras[image.camera_id]
        image_path = image_root / Path(name)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        candidate_mask = mask_root / f"{name}.png" if use_masks and mask_root is not None else None
        views.append(
            _View(
                image=image,
                camera=camera,
                image_path=image_path,
                mask_path=candidate_mask if candidate_mask is not None and candidate_mask.is_file() else None,
                source_id=str(record["source_id"]),
                sensor_id=str(record.get("sensor_id", image.camera_id)),
                capture_index=int(record["capture_index"]),
            )
        )
    return views


def _select_pairs(views: list[_View], fraction: float, neighbor_count: int) -> list[tuple[_View, _View]]:
    groups: dict[tuple[str, str, int], list[_View]] = defaultdict(list)
    for view in views:
        groups[(view.source_id, view.sensor_id, view.image.camera_id)].append(view)
    pairs: dict[tuple[int, int], tuple[_View, _View]] = {}
    for group in groups.values():
        group.sort(key=lambda item: item.capture_index)
        reference_count = max(1, round(len(group) * fraction))
        reference_indices = np.unique(np.linspace(0, len(group) - 1, reference_count).round().astype(int))
        centers = np.asarray([view.image.camera_center for view in group], dtype=np.float64)
        for reference_index in reference_indices:
            distances = np.linalg.norm(centers - centers[reference_index], axis=1)
            distances[reference_index] = np.inf
            for neighbor_index in np.argsort(distances)[: max(1, neighbor_count)]:
                first, second = group[reference_index], group[int(neighbor_index)]
                key = tuple(sorted((first.image.image_id, second.image.image_id)))
                pairs[key] = (first, second)
    return [pairs[key] for key in sorted(pairs)]


def _scene_scale(reconstruction: Reconstruction) -> float:
    if not reconstruction.points3D:
        raise ValueError("dense initialization requires sparse points")
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
    center = np.median(points, axis=0)
    return float(np.median(np.linalg.norm(points - center, axis=1)))


def _inside_image(pixels: np.ndarray, camera: Camera) -> np.ndarray:
    return (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )


def _sample_mask(path: Path, pixels: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x] >= 128


def _sample_rgb(path: Path, pixels: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, rgb.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, rgb.shape[0] - 1)
    return rgb[y, x]


def _voxel_select(candidates, voxel_size: float):
    selected = {}
    for candidate in candidates:
        key = tuple(np.floor(candidate[0] / voxel_size).astype(np.int64))
        previous = selected.get(key)
        if previous is None or candidate[2] < previous[2]:
            selected[key] = candidate
    return list(selected.values())


def _append_points(reconstruction: Reconstruction, candidates) -> None:
    next_point_id = max(reconstruction.points3D, default=0) + 1
    for point, color, error, observations in candidates:
        track = []
        for image_id, x, y in observations:
            image = reconstruction.images[image_id]
            point2d_index = len(image.points2D)
            image.points2D.append(ImagePoint2D(x=x, y=y, point3D_id=next_point_id))
            track.append((image_id, point2d_index))
        reconstruction.points3D[next_point_id] = Point3D(
            point3D_id=next_point_id,
            xyz=tuple(float(value) for value in point),
            rgb=tuple(int(value) for value in color),
            error=error,
            track=track,
        )
        next_point_id += 1
