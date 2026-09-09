"""Dense match の ray を world geometry と二視点三角化に変換する。"""

from __future__ import annotations

import numpy as np

from ..colmap.model import Camera, Image, qvec_to_rotation
from ..imaging.camera_geometry import camera_rays_to_pixels


def image_rays_world(image: Image, camera_rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(qvec_to_rotation(image.qvec), dtype=np.float64)
    directions = _normalize(np.asarray(camera_rays, dtype=np.float64) @ rotation)
    origins = np.repeat(np.asarray(image.camera_center, dtype=np.float64)[None, :], len(directions), axis=0)
    return origins, directions


def triangulate_rays(
    origins_a: np.ndarray,
    directions_a: np.ndarray,
    origins_b: np.ndarray,
    directions_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origins_a = np.asarray(origins_a, dtype=np.float64)
    origins_b = np.asarray(origins_b, dtype=np.float64)
    directions_a = _normalize(directions_a)
    directions_b = _normalize(directions_b)
    offset = origins_a - origins_b
    dot = np.sum(directions_a * directions_b, axis=1)
    first = np.sum(directions_a * offset, axis=1)
    second = np.sum(directions_b * offset, axis=1)
    denominator = 1 - dot * dot
    depth_a = np.divide(dot * second - first, denominator, out=np.full_like(dot, np.nan), where=denominator > 1e-10)
    depth_b = np.divide(second - dot * first, denominator, out=np.full_like(dot, np.nan), where=denominator > 1e-10)
    point_a = origins_a + depth_a[:, None] * directions_a
    point_b = origins_b + depth_b[:, None] * directions_b
    points = (point_a + point_b) * 0.5
    ray_gap = np.linalg.norm(point_a - point_b, axis=1)
    parallax = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return points, depth_a, depth_b, np.column_stack((ray_gap, parallax))


def project_world_points(camera: Camera, image: Image, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(qvec_to_rotation(image.qvec), dtype=np.float64)
    camera_points = np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(image.tvec)
    return camera_rays_to_pixels(camera, camera_points), camera_points[:, 2]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-12)
