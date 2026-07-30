"""Dense match を COLMAP camera model の ray と 3D point へ変換する。"""

from __future__ import annotations

import numpy as np

from ..colmap.model import Camera, Image, qvec_to_rotation


def pixels_to_camera_rays(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape((-1, 2))
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        distorted = np.column_stack(((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy))
        theta_distorted = np.linalg.norm(distorted, axis=1)
        theta = theta_distorted.copy()
        for _ in range(10):
            theta2 = theta * theta
            radial = 1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
            derivative = 1 + 3 * k1 * theta2 + 5 * k2 * theta2**2 + 7 * k3 * theta2**3 + 9 * k4 * theta2**4
            theta -= np.divide(
                theta * radial - theta_distorted,
                derivative,
                out=np.zeros_like(theta),
                where=np.abs(derivative) > 1e-12,
            )
        azimuth = np.arctan2(distorted[:, 1], distorted[:, 0])
        sin_theta = np.sin(theta)
        return np.column_stack(
            (sin_theta * np.cos(azimuth), sin_theta * np.sin(azimuth), np.cos(theta))
        )
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        else:
            fx = fy = params[0]
            cx, cy = params[1:3]
        rays = np.column_stack(((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy, np.ones(len(pixels))))
        return _normalize(rays)
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        focal, cx, cy = params[:3]
        k1 = params[3]
        k2 = params[4] if model == "RADIAL" else 0.0
        distorted = np.column_stack(((pixels[:, 0] - cx) / focal, (pixels[:, 1] - cy) / focal))
        radius_distorted = np.linalg.norm(distorted, axis=1)
        radius = radius_distorted.copy()
        for _ in range(10):
            radius2 = radius * radius
            radial = 1 + k1 * radius2 + k2 * radius2**2
            derivative = 1 + 3 * k1 * radius2 + 5 * k2 * radius2**2
            radius -= np.divide(
                radius * radial - radius_distorted,
                derivative,
                out=np.zeros_like(radius),
                where=np.abs(derivative) > 1e-12,
            )
        scale = np.divide(radius, radius_distorted, out=np.ones_like(radius), where=radius_distorted > 1e-12)
        rays = np.column_stack((distorted * scale[:, None], np.ones(len(pixels))))
        return _normalize(rays)
    raise ValueError(f"dense initialization does not support camera model: {model}")


def camera_rays_to_pixels(camera: Camera, rays: np.ndarray) -> np.ndarray:
    rays = np.asarray(rays, dtype=np.float64).reshape((-1, 3))
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        radius = np.linalg.norm(rays[:, :2], axis=1)
        theta = np.arctan2(radius, rays[:, 2])
        theta2 = theta * theta
        theta_distorted = theta * (
            1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )
        scale = np.divide(theta_distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        return np.column_stack((fx * rays[:, 0] * scale + cx, fy * rays[:, 1] * scale + cy))
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        if model == "PINHOLE":
            fx, fy, cx, cy = params
        else:
            fx = fy = params[0]
            cx, cy = params[1:3]
        normalized = rays[:, :2] / rays[:, 2:3]
        return np.column_stack((fx * normalized[:, 0] + cx, fy * normalized[:, 1] + cy))
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        focal, cx, cy = params[:3]
        k1 = params[3]
        k2 = params[4] if model == "RADIAL" else 0.0
        normalized = rays[:, :2] / rays[:, 2:3]
        radius2 = np.sum(normalized * normalized, axis=1)
        radial = 1 + k1 * radius2 + k2 * radius2**2
        return np.column_stack(
            (focal * normalized[:, 0] * radial + cx, focal * normalized[:, 1] * radial + cy)
        )
    raise ValueError(f"dense initialization does not support camera model: {model}")


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
