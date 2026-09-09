"""COLMAP camera model に対応する投影と ray geometry。"""

from __future__ import annotations

import numpy as np

from ..colmap.model import Camera
from . import fisheye_camera

SUPPORTED_MODELS = fisheye_camera.SUPPORTED_MODELS | {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}


def pixels_to_camera_rays(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape((-1, 2))
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model in fisheye_camera.SUPPORTED_MODELS:
        return fisheye_camera.pixels_to_camera_rays(model, params, pixels)
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
    raise ValueError(f"unsupported camera projection model: {model}")


def camera_rays_to_pixels(camera: Camera, rays: np.ndarray) -> np.ndarray:
    rays = np.asarray(rays, dtype=np.float64).reshape((-1, 3))
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model in fisheye_camera.SUPPORTED_MODELS:
        return fisheye_camera.camera_rays_to_pixels(model, params, rays)
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
    raise ValueError(f"unsupported camera projection model: {model}")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-12)
