"""Vectorized COLMAP fisheye camera projection shared by masks and dense matching."""

from __future__ import annotations

import numpy as np

SUPPORTED_MODELS = frozenset({"OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"})


def pixels_to_camera_rays(model: str, params, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape((-1, 2))
    fx, fy, cx, cy, radial, tangential = _split_parameters(model, params)
    distorted = np.column_stack(((pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy))
    if model == "THIN_PRISM_FISHEYE":
        equidistant = _remove_thin_prism_distortion(distorted, radial, tangential)
        theta = np.linalg.norm(equidistant, axis=1)
        theta_distorted = theta
        direction = equidistant
    else:
        theta_distorted = np.linalg.norm(distorted, axis=1)
        theta = _invert_radial(theta_distorted, radial)
        direction = distorted
    scale = np.divide(
        np.sin(theta),
        theta_distorted,
        out=np.ones_like(theta),
        where=theta_distorted > 1e-12,
    )
    return np.column_stack(
        (direction[:, 0] * scale, direction[:, 1] * scale, np.cos(theta))
    )


def camera_rays_to_pixels(model: str, params, rays: np.ndarray) -> np.ndarray:
    rays = np.asarray(rays, dtype=np.float64).reshape((-1, 3))
    fx, fy, cx, cy, radial, tangential = _split_parameters(model, params)
    radius = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(radius, rays[:, 2])
    equidistant_scale = np.divide(
        theta,
        radius,
        out=np.ones_like(radius),
        where=radius > 1e-12,
    )
    equidistant = rays[:, :2] * equidistant_scale[:, None]
    if model == "THIN_PRISM_FISHEYE":
        distorted = _apply_thin_prism_distortion(equidistant, radial, tangential)
    else:
        theta_distorted = _apply_radial(theta, radial)
        radial_scale = np.divide(
            theta_distorted,
            theta,
            out=np.ones_like(theta),
            where=theta > 1e-12,
        )
        distorted = equidistant * radial_scale[:, None]
    return np.column_stack((fx * distorted[:, 0] + cx, fy * distorted[:, 1] + cy))


def pixels_to_theta(model: str, params, pixels: np.ndarray) -> np.ndarray:
    rays = pixels_to_camera_rays(model, params, pixels)
    return np.arccos(np.clip(rays[:, 2], -1.0, 1.0))


def validate_forward_hemisphere(model: str, params, maximum_theta: float) -> None:
    _fx, _fy, _cx, _cy, radial, _tangential = _split_parameters(model, params)
    k1, k2, k3, k4 = radial
    derivative_coefficients = np.asarray(
        [1.0, 3.0 * k1, 5.0 * k2, 7.0 * k3, 9.0 * k4]
    )
    stationary_coefficients = np.asarray([3.0 * k1, 10.0 * k2, 21.0 * k3, 36.0 * k4])
    nonzero = np.flatnonzero(stationary_coefficients)
    roots = (
        np.polynomial.polynomial.polyroots(stationary_coefficients[: nonzero[-1] + 1])
        if len(nonzero)
        else np.asarray([], dtype=np.complex128)
    )
    maximum_x = maximum_theta * maximum_theta
    candidates = [0.0, maximum_x]
    candidates.extend(
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1e-10 * (1.0 + abs(root.real))
        and 0.0 < root.real < maximum_x
    )
    derivative_values = np.polynomial.polynomial.polyval(candidates, derivative_coefficients)
    if np.any(derivative_values <= 1e-10):
        raise ValueError("fisheye radial distortion is not monotonic within the valid hemisphere")
    distorted = _apply_radial(np.asarray([maximum_theta]), radial)[0]
    if not np.isfinite(distorted) or distorted <= 0.0:
        raise ValueError(f"invalid fisheye distorted radius: {distorted}")


def _split_parameters(model: str, params):
    values = np.asarray(params, dtype=np.float64)
    if model == "OPENCV_FISHEYE":
        if len(values) != 8:
            raise ValueError(f"OPENCV_FISHEYE requires 8 parameters, got {len(values)}")
        fx, fy, cx, cy, k1, k2, k3, k4 = values
        tangential = None
    elif model == "THIN_PRISM_FISHEYE":
        if len(values) != 12:
            raise ValueError(f"THIN_PRISM_FISHEYE requires 12 parameters, got {len(values)}")
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = values
        tangential = (p1, p2, sx1, sy1)
    else:
        raise ValueError(f"unsupported fisheye camera model: {model}")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"fisheye focal length must be positive: {(fx, fy)}")
    return fx, fy, cx, cy, (k1, k2, k3, k4), tangential


def _apply_radial(theta: np.ndarray, radial) -> np.ndarray:
    k1, k2, k3, k4 = radial
    theta2 = theta * theta
    return theta * (
        1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
    )


def _invert_radial(theta_distorted: np.ndarray, radial) -> np.ndarray:
    k1, k2, k3, k4 = radial
    theta = np.clip(theta_distorted, 0.0, np.pi / 2).copy()
    for _ in range(15):
        theta2 = theta * theta
        radial_scale = 1.0 + theta2 * (
            k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))
        )
        derivative = 1.0 + theta2 * (
            3.0 * k1
            + theta2 * (5.0 * k2 + theta2 * (7.0 * k3 + theta2 * 9.0 * k4))
        )
        theta -= np.divide(
            theta * radial_scale - theta_distorted,
            derivative,
            out=np.zeros_like(theta),
            where=np.abs(derivative) > 1e-12,
        )
        np.clip(theta, 0.0, np.pi / 2, out=theta)
    return theta


def _apply_non_radial_delta(points: np.ndarray, coefficients) -> np.ndarray:
    if coefficients is None:
        return np.zeros_like(points)
    p1, p2, sx1, sy1 = coefficients
    x, y = points[:, 0], points[:, 1]
    radius2 = x * x + y * y
    delta_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x) + sx1 * radius2
    delta_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y + sy1 * radius2
    return np.column_stack((delta_x, delta_y))


def _apply_thin_prism_distortion(points: np.ndarray, radial, tangential) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1)
    radial_distorted = _apply_radial(radius, radial)
    radial_scale = np.divide(
        radial_distorted,
        radius,
        out=np.ones_like(radius),
        where=radius > 1e-12,
    )
    return points * radial_scale[:, None] + _apply_non_radial_delta(points, tangential)


def _remove_thin_prism_distortion(points: np.ndarray, radial, tangential) -> np.ndarray:
    undistorted = points.copy()
    _clip_equidistant_radius(undistorted)
    for _ in range(20):
        applied = _apply_thin_prism_distortion(undistorted, radial, tangential)
        undistorted += points - applied
        _clip_equidistant_radius(undistorted)
    return undistorted


def _clip_equidistant_radius(points: np.ndarray) -> None:
    radius = np.linalg.norm(points, axis=1)
    scale = np.divide(
        np.minimum(radius, np.pi / 2),
        radius,
        out=np.ones_like(radius),
        where=radius > 1e-12,
    )
    points *= scale[:, None]
