"""Camera model と physical image circle の共通 valid-region geometry。"""

from __future__ import annotations

import math

import numpy as np


def render_mask(region: dict, width: int, height: int) -> np.ndarray:
    """Valid pixel を 1 とする uint8 mask を返す。"""
    _validate_dimensions(width, height)
    kind = region.get("kind")
    if kind == "full":
        return np.ones((height, width), dtype=np.uint8)
    if kind == "circle":
        return _circle_mask(region, width, height)
    if kind == "opencv_fisheye":
        mask = _opencv_fisheye_mask(region, width, height)
        physical_circle = region.get("physical_circle")
        if physical_circle is not None:
            mask &= _circle_mask(physical_circle, width, height)
        return mask
    raise ValueError(f"unsupported valid region kind: {kind}")


def bounding_box(region: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """Valid region を包含する axis-aligned pixel-edge bounds を返す。"""
    _validate_dimensions(width, height)
    kind = region.get("kind")
    if kind == "full":
        return 0.0, 0.0, float(width), float(height)
    if kind == "circle":
        bounds = _circle_bounds(region, width, height)
    elif kind == "opencv_fisheye":
        params = _fisheye_params(region)
        radial_limit = _distorted_radius_limit(params, _maximum_theta(region))
        bounds = (
            params[2] - params[0] * radial_limit,
            params[3] - params[1] * radial_limit,
            params[2] + params[0] * radial_limit,
            params[3] + params[1] * radial_limit,
        )
        physical_circle = region.get("physical_circle")
        if physical_circle is not None:
            physical = _circle_bounds(physical_circle, width, height)
            bounds = (
                max(bounds[0], physical[0]),
                max(bounds[1], physical[1]),
                min(bounds[2], physical[2]),
                min(bounds[3], physical[3]),
            )
    else:
        raise ValueError(f"unsupported valid region kind: {kind}")
    left, top, right, bottom = bounds
    clipped = (
        max(0.0, left),
        max(0.0, top),
        min(float(width), right),
        min(float(height), bottom),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"valid region has no pixels: {region}")
    return clipped


def _opencv_fisheye_mask(region: dict, width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy, *_distortion = _fisheye_params(region)
    radial_limit = _distorted_radius_limit(_fisheye_params(region), _maximum_theta(region))
    x_squared = ((np.arange(width, dtype=np.float64) + 0.5 - cx) / fx) ** 2
    y_squared = ((np.arange(height, dtype=np.float64) + 0.5 - cy) / fy) ** 2
    return (y_squared[:, np.newaxis] + x_squared[np.newaxis, :] < radial_limit**2).astype(
        np.uint8
    )


def _circle_mask(region: dict, width: int, height: int) -> np.ndarray:
    cx = float(region["cx"]) * width
    cy = float(region["cy"]) * height
    radius = float(region["r"]) * width
    if radius <= 0:
        raise ValueError(f"valid circle radius must be positive: {radius}")
    x_squared = (np.arange(width, dtype=np.float64) + 0.5 - cx) ** 2
    y_squared = (np.arange(height, dtype=np.float64) + 0.5 - cy) ** 2
    return (y_squared[:, np.newaxis] + x_squared[np.newaxis, :] < radius**2).astype(np.uint8)


def _circle_bounds(region: dict, width: int, height: int) -> tuple[float, float, float, float]:
    cx = float(region["cx"]) * width
    cy = float(region["cy"]) * height
    radius = float(region["r"]) * width
    if radius <= 0:
        raise ValueError(f"valid circle radius must be positive: {radius}")
    return cx - radius, cy - radius, cx + radius, cy + radius


def _fisheye_params(region: dict) -> tuple[float, float, float, float, float, float, float, float]:
    params = tuple(float(value) for value in region["params"])
    if len(params) != 8:
        raise ValueError(f"OPENCV_FISHEYE requires 8 parameters: {len(params)}")
    if params[0] <= 0 or params[1] <= 0:
        raise ValueError(f"fisheye focal length must be positive: {params[:2]}")
    return params  # type: ignore[return-value]


def _maximum_theta(region: dict) -> float:
    maximum_theta = float(region["max_theta_rad"])
    if not 0 < maximum_theta < math.pi / 2:
        raise ValueError(f"maximum fisheye theta must be within (0, pi/2): {maximum_theta}")
    return maximum_theta


def _distorted_radius_limit(
    params: tuple[float, float, float, float, float, float, float, float],
    theta: float,
) -> float:
    k1, k2, k3, k4 = params[4:]
    theta_squared = theta * theta
    radial = 1.0 + theta_squared * (
        k1 + theta_squared * (k2 + theta_squared * (k3 + theta_squared * k4))
    )
    distorted = theta * radial
    if not math.isfinite(distorted) or distorted <= 0:
        raise ValueError(f"invalid fisheye distorted radius: {distorted}")
    return distorted


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive: {width}x{height}")
