"""Camera model と physical image circle の共通 valid-region geometry。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import cast

import numpy as np

_BLOCK_WORKING_BYTES = 8 * 1024 * 1024


def cache_key(region: dict, width: int, height: int) -> str:
    _validate_dimensions(width, height)
    return f"{width}x{height}:" + json.dumps(
        region, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def render_mask(region: dict, width: int, height: int) -> np.ndarray:
    """Valid pixel を 1 とする uint8 mask を返す。"""
    _validate_dimensions(width, height)
    if region.get("kind") == "full":
        return np.ones((height, width), dtype=np.uint8)
    mask = np.empty((height, width), dtype=np.uint8)
    for start, block in _mask_blocks(region, width, height):
        mask[start : start + len(block)] = block
    return mask


def bounding_box(region: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """Valid pixel を包含する最小の pixel-edge bounds を返す。"""
    _validate_dimensions(width, height)
    if region.get("kind") == "full":
        return 0.0, 0.0, float(width), float(height)

    minimum_x = width
    minimum_y = height
    maximum_x = -1
    maximum_y = -1
    for start, block in _mask_blocks(region, width, height):
        populated_rows = np.flatnonzero(np.any(block, axis=1))
        if not len(populated_rows):
            continue
        populated_columns = np.flatnonzero(np.any(block, axis=0))
        minimum_x = min(minimum_x, int(populated_columns[0]))
        maximum_x = max(maximum_x, int(populated_columns[-1]))
        minimum_y = min(minimum_y, start + int(populated_rows[0]))
        maximum_y = max(maximum_y, start + int(populated_rows[-1]))
    if maximum_x < minimum_x or maximum_y < minimum_y:
        raise ValueError(f"valid region has no pixels: {region}")
    return float(minimum_x), float(minimum_y), float(maximum_x + 1), float(maximum_y + 1)


def _mask_blocks(region: dict, width: int, height: int) -> Iterator[tuple[int, np.ndarray]]:
    kind = region.get("kind")
    block_rows = max(1, _BLOCK_WORKING_BYTES // max(1, width * 8))
    pixel_x = np.arange(width, dtype=np.float64)

    if kind == "circle":
        circle_x_squared, circle_y_squared, circle_radius_squared = _circle_terms(
            region, pixel_x, height, width
        )
        for start in range(0, height, block_rows):
            end = min(height, start + block_rows)
            block = (
                circle_y_squared[start:end, np.newaxis] + circle_x_squared[np.newaxis, :]
                <= circle_radius_squared
            )
            yield start, block.astype(np.uint8)
        return

    if kind != "opencv_fisheye":
        raise ValueError(f"unsupported valid region kind: {kind}")
    params = _fisheye_params(region)
    fx, fy, cx, cy, *_distortion = params
    radial_limit_squared = _distorted_radius_limit(params, _maximum_theta(region)) ** 2
    projection_x_squared = ((pixel_x + 0.5 - cx) / fx) ** 2
    projection_y_squared = ((np.arange(height, dtype=np.float64) + 0.5 - cy) / fy) ** 2
    physical_circle = region.get("physical_circle")
    circle_terms = (
        _circle_terms(physical_circle, pixel_x, height, width)
        if physical_circle is not None
        else None
    )
    for start in range(0, height, block_rows):
        end = min(height, start + block_rows)
        block = (
            projection_y_squared[start:end, np.newaxis]
            + projection_x_squared[np.newaxis, :]
            < radial_limit_squared
        )
        if circle_terms is not None:
            circle_x_squared, circle_y_squared, circle_radius_squared = circle_terms
            block &= (
                circle_y_squared[start:end, np.newaxis] + circle_x_squared[np.newaxis, :]
                <= circle_radius_squared
            )
        yield start, block.astype(np.uint8)


def _circle_terms(
    region: dict,
    pixel_x: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Version 2 circle は SVG と同じ image-edge 正規化座標。Pixel center と比較するため +0.5。
    cx = float(region["cx"]) * width
    cy = float(region["cy"]) * height
    radius = float(region["r"]) * width
    if radius <= 0:
        raise ValueError(f"valid circle radius must be positive: {radius}")
    return (
        (pixel_x + 0.5 - cx) ** 2,
        (np.arange(height, dtype=np.float64) + 0.5 - cy) ** 2,
        radius * radius,
    )


def _fisheye_params(region: dict) -> tuple[float, float, float, float, float, float, float, float]:
    params = tuple(float(value) for value in region["params"])
    if len(params) != 8:
        raise ValueError(f"OPENCV_FISHEYE requires 8 parameters: {len(params)}")
    if params[0] <= 0 or params[1] <= 0:
        raise ValueError(f"fisheye focal length must be positive: {params[:2]}")
    return cast(tuple[float, float, float, float, float, float, float, float], params)


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
    # r(theta) の導関数は x=theta^2 の 4 次式。等間隔 sample は狭い反転区間を
    # 見落とすため、導関数の極値を与える 3 次式の全実根で正値性を検査する。
    derivative_coefficients = np.asarray([1.0, 3.0 * k1, 5.0 * k2, 7.0 * k3, 9.0 * k4])
    stationary_coefficients = np.asarray([3.0 * k1, 10.0 * k2, 21.0 * k3, 36.0 * k4])
    nonzero = np.flatnonzero(stationary_coefficients)
    roots = (
        np.polynomial.polynomial.polyroots(stationary_coefficients[: nonzero[-1] + 1])
        if len(nonzero)
        else np.asarray([], dtype=np.complex128)
    )
    maximum_x = theta * theta
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
