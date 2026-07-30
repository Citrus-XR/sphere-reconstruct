"""Camera model と physical image circle の共通 valid-region geometry。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator

import numpy as np

from . import fisheye_camera

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
    block_rows = max(1, _BLOCK_WORKING_BYTES // max(1, width * 96))
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
            block = _apply_operations(
                block,
                region.get("operations", []),
                pixel_x,
                start,
                end,
                width,
                height,
            )
            yield start, block.astype(np.uint8)
        return

    if kind != "fisheye":
        raise ValueError(f"unsupported valid region kind: {kind}")
    model = str(region["camera_model"])
    params = tuple(float(value) for value in region["params"])
    maximum_theta = _maximum_theta(region)
    fisheye_camera.validate_forward_hemisphere(model, params, maximum_theta)
    physical_circle = region.get("physical_circle")
    circle_terms = (
        _circle_terms(physical_circle, pixel_x, height, width)
        if physical_circle is not None
        else None
    )
    for start in range(0, height, block_rows):
        end = min(height, start + block_rows)
        grid_x, grid_y = np.meshgrid(
            pixel_x + 0.5,
            np.arange(start, end, dtype=np.float64) + 0.5,
        )
        pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        theta = fisheye_camera.pixels_to_theta(model, params, pixels)
        block = (theta < maximum_theta).reshape((end - start, width))
        if circle_terms is not None:
            circle_x_squared, circle_y_squared, circle_radius_squared = circle_terms
            physical = (
                circle_y_squared[start:end, np.newaxis] + circle_x_squared[np.newaxis, :]
                <= circle_radius_squared
            )
            physical = _apply_operations(
                physical,
                physical_circle.get("operations", []),
                pixel_x,
                start,
                end,
                width,
                height,
            )
            block &= physical
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


def _maximum_theta(region: dict) -> float:
    maximum_theta = float(region["max_theta_rad"])
    if not 0 < maximum_theta < math.pi / 2:
        raise ValueError(f"maximum fisheye theta must be within (0, pi/2): {maximum_theta}")
    return maximum_theta


def _apply_operations(
    block: np.ndarray,
    operations: list,
    pixel_x: np.ndarray,
    start: int,
    end: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = block.copy()
    pixel_y = np.arange(start, end, dtype=np.float64)
    for index, operation in enumerate(operations):
        mode = operation.get("mode")
        if mode not in {"add", "subtract"}:
            raise ValueError(f"valid-region operation {index} の mode が不正です: {mode}")
        radius = float(operation["r"]) * width
        if radius <= 0.0:
            raise ValueError(f"valid-region operation {index} の radius が不正です: {radius}")
        inside = (
            (pixel_x + 0.5 - float(operation["x"]) * width) ** 2
        )[np.newaxis, :] + (
            (pixel_y + 0.5 - float(operation["y"]) * height) ** 2
        )[:, np.newaxis] <= radius * radius
        if mode == "add":
            result |= inside
        else:
            result &= ~inside
    return result


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive: {width}x{height}")
