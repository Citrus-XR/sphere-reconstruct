"""Camera-model valid region が forward ray だけを保持することを検証する。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sphere_reconstruct.imaging import valid_region


def test_opencv_fisheye_mask_intersects_projection_and_physical_circle():
    width = height = 101
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [30.0, 32.0, 55.0, 48.0, 0.01, 0.005, -0.001, 0.0001],
        "max_theta_rad": 1.4,
        "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.48},
    }

    mask = valid_region.render_mask(region, width, height)

    assert mask.shape == (height, width)
    assert mask.dtype == np.uint8
    assert mask[48, 55] == 1
    assert mask[48, 3] == 0  # Physical circle 内でも projection hemisphere 外。
    assert mask[0, 55] == 0  # Projection 内でも physical circle 外。

    y, x = np.nonzero(mask)
    fx, fy, cx, cy, k1, k2, k3, k4 = region["params"]
    distorted_radius = np.sqrt(((x + 0.5 - cx) / fx) ** 2 + ((y + 0.5 - cy) / fy) ** 2)
    theta = region["max_theta_rad"]
    theta_squared = theta * theta
    limit = theta * (
        1.0
        + theta_squared
        * (k1 + theta_squared * (k2 + theta_squared * (k3 + theta_squared * k4)))
    )
    assert np.all(distorted_radius < limit)


def test_opencv_fisheye_bounds_contain_every_valid_pixel():
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [80.0, 70.0, 110.0, 95.0, 0.02, 0.01, 0.0, 0.0],
        "max_theta_rad": math.pi / 2 * 0.995,
        "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.47},
    }
    mask = valid_region.render_mask(region, 200, 180)
    left, top, right, bottom = valid_region.bounding_box(region, 200, 180)
    y, x = np.nonzero(mask)

    assert x.min() + 0.5 >= left
    assert y.min() + 0.5 >= top
    assert x.max() + 0.5 <= right
    assert y.max() + 0.5 <= bottom


def test_thin_prism_fisheye_mask_uses_inverse_camera_model():
    region = {
        "kind": "fisheye",
        "camera_model": "THIN_PRISM_FISHEYE",
        "params": [80.0, 78.0, 100.0, 90.0, 0.02, 0.01, 0.001, -0.0005, 0.0, 0.0, 0.0002, -0.0001],
        "max_theta_rad": 1.4,
        "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.48},
    }

    mask = valid_region.render_mask(region, 200, 180)
    left, top, right, bottom = valid_region.bounding_box(region, 200, 180)

    assert mask[90, 100] == 1
    assert mask[0, 0] == 0
    y, x = np.nonzero(mask)
    assert x.min() + 0.5 >= left
    assert y.min() + 0.5 >= top
    assert x.max() + 0.5 <= right
    assert y.max() + 0.5 <= bottom


def test_fisheye_theta_limit_must_exclude_the_back_hemisphere():
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [20.0, 20.0, 32.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        "max_theta_rad": math.pi / 2,
    }

    with pytest.raises(ValueError, match="pi/2"):
        valid_region.render_mask(region, 64, 64)


def test_non_monotonic_fisheye_distortion_is_rejected():
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [20.0, 20.0, 32.0, 32.0, -0.2, 0.0, 0.0, 0.0],
        "max_theta_rad": 1.4,
    }

    with pytest.raises(ValueError, match="not monotonic"):
        valid_region.render_mask(region, 64, 64)


def test_narrow_non_monotonic_interval_between_uniform_samples_is_rejected():
    center = 1.0
    half_width_squared = 1e-8
    denominator = center * center - half_width_squared
    k1 = (-2.0 * center / denominator) / 3.0
    k2 = (1.0 / denominator) / 5.0
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [20.0, 20.0, 32.0, 32.0, k1, k2, 0.0, 0.0],
        "max_theta_rad": 1.4,
    }

    with pytest.raises(ValueError, match="not monotonic"):
        valid_region.render_mask(region, 64, 64)


def test_bounding_box_rejects_disjoint_shapes_with_overlapping_aabbs():
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [20.0, 20.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        "max_theta_rad": 1.0,
        "physical_circle": {"cx": 0.75, "cy": 0.75, "r": 0.1},
    }

    assert valid_region.render_mask(region, 100, 100).sum() == 0
    with pytest.raises(ValueError, match="no pixels"):
        valid_region.bounding_box(region, 100, 100)
