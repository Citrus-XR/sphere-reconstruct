"""Camera-model valid region が forward ray だけを保持することを検証する。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sphere_reconstruct.imaging import valid_region


def test_opencv_fisheye_mask_intersects_projection_and_physical_circle():
    width = height = 101
    region = {
        "kind": "opencv_fisheye",
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
        "kind": "opencv_fisheye",
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


def test_fisheye_theta_limit_must_exclude_the_back_hemisphere():
    region = {
        "kind": "opencv_fisheye",
        "params": [20.0, 20.0, 32.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        "max_theta_rad": math.pi / 2,
    }

    with pytest.raises(ValueError, match="pi/2"):
        valid_region.render_mask(region, 64, 64)
