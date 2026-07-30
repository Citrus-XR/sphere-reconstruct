"""Native camera ray を使う dense initialization の幾何と model 追記を検証する。"""

from __future__ import annotations

import numpy as np
from PIL import Image

from sphere_reconstruct.colmap.model import Camera, Point3D, Reconstruction
from sphere_reconstruct.colmap.model import Image as ColmapImage
from sphere_reconstruct.dense_init.geometry import (
    camera_rays_to_pixels,
    pixels_to_camera_rays,
    triangulate_rays,
)
from sphere_reconstruct.dense_init.matcher import DenseMatches
from sphere_reconstruct.dense_init.pipeline import DenseInitializationConfig, densify_reconstruction


def test_opencv_fisheye_pixel_ray_roundtrip():
    camera = Camera(
        1,
        "OPENCV_FISHEYE",
        3840,
        3840,
        [1020.0, 1018.0, 1924.0, 1916.0, 0.019, 0.023, -0.0084, 0.00032],
        5,
    )
    pixels = np.asarray([[1924.0, 1916.0], [800.0, 1000.0], [3000.0, 2500.0]])

    rays = pixels_to_camera_rays(camera, pixels)
    restored = camera_rays_to_pixels(camera, rays)

    np.testing.assert_allclose(restored, pixels, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)


def test_ray_triangulation_recovers_known_point():
    point = np.asarray([[0.25, -0.1, 4.0]])
    origins_a = np.asarray([[0.0, 0.0, 0.0]])
    origins_b = np.asarray([[1.0, 0.0, 0.0]])
    directions_a = point - origins_a
    directions_b = point - origins_b

    triangulated, depth_a, depth_b, geometry = triangulate_rays(
        origins_a, directions_a, origins_b, directions_b
    )

    np.testing.assert_allclose(triangulated, point, atol=1e-10)
    assert depth_a[0] > 0 and depth_b[0] > 0
    assert geometry[0, 0] < 1e-10
    assert geometry[0, 1] > 0.5


class _ExactMatcher:
    def __init__(self, matches: DenseMatches):
        self.matches = matches

    def match(self, image_a, image_b, *, count):
        del image_a, image_b, count
        return self.matches


def test_dense_pipeline_appends_filtered_points(tmp_path):
    camera = Camera(1, "PINHOLE", 100, 100, [100.0, 100.0, 50.0, 50.0], 1)
    first = ColmapImage(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "front/frame_000000.jpg")
    second = ColmapImage(2, (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1, "front/frame_000001.jpg")
    base_points = {
        1: Point3D(1, (-1.0, 0.0, 5.0), (128, 128, 128), 0.5),
        2: Point3D(2, (0.0, 0.0, 5.0), (128, 128, 128), 0.5),
        3: Point3D(3, (1.0, 0.0, 5.0), (128, 128, 128), 0.5),
    }
    reconstruction = Reconstruction({1: camera}, {1: first, 2: second}, base_points)
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    for name in (first.name, second.name):
        path = image_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), (40, 80, 120)).save(path)
        mask = mask_root / f"{name}.png"
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (100, 100), 255).save(mask)
    target = np.asarray([[0.0, 0.0, 5.0], [0.4, 0.2, 6.0]])
    rays_a = target / target[:, 2:3]
    rays_b = (target - np.asarray([1.0, 0.0, 0.0])) / target[:, 2:3]
    pixels_a = np.column_stack((100 * rays_a[:, 0] + 50, 100 * rays_a[:, 1] + 50))
    pixels_b = np.column_stack((100 * rays_b[:, 0] + 50, 100 * rays_b[:, 1] + 50))
    matcher = _ExactMatcher(DenseMatches(pixels_a, pixels_b, np.ones(2)))
    catalog = {
        "images": [
            {"name": first.name, "source_id": "s1", "sensor_id": "front", "capture_index": 0},
            {"name": second.name, "source_id": "s1", "sensor_id": "front", "capture_index": 1},
        ]
    }

    result = densify_reconstruction(
        reconstruction,
        catalog,
        image_root,
        mask_root,
        matcher,
        DenseInitializationConfig(
            reference_fraction=0.5,
            neighbors_per_reference=1,
            matches_per_pair=2,
            maximum_ray_gap_ratio=0.01,
            voxel_size_ratio=0.00001,
            maximum_new_points=10,
        ),
    )

    assert result.pairs_considered == 1
    assert result.kept_points == 2
    assert len(reconstruction.points3D) == 5
    assert len(first.points2D) == 2
    assert len(second.points2D) == 2
    for point in list(reconstruction.points3D.values())[-2:]:
        assert len(point.track) == 2
