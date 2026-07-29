"""学習 profile が疎点群の裾を破壊せず診断値として公開することを検証する。"""

from __future__ import annotations

from sphere_reconstruct.colmap.model import Camera, Image, Point3D, Reconstruction
from sphere_reconstruct.colmap.train_profile import compute_profile


def test_sparse_point_radius_distribution_is_reported():
    reconstruction = Reconstruction(
        {1: Camera(1, "PINHOLE", 64, 64, [20.0, 20.0, 32.0, 32.0])},
        {
            1: Image(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "frame_000.jpg"),
            2: Image(2, (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1, "frame_001.jpg"),
        },
        {
            index: Point3D(index, (distance, 0.0, 0.0), (128, 128, 128), 0.5)
            for index, distance in enumerate((-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0), 1)
        },
    )

    profile = compute_profile(reconstruction)

    assert profile["sparse_point_radius_median"] == 2.0
    assert profile["sparse_point_radius_p95"] > 90.0
    assert profile["sparse_point_radius_p99"] >= profile["sparse_point_radius_p95"]
    assert profile["sparse_point_radius_max"] == 100.0
    assert profile["sparse_point_radius_p99_to_median"] > 40.0
