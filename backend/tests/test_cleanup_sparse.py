"""遠距離・低視差を組み合わせた sparse cleanup の保守的 filter。"""

from types import SimpleNamespace

from sphere_reconstruct.colmap.model import (
    Camera,
    Image,
    ImagePoint2D,
    Point3D,
    Reconstruction,
)
from sphere_reconstruct.stages.cleanup_sparse import (
    _filter_reconstruction,
    _has_triangulation_angle,
)


def _image(image_id: int, center_x: float, point_ids: list[int]) -> Image:
    return Image(
        image_id,
        (1.0, 0.0, 0.0, 0.0),
        (-center_x, 0.0, 0.0),
        1,
        f"frame_{image_id:06d}.jpg",
        [ImagePoint2D(float(index), 0.0, point_id) for index, point_id in enumerate(point_ids)],
    )


def test_triangulation_angle_requires_one_sufficient_observation_pair():
    point = Point3D(1, (0.0, 0.0, 10.0), (255, 255, 255), 0.5, [(1, 0), (2, 0)])

    assert _has_triangulation_angle(
        point,
        {1: _image(1, 0.0, []).camera_center, 2: _image(2, 1.0, []).camera_center},
        2.0,
    )
    assert not _has_triangulation_angle(
        point,
        {1: _image(1, 0.0, []).camera_center, 2: _image(2, 0.1, []).camera_center},
        2.0,
    )


def test_cleanup_removes_only_far_low_angle_point_and_clears_its_observations():
    images = {
        1: _image(1, 0.0, [1, 2, 3]),
        2: _image(2, 1.0, [1, 2]),
        3: _image(3, 10.0, [3]),
    }
    reconstruction = Reconstruction(
        {1: Camera(1, "PINHOLE", 100, 100, [100.0, 100.0, 50.0, 50.0], 1)},
        images,
        {
            1: Point3D(1, (0.0, 0.0, 0.1), (255, 255, 255), 0.5, [(1, 0), (2, 0)]),
            2: Point3D(2, (0.0, 0.0, 100.0), (255, 255, 255), 0.5, [(1, 1), (2, 1)]),
            3: Point3D(3, (0.0, 0.0, 100.0), (255, 255, 255), 0.5, [(1, 2), (3, 0)]),
        },
    )
    context = SimpleNamespace(
        primary_source=None,
        params={
            "far_distance_ratio": 0.3,
            "far_min_triangulation_deg": 2.0,
            "max_reprojection_error": 0.0,
            "min_track_length": 2,
        },
    )

    result = _filter_reconstruction(context, reconstruction)

    assert result["removed_points"] == 1
    assert result["removed_far_low_angle"] == 1
    assert set(reconstruction.points3D) == {1, 3}
    assert images[1].points2D[1].point3D_id == -1
    assert images[2].points2D[1].point3D_id == -1
    assert images[1].points2D[0].point3D_id == 1
