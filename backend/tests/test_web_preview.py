"""colmap.web_preview の単体テスト."""

from __future__ import annotations

import struct

from sphere_reconstruct.colmap import web_preview
from sphere_reconstruct.colmap.model import Camera, Image, Point3D, Reconstruction


def _make_recon(n_points: int = 10) -> Reconstruction:
    cameras = {1: Camera(1, "PINHOLE", 512, 512, [256.0, 256.0, 256.0, 256.0])}
    # 単位回転, t=(0,0,5) -> カメラ位置 C = -R^T t = (0,0,-5)
    images = {
        1: Image(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 5.0), 1, "front_lens0.jpg", []),
    }
    points = {}
    for i in range(n_points):
        points[i] = Point3D(i, (float(i), 0.0, 0.0), (i % 256, 0, 0), 0.5, [(1, i)])
    return Reconstruction(cameras=cameras, images=images, points3D=points)


def test_camera_position_from_identity():
    recon = _make_recon()
    wp = web_preview.build_web_preview(recon)
    img = wp.reconstruction_json["images"][0]
    # C = -R^T t = -(I)(0,0,5) = (0,0,-5)
    assert abs(img["position"][2] - (-5.0)) < 1e-6


def test_points_bin_header_and_stride():
    recon = _make_recon(n_points=10)
    wp = web_preview.build_web_preview(recon)
    num, stride = struct.unpack_from("<II", wp.points_bytes, 0)
    assert num == 10
    assert stride == 20
    # 全体サイズ = 8 (header) + 10 * 20
    assert len(wp.points_bytes) == 8 + 10 * 20


def test_downsample_to_max_points():
    recon = _make_recon(n_points=1000)
    wp = web_preview.build_web_preview(recon, max_points=100)
    assert wp.num_points_total == 1000
    assert wp.num_points_written == 100
    num, _ = struct.unpack_from("<II", wp.points_bytes, 0)
    assert num == 100


def test_reconstruction_json_stats():
    recon = _make_recon(n_points=5)
    wp = web_preview.build_web_preview(recon)
    stats = wp.reconstruction_json["stats"]
    assert stats["num_points3D"] == 5
    assert stats["num_images"] == 1
    assert stats["num_cameras"] == 1
