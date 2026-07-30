"""colmap.model の binary リーダーテスト.

合成した cameras/images/points3D.bin を書いて読み戻し, 一致を確認する.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from sphere_reconstruct.colmap import model


def _write_cameras(path: Path):
    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        # camera_id=1, model=PINHOLE(1), w=512, h=512, params fx,fy,cx,cy
        f.write(struct.pack("<iiQQ", 1, 1, 512, 512))
        f.write(struct.pack("<4d", 256.0, 256.0, 256.0, 256.0))


def _write_images(path: Path):
    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        # image_id=1, qvec, tvec, camera_id=1, name, 2 points2D
        f.write(struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1))
        f.write(b"front_lens0.jpg\x00")
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<ddQ", 10.0, 20.0, 100))  # registered to point 100
        f.write(struct.pack("<ddQ", 30.0, 40.0, 2**64 - 1))  # unregistered


def _write_points(path: Path):
    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        # pid=100, xyz, rgb, error, track_len=1, (image_id=1, pt2d_idx=0)
        f.write(struct.pack("<QdddBBBd", 100, 1.5, 2.5, 3.5, 200, 150, 100, 0.7))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<ii", 1, 0))


def test_read_model_roundtrip(tmp_path: Path):
    _write_cameras(tmp_path / "cameras.bin")
    _write_images(tmp_path / "images.bin")
    _write_points(tmp_path / "points3D.bin")

    recon = model.read_model(tmp_path)

    assert len(recon.cameras) == 1
    cam = recon.cameras[1]
    assert cam.model == "PINHOLE"
    assert cam.model_id == 1
    assert cam.width == 512 and cam.height == 512
    assert cam.params == [256.0, 256.0, 256.0, 256.0]

    assert len(recon.images) == 1
    img = recon.images[1]
    assert img.name == "front_lens0.jpg"
    assert img.qvec == (1.0, 0.0, 0.0, 0.0)
    assert img.tvec == (1.0, 2.0, 3.0)
    assert len(img.points2D) == 2
    assert img.num_registered_points == 1

    assert len(recon.points3D) == 1
    p = recon.points3D[100]
    assert p.xyz == (1.5, 2.5, 3.5)
    assert p.rgb == (200, 150, 100)
    assert abs(p.error - 0.7) < 1e-9
    assert p.track == [(1, 0)]


def test_summary(tmp_path: Path):
    _write_cameras(tmp_path / "cameras.bin")
    _write_images(tmp_path / "images.bin")
    _write_points(tmp_path / "points3D.bin")
    recon = model.read_model(tmp_path)
    s = recon.summary()
    assert s["num_cameras"] == 1
    assert s["num_images"] == 1
    assert s["num_points3D"] == 1
    assert abs(s["mean_reprojection_error"] - 0.7) < 1e-9
    assert abs(s["median_reprojection_error"] - 0.7) < 1e-9
    assert abs(s["p95_reprojection_error"] - 0.7) < 1e-9
    assert s["mean_track_length"] == 1.0
    assert s["num_observations"] == 1
    assert s["unique_camera_centers"] == 1
    assert s["camera_center_span"] == [0.0, 0.0, 0.0]


def test_equirectangular_camera_model(tmp_path: Path):
    with (tmp_path / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 17, 4096, 2048))
        f.write(struct.pack("<2d", 4096.0, 2048.0))
    camera = model.read_cameras_bin(tmp_path / "cameras.bin")[1]
    assert camera.model == "EQUIRECTANGULAR"
    assert camera.model_id == 17
    assert camera.params == [4096.0, 2048.0]


def test_unknown_camera_model_fails_instead_of_desynchronizing(tmp_path: Path):
    with (tmp_path / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 999, 512, 512))
    with pytest.raises(ValueError, match="camera model id: 999"):
        model.read_cameras_bin(tmp_path / "cameras.bin")


def test_camera_center_uses_colmap_world_to_camera_convention():
    assert model.camera_center((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == (
        -1.0,
        -2.0,
        -3.0,
    )


def test_points3d_writer_roundtrips_tracks(tmp_path):
    points = {
        7: model.Point3D(7, (1.0, 2.0, 3.0), (10, 20, 30), 0.4, [(2, 5), (3, 8)])
    }

    model.write_points3D_bin(tmp_path / "points3D.bin", points)

    assert model.read_points3D_bin(tmp_path / "points3D.bin") == points


def test_images_writer_preserves_reconstruction_order(tmp_path):
    images = {
        20: model.Image(
            image_id=20,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 0.0),
            camera_id=1,
            name="front/frame_000001.jpg",
        ),
        3: model.Image(
            image_id=3,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 0.0),
            camera_id=1,
            name="back/frame_000001.jpg",
        ),
    }

    model.write_images_bin(tmp_path / "images.bin", images)

    assert list(model.read_images_bin(tmp_path / "images.bin")) == [20, 3]
