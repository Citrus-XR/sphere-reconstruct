"""COLMAP sparse model の binary リーダー.

cameras.bin / images.bin / points3D.bin を読む. フォーマットは COLMAP 公式の
src/colmap/scene/reconstruction.cc / *.h に準拠 (little-endian).

純粋な struct 解析なので API プロセスからも使える (pycolmap 不要).

参考: https://colmap.github.io/format.html#binary-file-format
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


# COLMAP camera model id -> (name, num_params).
# https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),           # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 5),
    9: ("RADIAL_FISHEYE", 6),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class ImagePoint2D:
    x: float
    y: float
    point3D_id: int  # -1 (2**64-1 as stored) なら未対応


@dataclass
class Image:
    image_id: int
    qvec: tuple[float, float, float, float]  # qw, qx, qy, qz (world->cam)
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    points2D: list[ImagePoint2D] = field(default_factory=list)

    @property
    def num_registered_points(self) -> int:
        return sum(1 for p in self.points2D if p.point3D_id != _INVALID_POINT3D)


@dataclass
class Point3D:
    point3D_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: list[tuple[int, int]] = field(default_factory=list)  # (image_id, point2D_idx)


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3D: dict[int, Point3D]

    def summary(self) -> dict:
        errs = [p.error for p in self.points3D.values()]
        mean_err = sum(errs) / len(errs) if errs else 0.0
        track_lengths = [len(p.track) for p in self.points3D.values()]
        mean_track = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
        return {
            "num_cameras": len(self.cameras),
            "num_images": len(self.images),
            "num_points3D": len(self.points3D),
            "mean_reprojection_error": mean_err,
            "mean_track_length": mean_track,
        }


_INVALID_POINT3D = 2**64 - 1


def _read(fmt: str, f) -> tuple:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) < size:
        raise EOFError(f"unexpected EOF reading {fmt}")
    return struct.unpack(fmt, data)


def read_cameras_bin(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as f:
        (num,) = _read("<Q", f)
        for _ in range(num):
            camera_id, model_id, width, height = _read("<iiQQ", f)
            name, nparams = _CAMERA_MODELS.get(model_id, (f"UNKNOWN_{model_id}", 0))
            params = list(_read(f"<{nparams}d", f)) if nparams else []
            cameras[camera_id] = Camera(camera_id, name, width, height, params)
    return cameras


def read_images_bin(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with path.open("rb") as f:
        (num,) = _read("<Q", f)
        for _ in range(num):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read("<idddddddi", f)
            # name: null 終端バイト列.
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", "replace")
            (num_pts,) = _read("<Q", f)
            pts: list[ImagePoint2D] = []
            for _ in range(num_pts):
                x, y, pid = _read("<ddQ", f)
                pts.append(ImagePoint2D(x, y, pid))
            images[image_id] = Image(
                image_id=image_id,
                qvec=(qw, qx, qy, qz),
                tvec=(tx, ty, tz),
                camera_id=camera_id,
                name=name,
                points2D=pts,
            )
    return images


def read_points3D_bin(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    with path.open("rb") as f:
        (num,) = _read("<Q", f)
        for _ in range(num):
            pid, x, y, z, r, g, b, error = _read("<QdddBBBd", f)
            (track_len,) = _read("<Q", f)
            track: list[tuple[int, int]] = []
            for _ in range(track_len):
                image_id, pt2d_idx = _read("<ii", f)
                track.append((image_id, pt2d_idx))
            points[pid] = Point3D(pid, (x, y, z), (r, g, b), error, track)
    return points


def read_model(sparse_dir: Path) -> Reconstruction:
    """sparse/0/ ディレクトリ (cameras.bin, images.bin, points3D.bin) を読む."""
    cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    points = read_points3D_bin(sparse_dir / "points3D.bin")
    return Reconstruction(cameras=cameras, images=images, points3D=points)
