"""COLMAP sparse model の binary リーダー.

cameras.bin / images.bin / points3D.bin を読む. フォーマットは COLMAP 公式の
src/colmap/scene/reconstruction.cc / *.h に準拠 (little-endian).

純粋な struct 解析なので API プロセスからも使える (pycolmap 不要).

参考: https://colmap.github.io/format.html#binary-file-format
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# COLMAP camera model id -> (name, num_params).
# https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),  # f, cx, cy
    1: ("PINHOLE", 4),  # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),  # f, cx, cy, k
    3: ("RADIAL", 5),  # f, cx, cy, k1, k2
    4: ("OPENCV", 8),  # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("SIMPLE_DIVISION", 4),
    13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3),
    15: ("FISHEYE", 4),
    16: ("EUCM", 6),
    17: ("EQUIRECTANGULAR", 2),
}

# LFStudio v0.5.3 の COLMAP loader が実際に生成可能な camera model。FOV (ID 7) は
# metadata parser には存在するが image assembly で明示的に拒否される。
LFSTUDIO_SUPPORTED_CAMERA_MODEL_IDS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 17})


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]
    model_id: int | None = None


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

    @property
    def camera_center(self) -> tuple[float, float, float]:
        return camera_center(self.qvec, self.tvec)


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
        sorted_errors = sorted(errs)
        sorted_tracks = sorted(track_lengths)
        result = {
            "num_cameras": len(self.cameras),
            "num_images": len(self.images),
            "num_points3D": len(self.points3D),
            "mean_reprojection_error": mean_err,
            "median_reprojection_error": _percentile(sorted_errors, 0.5),
            "p95_reprojection_error": _percentile(sorted_errors, 0.95),
            "mean_track_length": mean_track,
            "median_track_length": _percentile(sorted_tracks, 0.5),
            "num_observations": sum(track_lengths),
        }
        if self.images:
            centers = [image.camera_center for image in self.images.values()]
            mins = [min(center[axis] for center in centers) for axis in range(3)]
            maxs = [max(center[axis] for center in centers) for axis in range(3)]
            spans = [maxs[axis] - mins[axis] for axis in range(3)]
            result.update(
                {
                    "camera_center_span": spans,
                    "camera_trajectory_diameter": sum(span * span for span in spans) ** 0.5,
                    "unique_camera_centers": len(
                        {tuple(round(value, 6) for value in center) for center in centers}
                    ),
                }
            )
        return result


_INVALID_POINT3D = 2**64 - 1


def write_cameras_bin(path: Path, cameras: dict[int, Camera]) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(cameras)))
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            model_id = camera.model_id
            if model_id is None:
                model_id = next(
                    (candidate for candidate, (name, _) in _CAMERA_MODELS.items() if name == camera.model),
                    None,
                )
            if model_id is None:
                raise ValueError(f"unsupported COLMAP camera model: {camera.model}")
            expected_params = _CAMERA_MODELS[model_id][1]
            if len(camera.params) != expected_params:
                raise ValueError(
                    f"camera {camera.camera_id} expects {expected_params} parameters, got {len(camera.params)}"
                )
            file.write(struct.pack("<iiQQ", camera.camera_id, model_id, camera.width, camera.height))
            file.write(struct.pack(f"<{expected_params}d", *camera.params))


def write_images_bin(
    path: Path,
    images: dict[int, Image],
    progress: Callable[[int, int], None] | None = None,
) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(images)))
        for image_number, image_id in enumerate(sorted(images), 1):
            image = images[image_id]
            file.write(
                struct.pack(
                    "<idddddddi",
                    image.image_id,
                    *image.qvec,
                    *image.tvec,
                    image.camera_id,
                )
            )
            file.write(image.name.encode("utf-8") + b"\0")
            file.write(struct.pack("<Q", len(image.points2D)))
            for point in image.points2D:
                point3d_id = _INVALID_POINT3D if point.point3D_id == -1 else point.point3D_id
                file.write(struct.pack("<ddQ", point.x, point.y, point3d_id))
            if progress is not None:
                progress(image_number, len(images))


def write_points3D_bin(path: Path, points: dict[int, Point3D]) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(points)))
        for point_id in sorted(points):
            point = points[point_id]
            file.write(
                struct.pack(
                    "<QdddBBBd",
                    point.point3D_id,
                    *point.xyz,
                    *point.rgb,
                    point.error,
                )
            )
            file.write(struct.pack("<Q", len(point.track)))
            for image_id, point2d_index in point.track:
                file.write(struct.pack("<ii", image_id, point2d_index))


def _percentile(sorted_values, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(len(sorted_values) - 1, low + 1)
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _read(fmt: str, f) -> tuple:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) < size:
        raise EOFError(f"unexpected EOF reading {fmt}")
    return struct.unpack(fmt, data)


def qvec_to_rotation(qvec) -> tuple[tuple[float, float, float], ...]:
    """COLMAP の world->camera quaternion を 3x3 回転行列へ変換する."""
    qw, qx, qy, qz = qvec
    return (
        (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )


def camera_center(qvec, tvec) -> tuple[float, float, float]:
    """world->camera pose から world 座標の中心 C=-R^T t を返す."""
    rotation = qvec_to_rotation(qvec)
    return tuple(-sum(rotation[row][axis] * tvec[row] for row in range(3)) for axis in range(3))


def read_cameras_bin(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as f:
        (num,) = _read("<Q", f)
        for _ in range(num):
            camera_id, model_id, width, height = _read("<iiQQ", f)
            try:
                name, nparams = _CAMERA_MODELS[model_id]
            except KeyError as error:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}") from error
            params = list(_read(f"<{nparams}d", f)) if nparams else []
            cameras[camera_id] = Camera(camera_id, name, width, height, params, model_id)
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
