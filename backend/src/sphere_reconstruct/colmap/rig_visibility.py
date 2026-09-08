"""Remove verified pairs that cannot share a ray in a calibrated rig."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from .database import MAX_IMAGE_ID
from .input_workspace import InputSpec


def remove_impossible_same_capture_pairs(
    database_path: Path,
    spec: InputSpec,
    catalog: dict,
    rig_config: list[dict],
) -> dict:
    """Delete same-capture edges whose calibrated angular caps do not intersect."""
    axes = _camera_axes(rig_config)
    if not axes:
        return {
            "enabled": False,
            "examined_pairs": 0,
            "removed_pairs": 0,
            "camera_pairs": {},
        }

    records_by_name = {
        str(record["name"]): record for record in catalog.get("images", [])
    }
    images_by_name = {str(image["name"]): image for image in spec.images}
    to_remove: set[int] = set()
    camera_pairs: dict[str, int] = defaultdict(int)
    examined_pairs = 0

    with sqlite3.connect(database_path) as connection:
        image_by_id = {
            int(image_id): images_by_name[name]
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
            if name in images_by_name
        }
        rows = connection.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows > 0"
        ).fetchall()
        for (pair_id,) in rows:
            first_id, second_id = _pair_ids(int(pair_id))
            first = image_by_id.get(first_id)
            second = image_by_id.get(second_id)
            if first is None or second is None:
                continue
            if (
                first["source_id"] != second["source_id"]
                or int(first["capture_index"]) != int(second["capture_index"])
            ):
                continue

            first_axis = _axis_for_name(str(first["name"]), axes)
            second_axis = _axis_for_name(str(second["name"]), axes)
            first_record = records_by_name.get(str(first["name"]))
            second_record = records_by_name.get(str(second["name"]))
            first_theta = _maximum_theta(first_record)
            second_theta = _maximum_theta(second_record)
            if (
                first_axis is None
                or second_axis is None
                or first_theta is None
                or second_theta is None
            ):
                continue

            examined_pairs += 1
            first_prefix, first_direction = first_axis
            second_prefix, second_direction = second_axis
            separation = math.acos(
                max(-1.0, min(1.0, _dot(first_direction, second_direction)))
            )
            if separation <= first_theta + second_theta:
                continue
            to_remove.add(int(pair_id))
            key = "|".join(sorted((first_prefix, second_prefix)))
            camera_pairs[key] += 1

        if to_remove:
            connection.executemany(
                "DELETE FROM matches WHERE pair_id = ?", ((pair_id,) for pair_id in to_remove)
            )
            connection.executemany(
                "DELETE FROM two_view_geometries WHERE pair_id = ?",
                ((pair_id,) for pair_id in to_remove),
            )
            connection.commit()

    return {
        "enabled": True,
        "examined_pairs": examined_pairs,
        "removed_pairs": len(to_remove),
        "camera_pairs": dict(sorted(camera_pairs.items())),
    }


def _camera_axes(rig_config: list[dict]) -> dict[str, tuple[float, float, float]]:
    axes: dict[str, tuple[float, float, float]] = {}
    for rig in rig_config:
        for camera in rig.get("cameras", []):
            prefix = str(camera["image_prefix"]).replace("\\", "/")
            quaternion = (
                (1.0, 0.0, 0.0, 0.0)
                if camera.get("ref_sensor") is True
                else tuple(float(value) for value in camera["cam_from_rig_rotation"])
            )
            axes[prefix] = _optical_axis_in_rig(quaternion)
    return axes


def _axis_for_name(
    name: str,
    axes: dict[str, tuple[float, float, float]],
) -> tuple[str, tuple[float, float, float]] | None:
    normalized = name.replace("\\", "/")
    prefixes = [prefix for prefix in axes if normalized.startswith(prefix)]
    if not prefixes:
        return None
    prefix = max(prefixes, key=len)
    return prefix, axes[prefix]


def _maximum_theta(record: dict | None) -> float | None:
    if record is None:
        return None
    region = record.get("valid_region", {})
    if region.get("kind") != "fisheye":
        return None
    theta = float(region["max_theta_rad"])
    if not 0.0 < theta < math.pi:
        raise ValueError(f"invalid rig visibility angle: {theta}")
    return theta


def _optical_axis_in_rig(quaternion: tuple[float, ...]) -> tuple[float, float, float]:
    if len(quaternion) != 4:
        raise ValueError("rig rotation must be a wxyz quaternion")
    w, x, y, z = quaternion
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(length) or length == 0.0:
        raise ValueError("rig rotation must be finite and non-zero")
    w, x, y, z = (value / length for value in (w, x, y, z))
    # R_cam_from_rig.T @ (0, 0, 1): the third row of the quaternion rotation matrix.
    return (
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _pair_ids(pair_id: int) -> tuple[int, int]:
    second_id = pair_id % MAX_IMAGE_ID
    return (pair_id - second_id) // MAX_IMAGE_ID, second_id
