"""Apply catalog calibration provenance to COLMAP camera identities."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def apply_camera_policies(database_path: Path, catalog: dict) -> list[int]:
    """Set focal-prior flags and return cameras whose intrinsics must stay fixed."""
    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    policy_by_name: dict[str, tuple[bool, bool]] = {}
    group_by_name: dict[str, str] = {}
    for group in catalog["camera_groups"]:
        group_id = str(group["id"])
        prior = group["has_prior_focal_length"]
        refine = group["refine_intrinsics"]
        if not isinstance(prior, bool) or not isinstance(refine, bool):
            raise ValueError(f"camera group {group_id}: calibration policy must contain booleans")
        for name in group["image_names"]:
            if name in policy_by_name:
                raise ValueError(
                    f"image {name} belongs to multiple camera groups: "
                    f"{group_by_name[name]}, {group_id}"
                )
            policy_by_name[name] = (prior, refine)
            group_by_name[name] = group_id

    catalog_names = [image["name"] for image in catalog["images"]]
    if len(catalog_names) != len(set(catalog_names)):
        raise ValueError("image catalog contains duplicate image names")
    _validate_names(set(catalog_names), set(policy_by_name), "camera groups")

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name, camera_id FROM images").fetchall()
        _validate_names(set(catalog_names), {name for name, _camera_id in rows}, "COLMAP database")
        policy_by_camera: dict[int, tuple[bool, bool]] = {}
        first_image_by_camera: dict[int, str] = {}
        for name, camera_id in rows:
            policy = policy_by_name[name]
            if camera_id in policy_by_camera and policy_by_camera[camera_id] != policy:
                raise ValueError(
                    f"COLMAP camera {camera_id} has conflicting calibration policies: "
                    f"{first_image_by_camera[camera_id]}, {name}"
                )
            policy_by_camera[camera_id] = policy
            first_image_by_camera[camera_id] = name

        for camera_id, (prior, _refine) in policy_by_camera.items():
            updated = connection.execute(
                "UPDATE cameras SET prior_focal_length = ? WHERE camera_id = ?",
                (int(prior), camera_id),
            )
            if updated.rowcount != 1:
                raise ValueError(f"COLMAP images refer to missing camera {camera_id}")

    return sorted(camera_id for camera_id, (_prior, refine) in policy_by_camera.items() if not refine)


def _validate_names(expected: set[str], actual: set[str], owner: str) -> None:
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"{owner} does not match image catalog: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )
