from __future__ import annotations

import sqlite3
from pathlib import Path

from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.colmap.rig_visibility import remove_impossible_same_capture_pairs


def _pair_id(first: int, second: int) -> int:
    maximum_image_id = 2_147_483_647
    return min(first, second) * maximum_image_id + max(first, second)


def _spec() -> InputSpec:
    return InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=4,
        source_count=1,
        primary_source_id="source",
        primary_image_names=[],
        sources=[],
        images=[
            {
                "name": "sources/source/lens0/frame_000000.png",
                "source_id": "source",
                "capture_index": 0,
                "sensor_id": "lens0",
            },
            {
                "name": "sources/source/lens1/frame_000000.png",
                "source_id": "source",
                "capture_index": 0,
                "sensor_id": "lens1",
            },
            {
                "name": "sources/source/lens0/frame_000001.png",
                "source_id": "source",
                "capture_index": 1,
                "sensor_id": "lens0",
            },
            {
                "name": "sources/source/lens1/frame_000001.png",
                "source_id": "source",
                "capture_index": 1,
                "sensor_id": "lens1",
            },
        ],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path="rig_config.json",
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )


def _catalog() -> dict:
    return {
        "images": [
            {
                "name": image["name"],
                "valid_region": {"kind": "fisheye", "max_theta_rad": 1.4},
            }
            for image in _spec().images
        ]
    }


def _rig(rotation: list[float]) -> list[dict]:
    return [
        {
            "cameras": [
                {"image_prefix": "sources/source/lens0/", "ref_sensor": True},
                {
                    "image_prefix": "sources/source/lens1/",
                    "cam_from_rig_rotation": rotation,
                },
            ]
        }
    ]


def _database(path: Path, pairs: list[tuple[int, int]]) -> None:
    names = [image["name"] for image in _spec().images]
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?)",
            list(enumerate(names, 1)),
        )
        rows = [(_pair_id(first, second), 30) for first, second in pairs]
        connection.executemany("INSERT INTO matches VALUES (?, ?)", rows)
        connection.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", rows)


def test_removes_only_same_capture_pairs_with_disjoint_angular_caps(tmp_path: Path):
    database = tmp_path / "database.db"
    _database(database, [(1, 2), (1, 3), (1, 4), (3, 4)])

    result = remove_impossible_same_capture_pairs(
        database,
        _spec(),
        _catalog(),
        _rig([0.0, 0.0, 1.0, 0.0]),
    )

    assert result == {
        "enabled": True,
        "examined_pairs": 2,
        "removed_pairs": 2,
        "camera_pairs": {"sources/source/lens0/|sources/source/lens1/": 2},
    }
    with sqlite3.connect(database) as connection:
        matches = {row[0] for row in connection.execute("SELECT pair_id FROM matches")}
        geometries = {
            row[0] for row in connection.execute("SELECT pair_id FROM two_view_geometries")
        }
    assert matches == {_pair_id(1, 3), _pair_id(1, 4)}
    assert geometries == matches


def test_keeps_same_capture_cameras_when_their_angular_caps_overlap(tmp_path: Path):
    database = tmp_path / "database.db"
    _database(database, [(1, 2)])

    result = remove_impossible_same_capture_pairs(
        database,
        _spec(),
        _catalog(),
        _rig([0.7071067811865476, 0.0, 0.7071067811865475, 0.0]),
    )

    assert result == {
        "enabled": True,
        "examined_pairs": 1,
        "removed_pairs": 0,
        "camera_pairs": {},
    }
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 1
