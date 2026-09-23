import sqlite3

import pytest

from sphere_reconstruct.colmap.camera_policy import apply_camera_policies


def _database(path, images):
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, params BLOB, prior_focal_length INTEGER);"
            "CREATE TABLE images (name TEXT UNIQUE, camera_id INTEGER);"
        )
        connection.executemany(
            "INSERT INTO cameras VALUES (?, ?, 1)",
            [(camera_id, b"unchanged intrinsics") for camera_id in sorted({value for _, value in images})],
        )
        connection.executemany("INSERT INTO images VALUES (?, ?)", images)


def _catalog(groups):
    return {
        "images": [{"name": name} for group in groups for name in group["image_names"]],
        "camera_groups": groups,
    }


def _group(group_id, names, *, prior, refine):
    return {
        "id": group_id,
        "image_names": names,
        "has_prior_focal_length": prior,
        "refine_intrinsics": refine,
    }


def test_camera_policies_preserve_intrinsics_and_distinguish_guesses_from_priors(tmp_path):
    database_path = tmp_path / "database.db"
    _database(database_path, [("fish0", 9), ("fish1", 2), ("guessed", 4), ("exif", 7)])
    catalog = _catalog([
        _group("fish", ["fish0", "fish1"], prior=True, refine=False),
        _group("phone-video", ["guessed"], prior=False, refine=True),
        _group("photo", ["exif"], prior=True, refine=True),
    ])

    frozen = apply_camera_policies(database_path, catalog)

    assert frozen == [2, 9]
    with sqlite3.connect(database_path) as connection:
        cameras = connection.execute(
            "SELECT camera_id, params, prior_focal_length FROM cameras ORDER BY camera_id"
        ).fetchall()
    assert cameras == [
        (2, b"unchanged intrinsics", 1),
        (4, b"unchanged intrinsics", 0),
        (7, b"unchanged intrinsics", 1),
        (9, b"unchanged intrinsics", 1),
    ]


@pytest.mark.parametrize("prior,refine", [(True, True), (False, False)])
def test_conflicting_shared_camera_policy_fails_without_partial_updates(tmp_path, prior, refine):
    database_path = tmp_path / "database.db"
    _database(database_path, [("guessed", 1), ("first", 2), ("second", 2)])
    catalog = _catalog([
        _group("guess", ["guessed"], prior=False, refine=True),
        _group("first", ["first"], prior=True, refine=False),
        _group("second", ["second"], prior=prior, refine=refine),
    ])

    with pytest.raises(ValueError, match="camera 2 has conflicting calibration policies"):
        apply_camera_policies(database_path, catalog)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT prior_focal_length FROM cameras").fetchall() == [(1,), (1,)]


def test_shared_camera_with_identical_policies_is_allowed(tmp_path):
    database_path = tmp_path / "database.db"
    _database(database_path, [("first", 2), ("second", 2)])
    catalog = _catalog([
        _group("first", ["first"], prior=True, refine=False),
        _group("second", ["second"], prior=True, refine=False),
    ])

    assert apply_camera_policies(database_path, catalog) == [2]


@pytest.mark.parametrize("image_names", [["missing"], ["present", "unexpected"]])
def test_database_image_coverage_must_match_catalog(tmp_path, image_names):
    database_path = tmp_path / "database.db"
    _database(database_path, [("present", 1)])
    catalog = _catalog([_group("phone", image_names, prior=False, refine=True)])

    with pytest.raises(ValueError, match="COLMAP database does not match image catalog"):
        apply_camera_policies(database_path, catalog)


def test_group_image_coverage_must_match_catalog(tmp_path):
    database_path = tmp_path / "database.db"
    _database(database_path, [("present", 1)])
    catalog = _catalog([_group("phone", [], prior=False, refine=True)])
    catalog["images"] = [{"name": "present"}]

    with pytest.raises(ValueError, match="camera groups does not match image catalog"):
        apply_camera_policies(database_path, catalog)


def test_image_cannot_belong_to_two_groups(tmp_path):
    database_path = tmp_path / "database.db"
    _database(database_path, [("present", 1)])
    catalog = _catalog([
        _group("first", ["present"], prior=False, refine=True),
        _group("second", ["present"], prior=False, refine=True),
    ])

    with pytest.raises(ValueError, match="belongs to multiple camera groups"):
        apply_camera_policies(database_path, catalog)


def test_missing_camera_rolls_back_policy_updates(tmp_path):
    database_path = tmp_path / "database.db"
    _database(database_path, [("first", 1), ("second", 2)])
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM cameras WHERE camera_id = 2")
    catalog = _catalog([_group("phone", ["first", "second"], prior=False, refine=True)])

    with pytest.raises(ValueError, match="missing camera 2"):
        apply_camera_policies(database_path, catalog)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT prior_focal_length FROM cameras").fetchall() == [(1,)]


def test_missing_database_is_not_created(tmp_path):
    database_path = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError):
        apply_camera_policies(database_path, _catalog([]))

    assert not database_path.exists()
