"""colmap.database の書き込みテスト.

ローカルに colmap バイナリが無いので, COLMAP 4.1.1 の schema をテスト内で作って
書き込みを検証する.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

from sphere_reconstruct.colmap import database as db


# COLMAP database_creator が作る最小限のテーブル (テスト用).
_SCHEMA = """
CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
"""


def _make_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


def test_pair_id_normalizes_order():
    a = db.pair_id_from_image_ids(3, 7)
    b = db.pair_id_from_image_ids(7, 3)
    assert a == b
    assert a == 3 * db.MAX_IMAGE_ID + 7


def test_add_camera_image_keypoints(tmp_path: Path):
    p = tmp_path / "database.db"
    _make_db(p)
    with db.ColmapDatabase(p) as d:
        cam = d.add_camera(db.CAMERA_MODEL_PINHOLE, 512, 512, [256.0, 256.0, 256.0, 256.0])
        img = d.add_image("front_lens0/frame_000000.jpg", cam)
        kp = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        d.add_keypoints(img, kp)

    # 読み戻し.
    conn = sqlite3.connect(str(p))
    model, w, h, params_blob = conn.execute(
        "SELECT model, width, height, params FROM cameras WHERE camera_id=?", (cam,)
    ).fetchone()
    assert model == db.CAMERA_MODEL_PINHOLE
    assert w == 512 and h == 512
    params = np.frombuffer(params_blob, dtype=np.float64)
    assert list(params) == [256.0, 256.0, 256.0, 256.0]

    rows, cols, data = conn.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (img,)
    ).fetchone()
    assert rows == 3 and cols == 2
    kp_read = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    assert np.allclose(kp_read, kp)
    conn.close()


def test_write_match_list(tmp_path: Path):
    pairs = [
        ("a.jpg", "b.jpg", np.array([[0, 1], [2, 3]])),
        ("a.jpg", "c.jpg", np.array([[5, 6]])),
        ("x.jpg", "y.jpg", np.empty((0, 2))),  # 空はスキップ
    ]
    out = tmp_path / "matches.txt"
    db.write_match_list(pairs, out)
    text = out.read_text()
    assert "a.jpg b.jpg" in text
    assert "0 1" in text
    assert "2 3" in text
    assert "a.jpg c.jpg" in text
    assert "5 6" in text
    # 空ペアは書かれない.
    assert "x.jpg y.jpg" not in text
