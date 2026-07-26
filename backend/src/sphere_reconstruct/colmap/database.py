"""COLMAP データベース (SQLite) への書き込み.

ALIKED/LightGlue 経路で使う. COLMAP の feature_extractor を使わず, 自前で抽出した
keypoints を DB に書き, matches はテキスト match list にして `matches_importer` に
幾何検証させる (raw match type).

DB スキーマは COLMAP 4.1.1 (`database_creator` が作るもの) に準拠. スキーマ自体は
COLMAP に作らせ (rigs/frames 等の新テーブルを含むため), ここでは INSERT だけ行う.

blob フォーマット (COLMAP 慣習, little-endian, C order):
  cameras.params: float64[]
  keypoints.data: float32[rows, cols]  (cols=2 で (x, y))
  matches は matches_importer 経由なので DB へ直接書かない.

参考: COLMAP scripts/python/database.py のスキーマ / blob 規約.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

# COLMAP camera model id.
CAMERA_MODEL_PINHOLE = 1
CAMERA_MODEL_OPENCV_FISHEYE = 5

MAX_IMAGE_ID = 2**31 - 1


def pair_id_from_image_ids(image_id1: int, image_id2: int) -> int:
    """COLMAP の pair_id. image_id1 < image_id2 に正規化して合成する."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


class ColmapDatabase:
    """`database_creator` が作った DB へ camera/image/keypoints を書き込む薄いラッパ."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. create it first with `colmap database_creator`.")
        self._conn = sqlite3.connect(str(path))

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> ColmapDatabase:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_camera(
        self,
        model_id: int,
        width: int,
        height: int,
        params: list[float],
        *,
        prior_focal_length: bool = True,
        camera_id: int | None = None,
    ) -> int:
        params_blob = np.asarray(params, dtype=np.float64).tobytes()
        cur = self._conn.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model_id, width, height, params_blob, 1 if prior_focal_length else 0),
        )
        return cur.lastrowid

    def add_image(self, name: str, camera_id: int, *, image_id: int | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
            (image_id, name, camera_id),
        )
        return cur.lastrowid

    def add_keypoints(self, image_id: int, keypoints_xy: np.ndarray) -> None:
        """keypoints_xy: (N, 2) float の配列 (x, y). float32 で格納する."""
        kp = np.ascontiguousarray(keypoints_xy, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] < 2:
            raise ValueError(f"keypoints must be (N, >=2), got {kp.shape}")
        rows, cols = kp.shape
        self._conn.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, rows, cols, kp.tobytes()),
        )

    def commit(self) -> None:
        self._conn.commit()


def write_match_list(pairs: list[tuple[str, str, np.ndarray]], out_path: Path) -> None:
    """matches_importer (--match_type raw) 用のテキスト match list を書く.

    フォーマット:
        image_name1 image_name2
        idx1 idx2
        idx1 idx2
        <空行>
        image_name3 image_name4
        ...

    pairs: (name1, name2, matches) のリスト. matches は (M, 2) の keypoint index 対 (int).
    """
    lines: list[str] = []
    for name1, name2, matches in pairs:
        if matches is None or len(matches) == 0:
            continue
        lines.append(f"{name1} {name2}")
        for m in matches:
            lines.append(f"{int(m[0])} {int(m[1])}")
        lines.append("")  # 空行でペア区切り
    out_path.write_text("\n".join(lines), encoding="utf-8")
