"""COLMAP 詳細パラメータ → 各サブコマンドの extra_args へのコンパイル.

UI の「品質プリセット」はフロントで具体値に展開して送る (プラグイン同様, プリセット選択で
スライダ値が snap する). ここは具体値だけを受け取り, 0/未設定 は「COLMAP 既定」= flag を出さない.

- SIFT 特徴 / SIFT マッチのフラグは backend=sift のときだけ有効 (ALIKED は自前抽出/マッチ).
- Mapper/BA のフラグは両 backend 共通 (COLMAP mapper は常に走る).
- two_view min_num_inliers は SIFT ではマッチャの幾何検証, ALIKED では matches_importer で使う.
"""

from __future__ import annotations


def _pos(v) -> bool:
    """設定済み (正の数) か. 0/None/負 は「未設定 = COLMAP 既定」扱い."""
    return v is not None and v > 0


def _num(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


# (params キー, COLMAP flag, 種別) — Mapper/BA は両 backend 共通.
_MAPPER_KEYS = [
    ("mapper_min_num_matches", "--Mapper.min_num_matches"),
    ("init_min_num_inliers", "--Mapper.init_min_num_inliers"),
    ("abs_pose_max_error", "--Mapper.abs_pose_max_error"),
    ("filter_max_reproj_error", "--Mapper.filter_max_reproj_error"),
    ("filter_min_tri_angle", "--Mapper.filter_min_tri_angle"),
    ("ba_local_max_num_iterations", "--Mapper.ba_local_max_num_iterations"),
    ("ba_global_max_num_iterations", "--Mapper.ba_global_max_num_iterations"),
    ("min_model_size", "--Mapper.min_model_size"),
]


def feature_extra_args(p: dict, backend: str) -> list[str]:
    if backend != "sift":
        return []
    out: list[str] = []
    if _pos(p.get("sift_max_num_features")):
        out += ["--SiftExtraction.max_num_features", _num(p["sift_max_num_features"])]
    if _pos(p.get("sift_max_image_size")):
        out += ["--SiftExtraction.max_image_size", _num(p["sift_max_image_size"])]
    if _pos(p.get("sift_peak_threshold")):
        out += ["--SiftExtraction.peak_threshold", _num(p["sift_peak_threshold"])]
    if _pos(p.get("sift_edge_threshold")):
        out += ["--SiftExtraction.edge_threshold", _num(p["sift_edge_threshold"])]
    if p.get("sift_affine_dsp"):
        # affine 形状推定 + domain size pooling はセットで. GPU SIFT を迂回し CPU で重くなる.
        out += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
    return out


def matcher_extra_args(p: dict, backend: str) -> list[str]:
    if backend != "sift":
        return []
    out: list[str] = []
    if _pos(p.get("max_num_matches")):
        out += ["--SiftMatching.max_num_matches", _num(p["max_num_matches"])]
    if p.get("guided_matching"):
        out += ["--SiftMatching.guided_matching", "1"]
    if _pos(p.get("two_view_min_num_inliers")):
        out += ["--TwoViewGeometry.min_num_inliers", _num(p["two_view_min_num_inliers"])]
    return out


def mapper_extra_args(p: dict) -> list[str]:
    out: list[str] = []
    for key, flag in _MAPPER_KEYS:
        if _pos(p.get(key)):
            out += [flag, _num(p[key])]
    if p.get("ba_use_gpu"):
        # GPU バンドル調整 (PBA). COLMAP CLI で BA を GPU に載せる (plugin の "Ceres GPU" 相当).
        out += ["--Mapper.ba_use_gpu", "1"]
    return out
