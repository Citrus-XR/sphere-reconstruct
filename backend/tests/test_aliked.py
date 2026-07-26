"""features.aliked_lightglue のロジックテスト (onnxruntime を mock).

match() の matches0 -> index 対変換, extract() の keypoint 正規化戻しを検証する.
実 ONNX は使わず, .run() が固定配列を返す fake session を注入する.
"""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.features import aliked_lightglue as al
from sphere_reconstruct.settings import AlikedConfig


class _FakeSession:
    def __init__(self, outputs):
        self._outputs = outputs

    def run(self, _out_names, _inputs):
        return self._outputs


def _engine():
    cfg = AlikedConfig(extractor_path="x", matcher_path="y", device="cpu")
    return al.AlikedLightGlue(cfg)


def test_extract_pixel_keypoints_passthrough():
    eng = _engine()
    # keypoints 既に画素座標 (大きな値) -> そのまま.
    kpts = np.array([[100.0, 200.0], [300.0, 50.0]], dtype=np.float32)
    desc = np.zeros((2, 128), dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    eng._extractor = _FakeSession([kpts[None], desc[None], scores[None]])
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    f = eng.extract(img)
    assert f.image_size == (640, 480)
    assert np.allclose(f.keypoints, kpts)
    assert f.descriptors.shape == (2, 128)


def test_extract_denormalizes_normalized_keypoints():
    eng = _engine()
    # keypoints が [-1,1] 正規化 -> 画素へ戻す.
    kpts = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)  # center, and (right, top)
    desc = np.zeros((2, 128), dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    eng._extractor = _FakeSession([kpts[None], desc[None], scores[None]])
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    f = eng.extract(img)
    # (0,0) -> center (320, 240); (1,-1) -> (640, 0)
    assert np.allclose(f.keypoints[0], [320.0, 240.0])
    assert np.allclose(f.keypoints[1], [640.0, 0.0])


def test_match_assignment_to_pairs():
    eng = _engine()
    # matches0[i] = j (or -1). i=0->2, i=1->-1, i=2->0
    matches0 = np.array([2, -1, 0], dtype=np.int64)
    mscores = np.array([0.9, 0.0, 0.7], dtype=np.float32)
    eng._matcher = _FakeSession([matches0, mscores])
    f0 = al.Features(
        np.zeros((3, 2), np.float32), np.zeros((3, 128), np.float32), np.zeros(3, np.float32), (640, 480)
    )
    f1 = al.Features(
        np.zeros((3, 2), np.float32), np.zeros((3, 128), np.float32), np.zeros(3, np.float32), (640, 480)
    )
    m = eng.match(f0, f1)
    # 有効対: (0,2), (2,0). i=1 は -1 で除外.
    assert m.shape == (2, 2)
    assert list(m[0]) == [0, 2]
    assert list(m[1]) == [2, 0]


def test_match_empty_features():
    eng = _engine()
    f0 = al.Features(
        np.empty((0, 2), np.float32), np.empty((0, 128), np.float32), np.empty(0, np.float32), (640, 480)
    )
    f1 = al.Features(
        np.zeros((3, 2), np.float32), np.zeros((3, 128), np.float32), np.zeros(3, np.float32), (640, 480)
    )
    m = eng.match(f0, f1)
    assert m.shape == (0, 2)
