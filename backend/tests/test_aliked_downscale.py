"""features.aliked_lightglue の縮小抽出 (max_extract_size) とデバイス判定の単体テスト.

8K 原寸は VRAM/RAM を溢れさせるため, 上限を超える画像は縮小して抽出し keypoint を
原寸座標へ戻す. onnxruntime は fake session で代替する.
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


def test_extract_downscales_and_rescales_keypoints_to_original():
    cfg = AlikedConfig(extractor_path="x", matcher_path="y", device="cpu", max_extract_size=256)
    eng = al.AlikedLightGlue(cfg)
    # 縮小後 (256) 空間の画素 keypoint. >1.5 なので正規化戻しは効かない.
    kpts = np.array([[128.0, 128.0], [0.0, 0.0]], dtype=np.float32)
    desc = np.zeros((2, 128), dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    eng._extractor = _FakeSession([kpts[None], desc[None], scores[None]])

    f = eng.extract(np.zeros((512, 512, 3), dtype=np.uint8))
    # scale = 256/512 = 0.5 -> keypoint は /scale = *2 で原寸へ.
    assert f.image_size == (512, 512)
    assert np.allclose(f.keypoints[0], [256.0, 256.0])
    assert np.allclose(f.keypoints[1], [0.0, 0.0])


def test_extract_no_downscale_when_within_cap():
    cfg = AlikedConfig(extractor_path="x", matcher_path="y", device="cpu", max_extract_size=2048)
    eng = al.AlikedLightGlue(cfg)
    kpts = np.array([[100.0, 200.0]], dtype=np.float32)
    eng._extractor = _FakeSession([kpts[None], np.zeros((1, 128), np.float32), np.array([0.9], np.float32)])
    f = eng.extract(np.zeros((480, 640, 3), dtype=np.uint8))
    assert f.image_size == (640, 480)
    assert np.allclose(f.keypoints, kpts)  # scale=1.0, 変化なし


def test_decide_device_cpu_forced():
    eng = al.AlikedLightGlue(AlikedConfig(extractor_path="x", matcher_path="y", extraction_device="cpu"))
    assert eng._decide_extractor_device(3840, 3840) == "cpu"


def test_decide_device_auto_large_without_vram_falls_to_cpu(monkeypatch):
    eng = al.AlikedLightGlue(
        AlikedConfig(extractor_path="x", matcher_path="y", extraction_device="auto", device="cuda:0")
    )
    monkeypatch.setattr(eng, "_free_vram_bytes", lambda: None)
    # VRAM 不明 + 14.7MP(>6) -> cpu.
    assert eng._decide_extractor_device(3840, 3840) == "cpu"
