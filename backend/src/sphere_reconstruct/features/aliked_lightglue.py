"""ALIKED + LightGlue の ONNX Runtime 推論 (Worker 専用).

ONNX 署名 (実モデルで確認済み):

ALIKED (aliked-n16rot.onnx):
  in : image [1,3,H,W] (RGB, 0..1), max_keypoints (scalar i64), min_score (scalar f32)
  out: keypoints [1,N,2] (入力画像の画素座標), descriptors [1,N,128], scores [1,N]

LightGlue (aliked-lightglue.onnx):
  in : kpts0 [1,N0,2], kpts1 [1,N1,2], desc0 [1,N0,128], desc1 [1,N1,128],
       image_size0 [1,2], image_size1 [1,2]
  out: matches0 [N0] (matches0[i]=j なら kpt0[i] が kpt1[j] に対応, -1 は未対応),
       mscores0 [N0]

onnxruntime は device に応じて CUDA/CPU provider を選ぶ. torch は使わない.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..settings import AlikedConfig


@dataclass
class Features:
    keypoints: np.ndarray   # (N, 2) float32, 画素座標
    descriptors: np.ndarray  # (N, 128) float32
    scores: np.ndarray       # (N,) float32
    image_size: tuple[int, int]  # (width, height)


class AlikedLightGlue:
    """ALIKED 抽出 + LightGlue マッチングの ONNX ラッパ. Worker で 1 インスタンス."""

    def __init__(self, config: AlikedConfig) -> None:
        self._config = config
        self._extractor = None
        self._matcher = None

    @classmethod
    def from_settings(cls) -> "AlikedLightGlue":
        from ..settings import get_settings

        cfg = get_settings().aliked
        if not cfg.is_configured():
            raise RuntimeError("aliked.extractor_path / matcher_path not configured")
        return cls(cfg)

    def load(self) -> None:
        if self._extractor is not None:
            return

        # Windows + CUDA: onnxruntime は CUDA/cuDNN の DLL を探索パスから探す. torch
        # (cu128) が同梱する CUDA 12.8 ランタイム DLL を使えるよう, torch の lib を
        # DLL 探索ディレクトリに足す (別途 CUDA インストール不要). 失敗しても CPU で動く.
        self._add_torch_cuda_dll_dir()

        import onnxruntime as ort  # noqa: PLC0415

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._config.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        ext = Path(self._config.extractor_path)
        mat = Path(self._config.matcher_path)
        if not ext.is_file():
            raise FileNotFoundError(f"ALIKED extractor not found: {ext}")
        if not mat.is_file():
            raise FileNotFoundError(f"LightGlue matcher not found: {mat}")
        self._extractor = ort.InferenceSession(str(ext), providers=providers)
        self._matcher = ort.InferenceSession(str(mat), providers=providers)

    @staticmethod
    def _add_torch_cuda_dll_dir() -> None:
        """torch 同梱の CUDA DLL を Windows の DLL 探索へ追加 (best-effort)."""
        import os
        import sys

        if sys.platform != "win32":
            return
        try:
            import torch  # noqa: PLC0415

            lib = Path(torch.__file__).parent / "lib"
            if lib.is_dir():
                os.add_dll_directory(str(lib))
        except Exception:
            # torch が無い / 失敗しても CPU provider で動作する.
            pass

    def extract(self, image_rgb: np.ndarray) -> Features:
        """RGB (H,W,3) uint8 画像から ALIKED 特徴を抽出する."""
        if self._extractor is None:
            self.load()
        assert self._extractor is not None

        h, w = image_rgb.shape[:2]
        # (1,3,H,W) float32 0..1.
        inp = image_rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        inp = np.ascontiguousarray(inp)

        outs = self._extractor.run(
            None,
            {
                "image": inp,
                "max_keypoints": np.array(self._config.max_keypoints, dtype=np.int64),
                "min_score": np.array(self._config.min_score, dtype=np.float32),
            },
        )
        kpts, desc, scores = outs[0], outs[1], outs[2]
        kpts = np.asarray(kpts).reshape(-1, 2).astype(np.float32)
        desc = np.asarray(desc).reshape(kpts.shape[0], -1).astype(np.float32)
        scores = np.asarray(scores).reshape(-1).astype(np.float32)
        # keypoints が正規化 [-1,1] の場合は画素へ戻す (実モデルは画素だが安全側で判定).
        if kpts.size and float(np.abs(kpts).max()) <= 1.5:
            kpts = (kpts + 1.0) * 0.5 * np.array([w, h], dtype=np.float32)
        return Features(keypoints=kpts, descriptors=desc, scores=scores, image_size=(w, h))

    def match(self, f0: Features, f1: Features) -> np.ndarray:
        """2 画像の特徴をマッチングし, (M, 2) の keypoint index 対を返す."""
        if f0.keypoints.shape[0] == 0 or f1.keypoints.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64)
        if self._matcher is None:
            self.load()
        assert self._matcher is not None

        outs = self._matcher.run(
            None,
            {
                "kpts0": f0.keypoints[None, ...],
                "kpts1": f1.keypoints[None, ...],
                "desc0": f0.descriptors[None, ...],
                "desc1": f1.descriptors[None, ...],
                "image_size0": np.array([[f0.image_size[0], f0.image_size[1]]], dtype=np.float32),
                "image_size1": np.array([[f1.image_size[0], f1.image_size[1]]], dtype=np.float32),
            },
        )
        matches0 = np.asarray(outs[0]).reshape(-1)  # matches0[i] = j or -1
        idx0 = np.nonzero(matches0 >= 0)[0]
        idx1 = matches0[idx0]
        return np.stack([idx0, idx1], axis=1).astype(np.int64)

    def unload(self) -> None:
        self._extractor = None
        self._matcher = None
