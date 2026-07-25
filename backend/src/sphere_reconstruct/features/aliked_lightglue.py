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

デバイス方針:
  - 魚眼は 180deg+ を円内へ圧縮するため角分解能が元々低く, 縮小抽出すると暗所/弱テク
    スチャで特徴が消える. よって ALIKED は全解像度で抽出する.
  - ただし ALIKED は深層網で稠密な特徴マップを作るため, 8K 級 (3840^2) を GPU で流すと
    VRAM を使い切って OOM する. extraction_device="auto" は空き VRAM と画素数から
    GPU/CPU を選び, 実行時に OOM が出たら CPU へフォールバックして以降も CPU を使う.
  - LightGlue マッチングは疎な keypoint のみで軽いので config.device (既定 GPU) で動かす.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..settings import AlikedConfig

# ALIKED 抽出に必要な VRAM の粗い見積り (bytes / megapixel). 実測: 4070Ti(12GB) で
# 2880^2 = 8.3MP が空き ~10GB でも OOM したことから, 1MP あたり ~1.3GB を要すると見る.
_ALIKED_BYTES_PER_MP = 1.3 * 1024**3
# 空き VRAM に対する安全係数. 見積り * この係数 が空きを超えたら CPU にする.
_VRAM_SAFETY = 1.3


def _is_oom_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "out of memory" in s
        or "cuda_error_out_of_memory" in s
        or "cudaerrormemoryallocation" in s
        or "failed to allocate memory" in s
        or "cublas" in s and "alloc" in s
    )


@dataclass
class Features:
    keypoints: np.ndarray   # (N, 2) float32, 画素座標
    descriptors: np.ndarray  # (N, 128) float32
    scores: np.ndarray       # (N,) float32
    image_size: tuple[int, int]  # (width, height)


class AlikedLightGlue:
    """ALIKED 抽出 + LightGlue マッチングの ONNX ラッパ. Worker で 1 インスタンス.

    抽出器 (extractor) は最初の画像サイズと VRAM 状況を見て遅延生成し, OOM 時に CPU へ
    作り直す. マッチャ (matcher) は load() で config.device に生成する.
    """

    def __init__(self, config: AlikedConfig) -> None:
        self._config = config
        self._extractor = None
        self._matcher = None
        self._extractor_device: str | None = None  # 実際に使っている抽出デバイス.
        self._dll_dir_added = False
        # GPU OOM で CPU へ落ちたときの理由 (呼び出し側が Console 警告に使う). None なら未発生.
        self.cpu_fallback_reason: str | None = None

    @classmethod
    def from_settings(cls) -> AlikedLightGlue:
        from ..settings import get_settings

        cfg = get_settings().aliked
        if not cfg.is_configured():
            raise RuntimeError("aliked.extractor_path / matcher_path not configured")
        return cls(cfg)

    @property
    def extractor_device(self) -> str | None:
        """実際に使われた抽出デバイス ("cuda"/"cpu"). 抽出前は None."""
        return self._extractor_device

    def load(self) -> None:
        """マッチャを config.device に読み込む. 抽出器は最初の extract() で遅延生成する."""
        if self._matcher is not None:
            return
        self._ensure_dll_dir()
        mat = Path(self._config.matcher_path)
        if not mat.is_file():
            raise FileNotFoundError(f"LightGlue matcher not found: {mat}")
        self._matcher = self._make_session(mat, self._config.device)

    def _ensure_dll_dir(self) -> None:
        if not self._dll_dir_added:
            self._add_torch_cuda_dll_dir()
            self._dll_dir_added = True

    # -- session helpers ---------------------------------------------------------
    def _make_session(self, onnx_path: Path, device: str):
        import onnxruntime as ort  # noqa: PLC0415

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(str(onnx_path), providers=providers)

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

    @staticmethod
    def _free_vram_bytes() -> int | None:
        """空き VRAM (bytes). 取得できなければ None."""
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except Exception:
            return None

    def _decide_extractor_device(self, h: int, w: int) -> str:
        """extraction_device 設定 + VRAM 見積りから抽出デバイスを決める."""
        mode = self._config.extraction_device
        if mode == "cpu":
            return "cpu"
        if mode == "cuda":
            return "cuda"
        # auto: config.device が CPU 指定ならそのまま CPU.
        if not self._config.device.startswith("cuda"):
            return "cpu"
        mp = (h * w) / 1_000_000.0
        need = mp * _ALIKED_BYTES_PER_MP * _VRAM_SAFETY
        free = self._free_vram_bytes()
        if free is None:
            # VRAM を測れないときは画素数だけで判断 (~6MP 以上は CPU).
            return "cpu" if mp > 6.0 else "cuda"
        return "cuda" if need <= free else "cpu"

    def _ensure_extractor(self, h: int, w: int) -> None:
        if self._extractor is not None:
            return
        self._ensure_dll_dir()
        ext = Path(self._config.extractor_path)
        if not ext.is_file():
            raise FileNotFoundError(f"ALIKED extractor not found: {ext}")
        device = self._decide_extractor_device(h, w)
        self._extractor = self._make_session(ext, device)
        self._extractor_device = device

    def _rebuild_extractor_on_cpu(self) -> None:
        ext = Path(self._config.extractor_path)
        self._extractor = self._make_session(ext, "cpu")
        self._extractor_device = "cpu"

    def extract(self, image_rgb: np.ndarray) -> Features:
        """RGB (H,W,3) uint8 画像から ALIKED 特徴を抽出する.

        長辺が max_extract_size を超える画像は縮小して抽出する (8K 原寸は VRAM も RAM も
        溢れてネイティブクラッシュするため). keypoint は原寸座標へ戻す. 抽出デバイスは
        縮小後サイズで判定するので, 上限内なら GPU に載る. GPU OOM 時は CPU で再試行.
        """
        import cv2  # noqa: PLC0415

        h0, w0 = image_rgb.shape[:2]
        cap = self._config.max_extract_size
        if cap and max(h0, w0) > cap:
            scale = cap / max(h0, w0)
            proc = cv2.resize(image_rgb, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            proc, scale = image_rgb, 1.0
        h, w = proc.shape[:2]
        self._ensure_extractor(h, w)

        inp = proc.astype(np.float32) / 255.0
        inp = np.ascontiguousarray(np.transpose(inp, (2, 0, 1))[None, ...])
        feed = {
            "image": inp,
            "max_keypoints": np.array(self._config.max_keypoints, dtype=np.int64),
            "min_score": np.array(self._config.min_score, dtype=np.float32),
        }
        try:
            outs = self._extractor.run(None, feed)
        except Exception as exc:  # noqa: BLE001
            if self._extractor_device == "cuda" and _is_oom_error(exc):
                # GPU が足りない. CPU で作り直して 1 度だけ再試行. 以降の画像も CPU.
                self.cpu_fallback_reason = f"GPU OOM at {w}x{h} (extract_cap={self._config.max_extract_size or 'off'})"
                self._rebuild_extractor_on_cpu()
                outs = self._extractor.run(None, feed)
            else:
                raise
        kpts, desc, scores = outs[0], outs[1], outs[2]
        kpts = np.asarray(kpts).reshape(-1, 2).astype(np.float32)
        desc = np.asarray(desc).reshape(kpts.shape[0], -1).astype(np.float32)
        scores = np.asarray(scores).reshape(-1).astype(np.float32)
        if kpts.size and float(np.abs(kpts).max()) <= 1.5:
            # 正規化 [-1,1] の場合は縮小後画素へ戻す (実モデルは画素だが安全側で判定).
            kpts = (kpts + 1.0) * 0.5 * np.array([w, h], dtype=np.float32)
        if scale != 1.0:
            kpts = kpts / scale  # 原寸座標へ戻す.
        return Features(keypoints=kpts, descriptors=desc, scores=scores, image_size=(w0, h0))

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
        self._extractor_device = None
