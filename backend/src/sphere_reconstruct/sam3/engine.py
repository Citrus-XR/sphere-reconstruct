"""SAM3 推論エンジン (Worker プロセス専用).

torch と sam3 パッケージをここで初めて import する. FastAPI プロセスからは絶対に
触らない. モデルロードは数秒〜十数秒かかるため, Worker プロセスの寿命内で 1 回だけ
行い, 複数フレームで使い回す.

手動配置方式: settings.sam3.repo_path を sys.path に足して import する. HuggingFace
token / 自動 DL は使わない (load_from_HF=False).

推論フロー (1 画像あたり):
  1. set_image(pil)          -> backbone を 1 回 forward
  2. 各 prompt term について set_text_prompt(term) -> grounding pass
  3. term ごとの mask を集め, 呼び出し側で union する

縮小は呼び出し側の mask Step が行い, ここには縮小済み画像が渡る前提.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from .settings import Sam3Paths, resolve


@dataclass
class Sam3Detection:
    """1 つの prompt term に対する検出結果."""

    prompt: str
    masks: list[np.ndarray] = field(default_factory=list)  # 各 instance の 2 値 mask (H,W)
    scores: list[float] = field(default_factory=list)


class Sam3Engine:
    """SAM3 image model のラッパ. Worker プロセスで 1 インスタンス."""

    def __init__(self, paths: Sam3Paths, confidence_threshold: float = 0.5) -> None:
        self._paths = paths
        self._confidence = confidence_threshold
        self._model = None
        self._processor = None
        self._torch = None
        self._autocast_dtype = None  # cuda + bf16/fp16 のとき torch.dtype, それ以外 None

    @classmethod
    def from_settings(cls) -> Sam3Engine:
        paths = resolve()
        if paths is None:
            raise RuntimeError("SAM3 paths not configured (sam3.repo_path / checkpoint_path)")
        from ..settings import get_settings

        return cls(paths, confidence_threshold=get_settings().sam3.confidence_threshold)

    def load(self) -> None:
        """モデルをメモリに載せる. 冪等 (2 回目以降は no-op)."""
        if self._model is not None:
            return

        # repo_path を sys.path 先頭へ (sam3 パッケージを import 可能にする).
        repo = str(self._paths.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import torch  # noqa: PLC0415

        self._torch = torch

        from sam3 import build_sam3_image_model  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415

        device = self._paths.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"SAM3 device is '{device}' but CUDA is not available on this machine")

        # SAM3 の build_sam3_image_model 内 _setup_device_and_mode は device == "cuda"
        # の完全一致でしか model.cuda() しない ("cuda:0" だと重みが CPU に残り, 入力
        # (cuda) と型が食い違う). そのため cuda 系は "cuda" へ正規化する.
        # 4070Ti など単一 GPU 前提. マルチ GPU で特定 index を使う要件が出たら,
        # build 後に手動で .to(device) する対応を足す.
        build_device = "cuda" if device.startswith("cuda") else device

        # Ampere 系で tf32 を有効化 (公式 demo に倣う). cpu では無害.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._model = build_sam3_image_model(
            checkpoint_path=str(self._paths.checkpoint_path),
            load_from_HF=False,
            device=build_device,
        )
        self._processor = Sam3Processor(
            self._model, confidence_threshold=self._confidence, device=build_device
        )

        # cuda では autocast で bf16/fp16 を使う (VRAM 節約 + 高速化). cpu は fp32 のまま.
        if build_device == "cuda":
            dt = self._paths.dtype.lower()
            if dt in ("bfloat16", "bf16"):
                self._autocast_dtype = torch.bfloat16
            elif dt in ("float16", "fp16", "half"):
                self._autocast_dtype = torch.float16

    def detect(self, image_rgb: np.ndarray, prompts: list[str]) -> list[Sam3Detection]:
        """RGB (H,W,3) uint8 画像に対し, prompt 各語で検出する.

        戻り値は prompt ごとの Sam3Detection. set_image は 1 回のみ実行し, 各 prompt
        で set_text_prompt を呼んで backbone を使い回す.
        """
        if self._processor is None:
            self.load()
        assert self._processor is not None

        from PIL import Image  # noqa: PLC0415

        pil = Image.fromarray(image_rgb)

        import contextlib  # noqa: PLC0415

        if self._autocast_dtype is not None:
            autocast_ctx = self._torch.autocast("cuda", dtype=self._autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            state = self._processor.set_image(pil)

            results: list[Sam3Detection] = []
            for term in prompts:
                self._processor.reset_all_prompts(state)
                state = self._processor.set_text_prompt(prompt=term, state=state)
                det = Sam3Detection(prompt=term)
                masks = state.get("masks")
                scores = state.get("scores")
                if masks is not None and len(masks) > 0:
                    # masks: (N,1,H,W) bool tensor at original resolution.
                    arr = masks.squeeze(1).detach().cpu().numpy().astype(np.uint8)
                    det.masks = [arr[i] for i in range(arr.shape[0])]
                    if scores is not None:
                        # autocast 下では bf16 になり numpy が非対応なので float32 化する.
                        sc = scores.detach().float().cpu().numpy().tolist()
                        det.scores = [float(s) for s in sc]
                results.append(det)
        return results

    def unload(self) -> None:
        """VRAM 解放. Worker プロセス終了時に呼ぶ (プロセス kill でも OS が回収する)."""
        self._model = None
        self._processor = None
        if self._torch is not None and self._paths.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
