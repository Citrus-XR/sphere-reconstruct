"""RoMaV2 を heavy worker 内だけで読み込む dense matcher adapter。"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DenseMatches:
    pixels_a: np.ndarray
    pixels_b: np.ndarray
    confidence: np.ndarray


class RomaV2Matcher:
    def __init__(self, *, setting: str, seed: int = 0):
        repository_root = Path(__file__).resolve().parents[4]
        os.environ.setdefault("TORCH_HOME", str(repository_root / ".runtime" / "torch"))
        import torch
        from romav2 import RoMaV2

        torch.manual_seed(seed)
        self._torch = torch
        self._model = RoMaV2(RoMaV2.Cfg(compile=False))
        self._model.apply_setting(setting)
        self._model.eval()

    def match(self, image_a: Path, image_b: Path, *, count: int) -> DenseMatches:
        with Image.open(image_a) as source_a, Image.open(image_b) as source_b:
            rgb_a = source_a.convert("RGB")
            rgb_b = source_b.convert("RGB")
            width_a, height_a = rgb_a.size
            width_b, height_b = rgb_b.size
            predictions = self._model.match(rgb_a, rgb_b)
        matches, confidence, _, _ = self._model.sample(predictions, count)
        pixels_a, pixels_b = self._model.to_pixel_coordinates(
            matches,
            H_A=height_a,
            W_A=width_a,
            H_B=height_b,
            W_B=width_b,
        )
        result = DenseMatches(
            pixels_a=pixels_a.detach().cpu().numpy().astype(np.float64),
            pixels_b=pixels_b.detach().cpu().numpy().astype(np.float64),
            confidence=confidence.detach().cpu().numpy().astype(np.float64),
        )
        del predictions, matches, confidence, pixels_a, pixels_b
        return result

    def close(self) -> None:
        model = self._model
        self._model = None
        if model is not None:
            model.to("cpu")
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
