"""FastDVDnet の固定タイル推論ラッパー."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

MIN_TILE_OVERLAP = 64
DEFAULT_TILE_OVERLAP = 80


class FastDvdnetEngine:
    def __init__(self, weights_path: Path, device: str = "auto") -> None:
        self._weights_path = weights_path
        self._requested_device = device
        self._torch = None
        self._model = None
        self.device = ""

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "FastDVDnet には denoise extra が必要です: `uv sync --extra denoise`"
            ) from error

        from .model import FastDVDnet  # noqa: PLC0415

        requested = self._requested_device.lower()
        device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"FastDVDnet device は '{device}' ですが CUDA を利用できません")

        state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        state = _strip_data_parallel_prefix(state)
        model = FastDVDnet(num_input_frames=5)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._torch = torch
        self._model = model
        self.device = device

    def denoise(
        self,
        frames_rgb: list[np.ndarray],
        *,
        sigma: float,
        tile_size: int,
        tile_overlap: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if len(frames_rgb) != 5:
            raise ValueError("FastDVDnet には 5 frame が必要です")
        shape = frames_rgb[0].shape
        if len(shape) != 3 or shape[2] != 3 or any(frame.shape != shape for frame in frames_rgb):
            raise ValueError("FastDVDnet の全 frame は同じ HxWx3 寸法でなければなりません")
        if any(frame.dtype != np.uint8 for frame in frames_rgb):
            raise ValueError("FastDVDnet frame は uint8 RGB でなければなりません")
        if sigma < 0 or sigma > 55:
            raise ValueError("FastDVDnet sigma は [0, 55] の範囲で指定してください")
        if (
            tile_size < 128
            or tile_overlap < MIN_TILE_OVERLAP
            or tile_overlap * 2 >= tile_size
        ):
            raise ValueError("FastDVDnet タイルの寸法が不正です")
        self.load()
        assert self._torch is not None and self._model is not None

        height, width = shape[:2]
        result = np.empty_like(frames_rgb[2])
        torch = self._torch
        total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
        completed_tiles = 0
        with torch.inference_mode():
            for top in range(0, height, tile_size):
                bottom = min(height, top + tile_size)
                input_top = max(0, top - tile_overlap)
                input_bottom = min(height, bottom + tile_overlap)
                for left in range(0, width, tile_size):
                    right = min(width, left + tile_size)
                    input_left = max(0, left - tile_overlap)
                    input_right = min(width, right + tile_overlap)
                    tiles = [
                        frame[input_top:input_bottom, input_left:input_right]
                        for frame in frames_rgb
                    ]
                    denoised = self._infer_tile(tiles, sigma)
                    crop_top = top - input_top
                    crop_left = left - input_left
                    result[top:bottom, left:right] = denoised[
                        crop_top : crop_top + bottom - top,
                        crop_left : crop_left + right - left,
                    ]
                    completed_tiles += 1
                    if progress is not None:
                        progress(completed_tiles, total_tiles)
        return result

    def _infer_tile(self, frames_rgb: list[np.ndarray], sigma: float) -> np.ndarray:
        torch = self._torch
        assert torch is not None and self._model is not None
        import torch.nn.functional as functional  # noqa: PLC0415

        packed = np.concatenate(frames_rgb, axis=2).transpose(2, 0, 1)
        value = torch.from_numpy(np.ascontiguousarray(packed)).unsqueeze(0)
        value = value.to(self.device, dtype=torch.float32).div_(255.0)
        height, width = value.shape[-2:]
        pad_bottom = (-height) % 4
        pad_right = (-width) % 4
        if pad_bottom or pad_right:
            value = functional.pad(value, (0, pad_right, 0, pad_bottom), mode="reflect")
        noise_map = torch.full(
            (1, 1, value.shape[-2], value.shape[-1]),
            sigma / 255.0,
            dtype=torch.float32,
            device=self.device,
        )
        output = self._model(value, noise_map).clamp_(0.0, 1.0)
        output = output[..., :height, :width]
        array = output.squeeze(0).permute(1, 2, 0).mul_(255.0).round_().byte().cpu().numpy()
        return array


def _strip_data_parallel_prefix(state: dict) -> dict:
    """Upstream checkpoint の ``nn.DataParallel`` wrapper 名を単一 GPU model へ戻す."""
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state
