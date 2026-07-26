"""時系列 denoise stage の境界処理と parameter contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sphere_reconstruct.denoise.engine import FastDvdnetEngine, _strip_data_parallel_prefix
from sphere_reconstruct.stages.denoise_frames import (
    DenoiseFrames,
    _frame_count,
    _reflect_index,
    _run_fastdvdnet_stream,
)


def test_reflect_index_matches_five_frame_boundary_window():
    assert [_reflect_index(offset, 6) for offset in (-2, -1, 0, 1, 2)] == [2, 1, 0, 1, 2]
    assert [_reflect_index(5 + offset, 6) for offset in (-2, -1, 0, 1, 2)] == [3, 4, 5, 4, 3]


def test_denoise_parameters_are_deterministic():
    params = DenoiseFrames().normalize_params({"method": "fastdvdnet"})
    assert params["sigma"] == 10.0
    assert params["tile_size"] == 512
    assert params["tile_overlap"] == 80
    assert params["requested_device"] == "auto"
    assert params["hardware_decode"] == "auto"
    assert len(params["model_sha256"]) == 64


def test_upstream_data_parallel_checkpoint_keys_are_normalized():
    state = {"module.temp1.weight": object(), "module.temp2.bias": object()}
    normalized = _strip_data_parallel_prefix(state)
    assert set(normalized) == {"temp1.weight", "temp2.bias"}


@pytest.mark.parametrize(
    "params",
    [
        {"method": "unknown"},
        {"method": "fastdvdnet", "sigma": 56},
        {"method": "fastdvdnet", "tile_size": 128, "tile_overlap": 64},
        {"method": "fastdvdnet", "tile_size": 512, "tile_overlap": 63},
        {"method": "ffmpeg_adaptive", "temporal_window": 4},
    ],
)
def test_invalid_denoise_parameters_fail(params):
    with pytest.raises(ValueError):
        DenoiseFrames().normalize_params(params)


def test_frame_count_never_reflects_a_selected_center():
    stream = SimpleNamespace(nb_frames=None, fps=10.0)
    frames = [{"source_frame": 20}]
    assert _frame_count(stream, 1.9, frames) == 21


def test_fastdvdnet_stream_uses_exact_source_rate_windows(tmp_path, monkeypatch):
    frames = [
        {"index": 0, "source_frame": 0},
        {"index": 1, "source_frame": 3},
        {"index": 2, "source_frame": 9},
    ]
    windows: list[list[int]] = []

    class FakeEngine:
        def denoise(self, values, **kwargs):
            windows.append([int(value[0, 0, 0]) for value in values])
            kwargs["progress"](1, 1)
            return values[2]

    def fake_frames(_source, *, frame_indices, progress, **_kwargs):
        for current, index in enumerate(frame_indices, 1):
            progress(current, len(frame_indices))
            yield index, np.full((8, 8, 3), index, dtype=np.uint8)

    monkeypatch.setattr(
        "sphere_reconstruct.stages.denoise_frames.ffmpeg.iter_selected_rgb_frames",
        fake_frames,
    )
    completed: list[bool] = []
    details: list[float] = []
    results = _run_fastdvdnet_stream(
        tmp_path / "source.mp4",
        stream_index=0,
        frames=frames,
        frame_count=10,
        width=8,
        height=8,
        output_dir=tmp_path / "output",
        engine=FakeEngine(),
        sigma=10,
        tile_size=512,
        tile_overlap=80,
        jpeg_quality=98,
        ffmpeg_bin=None,
        scratch_dir=tmp_path / "scratch",
        hardware_decode="none",
        pixel_format="yuv420p",
        on_frame=lambda: completed.append(True),
        on_detail=details.append,
    )
    assert windows == [[2, 1, 0, 1, 2], [1, 2, 3, 4, 5], [7, 8, 9, 8, 7]]
    assert len(results) == len(completed) == 3
    assert details


def test_tile_halo_matches_full_receptive_field():
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    class Radius74Model(torch.nn.Module):
        def forward(self, value, _noise_map):
            result = value[:, 6:9]
            for _ in range(74):
                result = functional.avg_pool2d(result, kernel_size=3, stride=1, padding=1)
            return result

    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(5)]
    engine = FastDvdnetEngine(Path("unused"), device="cpu")
    engine._torch = torch
    engine._model = Radius74Model().eval()
    engine.device = "cpu"
    full = engine._infer_tile(frames, 10)
    tiled = engine.denoise(frames, sigma=10, tile_size=256, tile_overlap=80)
    assert np.array_equal(tiled, full)
