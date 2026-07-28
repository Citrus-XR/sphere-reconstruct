"""COLMAP runtime metadata から Ceres CUDA / cuDSS capability を判定する。"""

from __future__ import annotations

import json

from sphere_reconstruct.diagnostics import (
    _bundle_adjustment_capabilities,
    _parse_nvidia_smi_inventory,
    _pinned_cuda_ba_compatible,
)


def test_pinned_runtime_metadata_enables_sparse_gpu_bundle_adjustment(tmp_path):
    runtime = tmp_path / "colmap"
    binary = runtime / "bin" / "colmap.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (runtime / "sphere-colmap-capabilities.json").write_text(
        json.dumps(
            {
                "cudss_version": "0.8.0.10",
                "gpu_bundle_adjustment_dense": True,
                "gpu_bundle_adjustment_sparse": True,
            }
        ),
        encoding="utf-8",
    )

    capabilities = _bundle_adjustment_capabilities(binary)

    assert capabilities["dense"] is True
    assert capabilities["sparse"] is True
    assert capabilities["cudss"] == "0.8.0.10"


def test_cpu_only_ceres_marker_disables_gpu_bundle_adjustment(tmp_path):
    binary = tmp_path / "bin" / "colmap.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (binary.parent / "ceres.dll").write_bytes(
        b"CUDA Ceres was compiled without support for CUDA. CUDA_SPARSE"
    )

    capabilities = _bundle_adjustment_capabilities(binary)

    assert capabilities["dense"] is False
    assert capabilities["sparse"] is False


def test_nvidia_inventory_exposes_pinned_cuda_ba_requirements():
    inventory = _parse_nvidia_smi_inventory(
        "NVIDIA GeForce RTX 4070 Ti, 591.86, 8.9\nNVIDIA GeForce GTX 1080, 576.88, 6.1\nmalformed"
    )

    assert inventory == [
        {
            "name": "NVIDIA GeForce RTX 4070 Ti",
            "driver_version": "591.86",
            "driver_major": 591,
            "compute_capability": 8.9,
        },
        {
            "name": "NVIDIA GeForce GTX 1080",
            "driver_version": "576.88",
            "driver_major": 576,
            "compute_capability": 6.1,
        },
    ]
    assert _pinned_cuda_ba_compatible(inventory) is True
    assert _pinned_cuda_ba_compatible(list(reversed(inventory))) is False
    assert _pinned_cuda_ba_compatible([]) is False
