"""起動時・UI から共通利用する dependency diagnostics."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from ctypes.util import find_library
from pathlib import Path

from .colmap.runner import resolve_colmap_bin, resolve_vocab_tree_path
from .sam3.settings import quick_check as sam3_quick_check
from .settings import get_settings, workspace_root


def diagnose() -> dict:
    settings = get_settings()
    checks = {
        "workspace": _workspace_check(),
        "filesystem_roots": _filesystem_roots_check(),
        "ffmpeg": _ffmpeg_check(),
        "ffprobe": _binary_check(settings.binaries.ffprobe, "ffprobe", ["-version"]),
        "colmap": _colmap_check(),
        "vocab_tree": _vocab_tree_check(),
        "jpegtran": _jpegtran_check(),
        "sam3": _sam3_check(),
        "cuda_runtime": _cuda_runtime_check(),
    }
    required = ("workspace", "filesystem_roots", "ffmpeg", "ffprobe", "colmap")
    return {
        "ready": all(checks[name]["ok"] for name in required),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "checks": checks,
    }


def _binary_check(explicit: str, name: str, version_args: list[str]) -> dict:
    resolved = str(Path(explicit)) if explicit else shutil.which(name)
    if not resolved or not Path(resolved).is_file():
        return {"ok": False, "path": resolved, "message": f"{name} not found"}
    result = _run([resolved, *version_args])
    return {
        "ok": result["returncode"] == 0,
        "path": resolved,
        "version": result["output"].splitlines()[0] if result["output"] else "",
        "message": "ok" if result["returncode"] == 0 else result["output"][-500:],
    }


def _ffmpeg_check() -> dict:
    settings = get_settings()
    check = _binary_check(settings.binaries.ffmpeg, "ffmpeg", ["-version"])
    methods: list[str] = []
    if check["ok"]:
        result = _run([check["path"], "-hide_banner", "-v", "error", "-hwaccels"])
        if result["returncode"] == 0:
            methods = [
                line.strip()
                for line in result["output"].splitlines()
                if line.strip() and not line.lower().startswith("hardware acceleration")
            ]
    check["hardware_decode"] = {
        "preference": settings.frame_extraction.hwaccel,
        "required": settings.frame_extraction.require_hwaccel,
        "available_methods": methods,
        "source_probe": "performed when frame extraction starts",
    }
    preference = settings.frame_extraction.hwaccel.lower()
    if check["ok"]:
        available_text = ", ".join(methods) if methods else "none"
        check["message"] = f"ok; hardware decode={preference}, available={available_text}"
    if settings.frame_extraction.require_hwaccel and settings.frame_extraction.hwaccel in {
        "",
        "none",
        "software",
    }:
        check["ok"] = False
        check["message"] = "hardware decode is required but disabled"
    elif settings.frame_extraction.require_hwaccel and preference != "auto" and preference not in methods:
        check["ok"] = False
        check["message"] = f"required hardware decoder is not compiled into FFmpeg: {preference}"
    return check


def _jpegtran_check() -> dict:
    check = _binary_check(get_settings().binaries.jpegtran, "jpegtran", ["-version"])
    check["optional"] = True
    if not check["ok"]:
        check["message"] = "jpegtran not found; lossless fisheye training crop is unavailable"
    return check


def _vocab_tree_check() -> dict:
    try:
        path = resolve_vocab_tree_path(get_settings().binaries.vocab_tree or None)
    except (FileNotFoundError, ValueError) as error:
        return {"ok": False, "optional": True, "path": None, "message": str(error)}
    if path is None:
        return {
            "ok": False,
            "optional": True,
            "path": None,
            "message": "vocabulary tree not found; loop closure is unavailable",
        }
    return {"ok": True, "optional": True, "path": str(path), "message": "ok"}


def _colmap_check() -> dict:
    try:
        binary = resolve_colmap_bin(get_settings().binaries.colmap or None)
    except (FileNotFoundError, ValueError) as error:
        return {"ok": False, "path": None, "message": str(error), "capabilities": {}}
    version = _run([binary, "version"])
    commands = _run([binary, "-h"])
    features = _run([binary, "feature_extractor", "-h"])
    matchers = _run([binary, "sequential_matcher", "-h"])
    global_mapper = _run([binary, "global_mapper", "-h"])
    combined = "\n".join(
        (commands["output"], features["output"], matchers["output"], global_mapper["output"])
    )
    version_match = re.search(r"COLMAP\s+(\d+)\.(\d+)", version["output"])
    modern_camera_models = bool(
        version_match and (int(version_match.group(1)), int(version_match.group(2))) >= (4, 1)
    )
    bundle_adjustment = _bundle_adjustment_capabilities(Path(binary))
    capabilities = {
        "global_mapper": "global_mapper" in commands["output"],
        "aliked": "--AlikedExtraction.max_num_features" in features["output"],
        "aliked_bruteforce": "--AlikedMatching.brute_force" in matchers["output"],
        "aliked_lightglue": "--AlikedMatching.lightglue" in matchers["output"],
        "equirectangular": modern_camera_models,
        "gpu_bundle_adjustment": bundle_adjustment["sparse"] and "ba_ceres_use_gpu" in combined,
        "gpu_bundle_adjustment_dense": bundle_adjustment["dense"],
        "gpu_bundle_adjustment_sparse": bundle_adjustment["sparse"],
        "onnx_cuda_runtime": bundle_adjustment["onnx_cuda"],
    }
    ok = version["returncode"] == 0 and all(
        capabilities[name] for name in ("global_mapper", "aliked", "equirectangular")
    )
    return {
        "ok": ok,
        "path": binary,
        "version": version["output"].strip(),
        "capabilities": capabilities,
        "ceres_cuda": bundle_adjustment["dense"],
        "cudss": bundle_adjustment["cudss"],
        "cudnn": bundle_adjustment["cudnn"],
        "runtime_metadata": bundle_adjustment["metadata"],
        "message": "ok" if ok else "COLMAP 4.1+ capabilities are incomplete",
    }


def _bundle_adjustment_capabilities(binary: Path) -> dict:
    metadata_paths = (
        binary.parent / "sphere-colmap-capabilities.json",
        binary.parent.parent / "sphere-colmap-capabilities.json",
    )
    metadata_path = next((path for path in metadata_paths if path.is_file()), None)
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        return {
            "dense": bool(metadata.get("gpu_bundle_adjustment_dense")),
            "sparse": bool(metadata.get("gpu_bundle_adjustment_sparse")),
            "cudss": metadata.get("cudss_version"),
            "cudnn": metadata.get("cudnn_version"),
            "onnx_cuda": bool(metadata.get("onnx_cuda_runtime_bundled")),
            "metadata": str(metadata_path),
        }

    ceres_candidates = [
        *binary.parent.glob("ceres*.dll"),
        *binary.parent.glob("libceres*.so*"),
        *binary.parent.glob("libceres*.dylib"),
    ]
    ceres = next((path for path in ceres_candidates if path.is_file()), None)
    cudss = next(iter(binary.parent.glob("*cudss*")), None) or find_library("cudss")
    if ceres is None:
        return {
            "dense": False,
            "sparse": False,
            "cudss": None,
            "cudnn": None,
            "onnx_cuda": False,
            "metadata": None,
        }
    binary_strings = ceres.read_bytes()
    no_cuda = b"Ceres was compiled without support for CUDA" in binary_strings
    dense = not no_cuda and b"CUDA" in binary_strings
    sparse = dense and bool(cudss) and b"CERES_NO_CUDSS" not in binary_strings
    return {
        "dense": dense,
        "sparse": sparse,
        "cudss": str(cudss) if cudss else None,
        "cudnn": str(next(iter(binary.parent.glob("cudnn64_*.dll")), "")) or None,
        "onnx_cuda": bool(
            next(iter(binary.parent.glob("onnxruntime_providers_cuda.dll")), None)
            and next(iter(binary.parent.glob("cudnn64_*.dll")), None)
        ),
        "metadata": None,
    }


def _workspace_check() -> dict:
    root = workspace_root()
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix="doctor-", delete=True):
            pass
    except OSError as error:
        return {"ok": False, "path": str(root), "message": str(error)}
    return {"ok": True, "path": str(root), "message": "ok"}


def _filesystem_roots_check() -> dict:
    roots = [root.expanduser().resolve() for root in get_settings().filesystem.allowed_roots]
    valid = [str(root) for root in roots if root.is_dir()]
    invalid = [str(root) for root in roots if not root.is_dir()]
    ok = bool(valid)
    return {
        "ok": ok,
        "paths": valid,
        "missing_paths": invalid,
        "message": "ok" if ok else "filesystem.allowed_roots has no accessible directory",
    }


def _sam3_check() -> dict:
    check = sam3_quick_check()
    return {
        "ok": check.ok,
        "optional": True,
        "message": check.message,
        "checkpoint_size": check.checkpoint_size,
    }


def _cuda_runtime_check() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_inventory: list[dict] = []
    if nvidia_smi:
        query = _run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
        if query["returncode"] == 0:
            gpu_inventory = _parse_nvidia_smi_inventory(query["output"])
    torch_lib = (
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        if os.name == "nt"
        else next(iter((Path(sys.prefix) / "lib").glob("python*/site-packages/torch/lib")), None)
    )
    cudnn = None
    if torch_lib and Path(torch_lib).is_dir():
        cudnn = next(iter(Path(torch_lib).glob("cudnn*")), None)
    return {
        "ok": nvidia_smi is not None,
        "optional": True,
        "nvidia_smi": nvidia_smi,
        "gpus": gpu_inventory,
        "pinned_cuda_ba_compatible": _pinned_cuda_ba_compatible(gpu_inventory),
        "torch_library_dir": str(torch_lib) if torch_lib else None,
        "cudnn": str(cudnn) if cudnn else None,
        "message": "ok" if nvidia_smi else "NVIDIA GPU not detected; CPU mode remains available",
    }


def _parse_nvidia_smi_inventory(output: str) -> list[dict]:
    inventory = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, driver_version, compute_capability_text = fields
        try:
            driver_major = int(driver_version.split(".", 1)[0])
            compute_capability = float(compute_capability_text)
        except ValueError:
            continue
        inventory.append(
            {
                "name": name,
                "driver_version": driver_version,
                "driver_major": driver_major,
                "compute_capability": compute_capability,
            }
        )
    return inventory


def _pinned_cuda_ba_compatible(inventory: list[dict]) -> bool:
    return bool(inventory) and (
        inventory[0]["driver_major"] >= 580 and inventory[0]["compute_capability"] >= 7.5
    )


def _run(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "output": str(error)}
    return {
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip(),
    }
