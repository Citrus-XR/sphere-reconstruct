"""起動時・UI から共通利用する dependency diagnostics."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from ctypes.util import find_library
from importlib.util import find_spec
from pathlib import Path

from .colmap.runner import resolve_colmap_bin
from .denoise.weights import MODEL_SHA256, default_model_path
from .infrastructure.filesystem import sha256_file
from .sam3.settings import quick_check as sam3_quick_check
from .settings import get_settings, workspace_root


def diagnose() -> dict:
    settings = get_settings()
    checks = {
        "workspace": _workspace_check(),
        "filesystem_roots": _filesystem_roots_check(),
        "ffmpeg": _binary_check(settings.binaries.ffmpeg, "ffmpeg", ["-version"]),
        "ffprobe": _binary_check(settings.binaries.ffprobe, "ffprobe", ["-version"]),
        "colmap": _colmap_check(),
        "sam3": _sam3_check(),
        "denoise": _denoise_check(),
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


def _colmap_check() -> dict:
    try:
        binary = resolve_colmap_bin(get_settings().binaries.colmap or None)
    except (FileNotFoundError, ValueError) as error:
        return {"ok": False, "path": None, "message": str(error), "capabilities": {}}
    version = _run([binary, "version"])
    commands = _run([binary, "-h"])
    features = _run([binary, "feature_extractor", "-h"])
    matchers = _run([binary, "sequential_matcher", "-h"])
    combined = "\n".join((commands["output"], features["output"], matchers["output"]))
    version_match = re.search(r"COLMAP\s+(\d+)\.(\d+)", version["output"])
    modern_camera_models = bool(
        version_match and (int(version_match.group(1)), int(version_match.group(2))) >= (4, 1)
    )
    cudss = next(iter(Path(binary).parent.glob("*cudss*")), None) or find_library("cudss")
    capabilities = {
        "global_mapper": "global_mapper" in commands["output"],
        "aliked": "--AlikedExtraction.max_num_features" in features["output"],
        "aliked_bruteforce": "--AlikedMatching.brute_force" in matchers["output"],
        "aliked_lightglue": "--AlikedMatching.lightglue" in matchers["output"],
        "equirectangular": modern_camera_models,
        "gpu_bundle_adjustment": bool(cudss) and ("ba_use_gpu" in combined or "ba_ceres_use_gpu" in combined),
    }
    ok = version["returncode"] == 0 and all(
        capabilities[name] for name in ("global_mapper", "aliked", "equirectangular")
    )
    return {
        "ok": ok,
        "path": binary,
        "version": version["output"].strip(),
        "capabilities": capabilities,
        "cudss": str(cudss) if cudss else None,
        "message": "ok" if ok else "COLMAP 4.1+ capabilities are incomplete",
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


def _denoise_check() -> dict:
    settings = get_settings()
    configured = settings.denoise.model_path
    model_path = Path(configured).expanduser().resolve() if configured else default_model_path()
    model_cached = model_path.is_file()
    model_verified = model_cached and sha256_file(model_path) == MODEL_SHA256
    ffmpeg_binary = settings.binaries.ffmpeg or shutil.which("ffmpeg")
    filters = _run([ffmpeg_binary, "-hide_banner", "-filters"]) if ffmpeg_binary else None
    hwaccels = _run([ffmpeg_binary, "-hide_banner", "-hwaccels"]) if ffmpeg_binary else None
    adaptive_available = bool(filters and "atadenoise" in filters["output"])
    cuda_decode_candidate = bool(hwaccels and "cuda" in hwaccels["output"].lower())
    torch_available = find_spec("torch") is not None
    fastdvdnet_ready = torch_available and (not configured or model_verified)
    if configured and not model_verified:
        message = "設定された FastDVDnet model が無いか SHA-256 が一致しません"
    elif fastdvdnet_ready or adaptive_available:
        message = "ok"
    else:
        message = "時系列ノイズ除去 runtime がありません"
    return {
        "ok": fastdvdnet_ready or adaptive_available,
        "optional": True,
        "torch": torch_available,
        "fastdvdnet_ready": fastdvdnet_ready,
        "ffmpeg_adaptive": adaptive_available,
        "hardware_decode_requested": settings.denoise.hardware_decode,
        "cuda_decode_candidate": cuda_decode_candidate,
        "model_path": str(model_path),
        "model_cached": model_cached,
        "model_verified": model_verified,
        "auto_download": not bool(configured),
        "message": message,
    }


def _cuda_runtime_check() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
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
        "torch_library_dir": str(torch_lib) if torch_lib else None,
        "cudnn": str(cudnn) if cudnn else None,
        "message": "ok" if nvidia_smi else "NVIDIA GPU not detected; CPU mode remains available",
    }


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
