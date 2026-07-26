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
from pathlib import Path

from .colmap.runner import resolve_colmap_bin
from .sam3.settings import quick_check as sam3_quick_check
from .settings import get_settings, workspace_root


def diagnose() -> dict:
    settings = get_settings()
    checks = {
        "workspace": _workspace_check(),
        "ffmpeg": _binary_check(settings.binaries.ffmpeg, "ffmpeg", ["-version"]),
        "ffprobe": _binary_check(settings.binaries.ffprobe, "ffprobe", ["-version"]),
        "colmap": _colmap_check(),
        "sam3": _sam3_check(),
        "cuda_runtime": _cuda_runtime_check(),
    }
    required = ("workspace", "ffmpeg", "ffprobe", "colmap")
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
