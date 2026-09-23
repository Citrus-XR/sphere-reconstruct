"""Detached process execution and invariants shared by reconstruction experiments."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def launch_detached(args) -> None:
    if sys.platform != "win32":
        raise RuntimeError("--detach uses Windows WMI and requires the remote Windows host")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(sys.argv[0]).resolve()),
               *[arg for arg in sys.argv[1:] if arg != "--detach"]]
    command_line = subprocess.list2cmdline(command).replace("'", "''")
    directory = str(args.output.parent.resolve()).replace("'", "''")
    script = (
        "Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine='{command_line}'; CurrentDirectory='{directory}'}} "
        "| Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
                            capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    if record["ReturnValue"] != 0:
        raise RuntimeError(f"WMI process creation failed: {record}")
    record.update(command=command, output=str(args.output), launched_at=datetime.now(UTC).isoformat())
    write_json(args.output.with_suffix(".launch.json"), record)
    print(json.dumps(record, indent=2), flush=True)


def run(command: list[str], root: Path, label: str, status: dict) -> None:
    write_json(root / f"{label}.command.json", {"command": command})
    with (root / f"{label}.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status.update(phase=label, child_pid=process.pid)
        write_json(root / "status.json", status)
        assert process.stdout is not None
        last_print = 0.0
        try:
            for line in process.stdout:
                log.write(line)
                if time.monotonic() - last_print > 25:
                    print(f"{label}: {line.strip()}", flush=True)
                    last_print = time.monotonic()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    status.pop("child_pid", None)
    if code:
        raise RuntimeError(f"{label} exited with {code}; see {root / (label + '.log')}")


def validate_primary_preserved(reference, candidate) -> dict:
    import numpy as np

    missing = reference.images.keys() - candidate.images.keys()
    if missing:
        raise RuntimeError(f"primary images were lost: {sorted(missing)[:10]}")
    worst = 0.0
    for image_id, before in reference.images.items():
        after = candidate.images[image_id]
        if before.name != after.name or before.camera_id != after.camera_id:
            raise RuntimeError(f"primary image identity changed: {image_id}")
        for image in (before, after):
            if (not np.isfinite(image.camera_center).all() or not np.isfinite(image.qvec).all()
                    or abs(float(np.linalg.norm(image.qvec)) - 1) > 1e-9):
                raise RuntimeError(f"invalid camera pose: {image.name}")
        displacement = float(np.linalg.norm(np.asarray(before.camera_center) - after.camera_center))
        worst = max(worst, displacement)
        same_rotation = abs(float(np.dot(before.qvec, after.qvec)))
        if displacement > 1e-7 or abs(same_rotation - 1) > 1e-9:
            raise RuntimeError(f"fixed primary pose changed: {before.name}, displacement={displacement}")
    for camera_id in {image.camera_id for image in reference.images.values()}:
        before = reference.cameras[camera_id]
        after = candidate.cameras[camera_id]
        if (before.model, before.width, before.height) != (after.model, after.width, after.height):
            raise RuntimeError(f"primary camera identity changed: {camera_id}")
        if not np.allclose(before.params, after.params, rtol=0, atol=1e-10):
            raise RuntimeError(f"primary intrinsics changed: {camera_id}")
    return {"images": len(reference.images), "maximum_center_displacement": worst}
