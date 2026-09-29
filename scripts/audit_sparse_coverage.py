"""画像ごとの sparse 観測を原モデル、清理結果、現行モデルで比較する。"""

from __future__ import annotations

import argparse
import contextlib
import json
import struct
import sys
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from sphere_reconstruct.colmap.model import read_cameras_bin
from experiment_support import launch_detached, write_json


INVALID_POINT_ID = 2**64 - 1
OBSERVATION_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<u8")])
IMAGE_HEADER = struct.Struct("<idddddddi")
POINT_HEADER = struct.Struct("<QdddBBBdQ")


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def point_ids(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        result = np.empty(count, dtype=np.uint64)
        for index in range(count):
            row = POINT_HEADER.unpack(read_exact(stream, POINT_HEADER.size))
            result[index] = row[0]
            stream.seek(row[-1] * 8, 1)
        if stream.read(1):
            raise ValueError(f"trailing bytes in {path}")
    return result


def image_rows(path: Path, cameras: dict, subsets: dict[str, np.ndarray]):
    with path.open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(count):
            header = IMAGE_HEADER.unpack(read_exact(stream, IMAGE_HEADER.size))
            name_bytes = bytearray()
            while (byte := read_exact(stream, 1)) != b"\0":
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8")
            num_points = struct.unpack("<Q", read_exact(stream, 8))[0]
            observations = np.frombuffer(
                read_exact(stream, num_points * OBSERVATION_DTYPE.itemsize),
                dtype=OBSERVATION_DTYPE,
            )
            registered = observations[observations["id"] != INVALID_POINT_ID]
            camera = cameras[header[-1]]
            rows = {}
            for label, ids in subsets.items():
                selected = registered if ids is None else registered[np.isin(registered["id"], ids)]
                cells = np.floor(
                    np.column_stack((selected["x"] / camera.width, selected["y"] / camera.height)) * 16
                ).astype(np.int64)
                valid = np.all((cells >= 0) & (cells < 16), axis=1)
                counts = np.bincount(cells[valid, 1] * 16 + cells[valid, 0], minlength=256)
                rows[label] = {
                    "points": len(selected),
                    "cells_with_3_points": int(np.count_nonzero(counts >= 3)),
                    "cells": counts.tolist(),
                }
            yield name, rows
        if stream.read(1):
            raise ValueError(f"trailing bytes in {path}")


def audit(args: argparse.Namespace) -> None:
    with np.load(args.old_assessment) as assessment:
        old_kept = assessment["point_ids"][assessment["reasons"] == "keep"]
        old_reasons = Counter(assessment["reasons"].tolist())
    current_ids = point_ids(args.current / "points3D.bin")
    with args.preview_points.open("rb") as stream:
        preview_count, preview_stride = struct.unpack("<II", read_exact(stream, 8))
    if preview_stride != 20 or preview_count > len(current_ids):
        raise ValueError("unexpected preview point header")
    preview_ids = current_ids[
        np.asarray([int(index * len(current_ids) / preview_count) for index in range(preview_count)])
    ]
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    roles = {row["name"]: row["source_role"] for row in catalog["images"]}
    raw_cameras = read_cameras_bin(args.raw / "cameras.bin")
    current_cameras = read_cameras_bin(args.current / "cameras.bin")
    raw_rows = dict(image_rows(args.raw / "images.bin", raw_cameras, {"raw": None, "old_cleaned": old_kept}))
    current_rows = dict(image_rows(
        args.current / "images.bin", current_cameras, {"current": None, "preview": preview_ids}
    ))
    if set(raw_rows) != set(current_rows):
        raise ValueError("raw and current image sets differ")
    summary = {}
    report_rows = []
    for name, stages in raw_rows.items():
        stages.update(current_rows[name])
        role = roles[name]
        record = {"name": name, "role": role}
        for label, values in stages.items():
            record[label] = {key: value for key, value in values.items() if key != "cells"}
        record["lost_cells_raw_to_old"] = int(np.count_nonzero(
            (np.asarray(stages["raw"]["cells"]) >= 3)
            & (np.asarray(stages["old_cleaned"]["cells"]) < 3)
        ))
        record["lost_cells_raw_to_current"] = int(np.count_nonzero(
            (np.asarray(stages["raw"]["cells"]) >= 3)
            & (np.asarray(stages["current"]["cells"]) < 3)
        ))
        report_rows.append(record)
        group = summary.setdefault(role, {"images": 0})
        group["images"] += 1
        for label in stages:
            group[f"{label}_zero_images"] = group.get(f"{label}_zero_images", 0) + (stages[label]["points"] == 0)
            group[f"{label}_under_10_images"] = group.get(f"{label}_under_10_images", 0) + (stages[label]["points"] < 10)
            group[f"{label}_observations"] = group.get(f"{label}_observations", 0) + stages[label]["points"]
    result = {
        "models": {"raw_points": sum(old_reasons.values()), "old_cleaned_points": len(old_kept),
                   "current_points": len(current_ids), "preview_points": preview_count},
        "old_cleanup_reasons": old_reasons,
        "summary_by_role": summary,
        "images": report_rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "images"}, indent=2))
    for label, key in (("raw_sparse", lambda row: row["raw"]["points"]),
                       ("lost_cells", lambda row: -row["lost_cells_raw_to_current"])):
        print(label, json.dumps(sorted(report_rows, key=key)[:20], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--old-assessment", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--preview-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "started_at": datetime.now(UTC).isoformat()}
    write_json(args.output / "status.json", status)
    with (args.output / "worker.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                audit(args)
                status["state"] = "completed"
            except BaseException as error:
                status.update(state="failed", error=repr(error))
                traceback.print_exc()
                raise
            finally:
                status["finished_at"] = datetime.now(UTC).isoformat()
                write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
