"""Native fisheye の固定 pose 再三角化を隔離 database で比較する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from contextlib import closing
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from sphere_reconstruct.colmap import runner
from sphere_reconstruct.colmap.model import qvec_to_rotation
from sphere_reconstruct.colmap.quality import mapper_extra_args

MAX_IMAGE_ID = 2_147_483_647


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"size": path.stat().st_size, "sha256": digest}


def readonly_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def filter_graph(source: Path, destination: Path, records: dict, centers: dict,
                 validation_names: set[str], minimum_baseline: float) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    with closing(readonly_database(source)) as original, closing(sqlite3.connect(destination)) as target:
        original.backup(target)
    with closing(sqlite3.connect(destination)) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        pairs = db.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0").fetchall()
        removed = []
        summary = {"input_pairs": len(pairs), "input_matches": sum(n for _, n in pairs),
                   "withheld_pairs": 0, "short_baseline_pairs": 0,
                   "retained_pairs": 0, "retained_matches": 0}
        support = {name: 0 for name in names.values()}
        for pair_id, count in pairs:
            first, second = divmod(pair_id, MAX_IMAGE_ID)
            a, b = names[first], names[second]
            if a in validation_names or b in validation_names:
                summary["withheld_pairs"] += 1
                removed.append((pair_id,))
                continue
            same_source = records[a]["source_id"] == records[b]["source_id"]
            if same_source and np.linalg.norm(centers[a] - centers[b]) < minimum_baseline:
                summary["short_baseline_pairs"] += 1
                removed.append((pair_id,))
                continue
            summary["retained_pairs"] += 1
            summary["retained_matches"] += count
            support[a] += count
            support[b] += count
        db.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", removed)
        db.executemany("DELETE FROM matches WHERE pair_id=?", removed)
        db.commit()
    summary["support_by_image"] = support
    summary["training_images_without_matches"] = sum(
        value == 0 for name, value in support.items() if name not in validation_names
    )
    return summary


def execute(args: argparse.Namespace) -> None:
    output = args.output
    source_model = args.project / "reconstruct" / "sparse" / "0"
    source_database = args.project / "reconstruct" / "database.db"
    spec_path = args.project / "extract_features" / "input_spec.json"
    manifest_path = args.project / "manifests" / "reconstruct.json"
    watched = [source_database, manifest_path, spec_path, *sorted(source_model.glob("*.bin"))]
    before = {str(p.relative_to(args.project)): fingerprint(p) for p in watched}
    reference = output / "reference"
    shutil.copytree(source_model, reference)
    shutil.copy2(spec_path, output / "input_spec.json")
    shutil.copy2(manifest_path, output / "reconstruct_manifest.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    params = json.loads(manifest_path.read_text(encoding="utf-8"))["params"]
    tri_keys = ("filter_max_reproj_error", "filter_min_tri_angle", "tri_create_max_angle_error",
                "tri_continue_max_angle_error", "tri_merge_max_reproj_error",
                "tri_complete_max_reproj_error", "tri_min_angle")
    if any(key not in params for key in tri_keys):
        raise ValueError("source manifest must explicitly contain all seven triangulation parameters")
    preview = json.loads((args.project / "reconstruct" / "preview" / "reconstruction.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in spec["images"]}
    centers = {image["name"]: -np.asarray(qvec_to_rotation(image["qvec"])).T @ np.asarray(image["tvec"])
               for image in preview["images"]}
    if centers.keys() != records.keys():
        raise ValueError("benchmark requires all input images to have registered poses")
    captures = {}
    for name, record in records.items():
        key = (record["source_id"], record["capture_index"])
        captures.setdefault(key, []).append(centers[name])
    capture_centers = {key: np.mean(value, axis=0) for key, value in captures.items()}
    primary_captures = sorted(key for key in captures if key[0] == spec["primary_source_id"])
    positions = np.array([capture_centers[key] for key in primary_captures])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    positive_steps = steps[steps > 1e-10]
    if not len(positive_steps):
        raise ValueError("primary trajectory has no translational baseline")
    median_step = float(np.median(positive_steps))
    # Capture 単位で分けるため同時刻の二つの lens が train/test に分離しない。
    validation_names = sorted(name for name, record in records.items() if record["capture_index"] % 5 == 2)
    write_json(output / "validation_names.json", validation_names)
    plan = {"reference_fingerprints": before, "source_params": params,
            "median_capture_step": median_step, "validation_images": len(validation_names),
            "validation_split": "capture_index % 5 == 2, both sensors together",
            "scope": "depth validation conditional on reused poses and verified correspondences",
            "cases": args.cases}
    write_json(output / "plan.json", plan)
    results = []
    for case in args.cases:
        destination = output / case
        destination.mkdir(exist_ok=False)
        model_dir = destination / "sparse" / "0"
        model_dir.mkdir(parents=True)
        withheld = set() if case.startswith("fixed_") else set(validation_names)
        multiplier = 2 if case == "heldout_wide2" else 4 if case == "heldout_wide4" else 0
        # Pair filtering は rig center、triangulation は各 sensor の固有 center を使う。
        pair_centers = {name: capture_centers[(record["source_id"], record["capture_index"])]
                        for name, record in records.items()}
        print(f"Preparing {case}", flush=True)
        graph = filter_graph(source_database, destination / "database.db", records,
                             pair_centers, withheld, multiplier * median_step)
        write_json(destination / "graph.json", graph)
        effective = dict(params)
        if case in {"heldout_far", "fixed_far"}:
            effective.update(tri_min_angle=1.5, filter_min_tri_angle=1.5)
        command = ["point_triangulator", "--database_path", str(destination / "database.db"),
                   "--image_path", str(args.project / "extract_features" / spec["image_path"]),
                   "--input_path", str(reference), "--output_path", str(model_dir),
                   "--clear_points", "1", "--refine_intrinsics", "0",
                   "--Mapper.fix_existing_frames", "1", "--Mapper.ba_refine_sensor_from_rig", "0",
                   "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_use_gpu", "0",
                   "--Mapper.extract_colors", "1", "--Mapper.random_seed", str(params["random_seed"]),
                   "--Mapper.ba_local_backend", "CERES", "--Mapper.ba_global_backend", "CERES",
                   "--Mapper.tri_ignore_two_view_tracks", "1", "--Mapper.tri_max_transitivity", "1",
                   "--Mapper.tri_complete_max_transitivity", "5", "--Mapper.tri_re_max_angle_error", "5",
                   "--Mapper.tri_re_min_ratio", "0.2", "--Mapper.tri_re_max_trials", "1",
                   *mapper_extra_args({k: v for k, v in effective.items() if k != "ba_use_gpu"})]
        result = {"case": case, "minimum_pair_baseline": multiplier * median_step,
                  "withheld_images": len(withheld), "params": effective, "command": [args.colmap, *command]}
        write_json(destination / "command.json", result)
        print(f"Starting {case}: {graph['retained_pairs']} pairs, baseline >= {multiplier * median_step:.6g}", flush=True)
        started = time.monotonic()
        runner.run_command(args.colmap, command, log_path=destination / "triangulator.log")
        result["elapsed_seconds"] = time.monotonic() - started
        result["status"] = "succeeded"
        write_json(destination / "result.json", result)
        results.append(result)
        write_json(output / "results.json", results)
        print(f"Finished {case} in {result['elapsed_seconds']:.1f}s", flush=True)
    after = {str(p.relative_to(args.project)): fingerprint(p) for p in watched}
    if before != after:
        raise RuntimeError("source artifacts changed during experiment; comparison requires review")
    write_json(output / "status.json", {"status": "succeeded", "source_unchanged": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--cases", nargs="+", choices=["fixed_all", "fixed_far", "heldout_far", "heldout_all", "heldout_wide2", "heldout_wide4"],
                        default=["fixed_all", "heldout_all", "heldout_wide2"])
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.project):
        raise ValueError("experiment output must be outside the source project")
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), str(args.project), str(args.output),
                   "--colmap", args.colmap, "--cases", *args.cases, "--worker"]
        options = ({"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW | 0x01000000} if os.name == "nt" else {"start_new_session": True})
        with (args.output / "benchmark.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, **options)
        print(json.dumps({"pid": child.pid, "output": str(args.output)}))
        return
    write_json(args.output / "status.json", {"status": "running", "pid": os.getpid()})
    try:
        execute(args)
    except Exception:
        write_json(args.output / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
