"""同じ native-fisheye database で三角化・局所 BA を順次比較する。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from sphere_reconstruct.colmap import model, runner, trajectory_quality
from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.stages.reconstruct import _incremental_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   str(args.project), str(args.output), "--colmap", args.colmap]
        options = ({"creationflags": subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                    | 0x01000000} if os.name == "nt" else {"start_new_session": True})
        with (args.output / "benchmark.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       stdin=subprocess.DEVNULL, **options)
        print(json.dumps({"pid": process.pid, "output": str(args.output)}))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    spec = InputSpec.read(args.project / "extract_features" / "input_spec.json")
    params = json.loads((args.project / "manifests" / "reconstruct.json").read_text(encoding="utf-8"))["params"]
    medium = {"filter_max_reproj_error": 1.5, "filter_min_tri_angle": 3.0,
              "tri_create_max_angle_error": 1.0, "tri_continue_max_angle_error": 1.0,
              "tri_merge_max_reproj_error": 1.5, "tri_complete_max_reproj_error": 1.5,
              "tri_min_angle": 3.0}
    cases = [("strict_repeat", {}, []), ("strict_local12", {}, ["--Mapper.ba_local_num_images", "12"]),
             ("medium", medium, [])]
    results = []
    for name, overrides, additional in cases:
        destination = args.output / name
        destination.mkdir(exist_ok=False)
        database = destination / "database.db"
        shutil.copy2(args.project / "match_features" / "database.db", database)
        started = time.monotonic()
        effective = {**params, **overrides}
        result = {"case": name, "params": effective, "additional_args": additional}
        print(f"Starting {name}", flush=True)
        try:
            command = runner.mapper(
                args.colmap, database_path=database,
                image_path=args.project / "extract_features" / spec.image_path,
                output_path=destination / "sparse", refine_intrinsics=spec.refine_intrinsics,
                refine_rig=spec.refine_rig, multiple_models=spec.multiple_models,
                extra_args=[*_incremental_args(effective, video_data=True), *additional],
                log_path=destination / "mapper.log",
            )
            reconstruction = model.read_model(destination / "sparse" / "0")
            result.update(reconstruction.summary())
            result["track_length_counts"] = dict(Counter(len(point.track) for point in reconstruction.points3D.values()))
            result["trajectory"] = trajectory_quality.evaluate_primary_trajectory(
                reconstruction, spec.images, spec.primary_source_id, max_step_ratio=10.0)
            result["command"] = command.command
            result["returncode"] = command.returncode
        except Exception as error:
            result["error"] = repr(error)
        result["elapsed_seconds"] = time.monotonic() - started
        log_path = destination / "mapper.log"
        result["linear_solver_failures"] = (
            log_path.read_text(encoding="utf-8").count("Linear solver failure") if log_path.is_file() else None)
        (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
