"""Complete missing image neighbors and repeat registration against fixed primary poses."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from experiment_support import (
    launch_detached,
    run,
    validate_primary_preserved,
    write_json,
)


def execute(args, status):
    sys.path.insert(0, str(args.backend_src))
    from sphere_reconstruct.colmap import model as cm
    from sphere_reconstruct.colmap.camera_policy import apply_camera_policies
    from sphere_reconstruct.colmap.sequence_pairs import missing_sequence_pairs
    from sphere_reconstruct.pipeline.prepared_images import load_catalog
    from sphere_reconstruct.stages.reconstruct import Reconstruct, _incremental_args

    catalog = load_catalog(args.project)
    records = {row["name"]: row for row in catalog["images"]}
    calibration = cm.read_cameras_bin(args.calibration_model / "cameras.bin")
    database = args.output / "database.db"
    source_database = args.database or (args.project / "match_features/database.db")
    with (sqlite3.connect(source_database.resolve().as_uri() + "?mode=ro", uri=True) as original,
          sqlite3.connect(database) as target):
        original.backup(target)
    constant_ids = apply_camera_policies(database, catalog)
    with sqlite3.connect(database) as db:
        for camera_id, camera in calibration.items():
            if camera_id not in constant_ids:
                updated = db.execute("UPDATE cameras SET params=?,prior_focal_length=0 WHERE camera_id=?",
                                     (np.asarray(camera.params, dtype='<f8').tobytes(), camera_id))
                if updated.rowcount != 1:
                    raise ValueError(f"calibration camera absent from database: {camera_id}")
    pairs, sequence = missing_sequence_pairs(database, catalog["images"], set(args.ordered_source), overlap=args.window)
    pair_list = args.output / "pairs.txt"
    pair_list.write_text("\n".join(f"{first} {second}" for first, second in pairs) + "\n", encoding="utf-8")
    report = {"strategy": "image_sequence_completion_then_primary_anchored_registration",
              "sequence_matching": sequence, "primary": str(args.primary),
              "source_database": str(source_database), "matches_only": args.matches_only,
              "calibration_model": str(args.calibration_model), "input_images": len(catalog["images"]),
              "limitations": ["Added neighbors do not guarantee correct matches on repeated or reflective surfaces.",
                              "Unregistered images are reported separately from repaired poses."]}
    write_json(args.output / "report.json", report)
    matching = json.loads((args.project / "manifests/match_features.json").read_text(encoding="utf-8"))["params"]
    if (matching["feature_type"], matching["matcher_type"]) != ("SIFT", "bruteforce"):
        raise ValueError("this experiment currently requires the existing SIFT brute-force features")
    if pairs:
        run([args.colmap, "matches_importer", "--database_path", str(database),
             "--match_list_path", str(pair_list), "--match_type", "pairs",
             "--FeatureMatching.use_gpu", "1" if args.matching_gpu else "0",
             "--FeatureMatching.num_threads", "8", "--FeatureMatching.guided_matching", "1",
             "--FeatureMatching.max_num_matches", str(matching["max_num_matches"]),
             "--TwoViewGeometry.min_num_inliers", str(matching["min_num_inliers"])], args.output, "complete_matches", status)
    if args.matches_only:
        print(json.dumps(sequence, indent=2), flush=True)
        return
    primary = cm.read_model(args.primary)
    if any(records[image.name]["source_id"] != catalog["primary_source_id"] for image in primary.images.values()):
        raise ValueError("reference must be a primary-only image reconstruction")
    fixed = args.output / "constant_cameras.txt"
    fixed.write_text("\n".join(map(str, constant_ids)) + "\n", encoding="utf-8")
    previous = json.loads((args.project / "manifests/reconstruct.json").read_text(encoding="utf-8"))
    params = Reconstruct().normalize_params(previous["params"])
    spec = json.loads((args.project / "extract_features/input_spec.json").read_text(encoding="utf-8"))
    image_path = args.project / "extract_features" / spec["image_path"]
    sparse = args.output / "sparse/0"
    sparse.mkdir(parents=True)
    run([args.colmap, "mapper", "--database_path", str(database), "--image_path", str(image_path),
         "--input_path", str(args.primary), "--output_path", str(sparse),
         "--Mapper.constant_camera_list_path", str(fixed), "--Mapper.fix_existing_frames", "1",
         "--Mapper.structure_less_registration_fallback", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
         "--Mapper.multiple_models", "0", "--Mapper.num_threads", "8",
         *_incremental_args(params, video_data=True)], args.output, "register", status)
    candidate = cm.read_model(sparse)
    report["primary_preservation"] = validate_primary_preserved(primary, candidate)
    report["summary"] = candidate.summary()
    report["by_source"] = {source["id"]: {"label": source["label"],
                                         "registered": sum(records[image.name]["source_id"] == source["id"] for image in candidate.images.values()),
                                         "input": sum(row["source_id"] == source["id"] for row in catalog["images"])}
                           for source in catalog["sources"]}
    registered = {image.name for image in candidate.images.values()}
    report["unregistered_images"] = sorted(set(records) - registered)
    write_json(args.output / "report.json", report)
    del candidate, primary, calibration
    colored = args.output / "colored"
    colored.mkdir()
    run([args.colmap, "color_extractor", "--input_path", str(sparse), "--output_path", str(colored),
         "--image_path", str(image_path), "--num_threads", "8"], args.output, "colors", status)
    shutil.copy2(colored / "points3D.bin", sparse / "points3D.bin")
    print(json.dumps(report["by_source"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "primary", "calibration-model", "backend-src", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--colmap", required=True)
    parser.add_argument("--ordered-source", action="append", required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--matching-gpu", action="store_true")
    parser.add_argument("--database", type=Path, help="Reuse an immutable matching experiment database")
    parser.add_argument("--matches-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    args.output = args.output.resolve()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "phase": "prepare", "started_at": datetime.now(UTC).isoformat()}
    with ((args.output / "worker.log").open("x", encoding="utf-8", buffering=1) as log,
          contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)):
        write_json(args.output / "status.json", status)
        try:
            execute(args, status)
        except BaseException as error:
            status.update(state="failed", error=repr(error), exit_code=1)
            traceback.print_exc()
            raise
        else:
            status.update(state="completed", exit_code=0)
        finally:
            status["finished_at"] = datetime.now(UTC).isoformat()
            write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
