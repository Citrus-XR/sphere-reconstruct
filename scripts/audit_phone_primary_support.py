"""360-only の独立した 3D track で phone pose の対応支持を比較する。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pycolmap
from experiment_support import launch_detached, write_json


def primary_correspondences(primary, baseline, connection, records, output, status, *, query_ids=None):
    from sphere_reconstruct.colmap.temporal_pose import unique_correspondences

    primary_ids = set(primary.reg_image_ids())
    baseline_ids = set(baseline.reg_image_ids())
    phone_ids = baseline_ids - primary_ids if query_ids is None else set(query_ids)
    if phone_ids & primary_ids or not phone_ids <= baseline_ids:
        raise ValueError("query images must exist in the baseline and be outside the primary model")
    pairs = defaultdict(list)
    for (pair_id,) in connection.execute("SELECT pair_id FROM two_view_geometries WHERE rows>0"):
        low, high = divmod(pair_id, 2147483647)
        if low in primary_ids and high in phone_ids:
            pairs[high].append((pair_id, low, True))
        elif high in primary_ids and low in phone_ids:
            pairs[low].append((pair_id, high, False))
    point_captures = {}
    cohorts = {}
    for index, image_id in enumerate(sorted(phone_ids)):
        votes = defaultdict(set)
        for pair_id, anchor_id, reverse in pairs[image_id]:
            row = connection.execute("SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
            if row[1] != 2 or len(row[2]) != row[0] * 8:
                raise ValueError(f"invalid match array: {pair_id}")
            matches = np.frombuffer(row[2], dtype='<u4').reshape(-1, 2)
            if reverse:
                matches = matches[:, ::-1]
            anchor = primary.images[anchor_id]
            for query_index, anchor_index in matches:
                observation = anchor.point2D(int(anchor_index))
                if not observation.has_point3D():
                    continue
                point_id = observation.point3D_id
                if point_id not in point_captures:
                    point_captures[point_id] = {records[primary.images[e.image_id].name]["capture_index"]
                                                for e in primary.points3D[point_id].track.elements}
                if len(point_captures[point_id]) >= 2:
                    votes[int(query_index), point_id].add(records[anchor.name]["capture_index"])
        selected = unique_correspondences(votes)
        query = baseline.images[image_id]
        pixels = np.array([query.point2D(i).xy for i, _ in selected], dtype=float).reshape(-1, 2)
        xyz = np.array([primary.points3D[p].xyz for _, p in selected], dtype=float).reshape(-1, 3)
        cohorts[image_id] = (pixels, xyz, [votes[key] for key in selected])
        if index % 50 == 0:
            status.update(phase="primary_correspondences", processed=index, total=len(phone_ids))
            write_json(output / "status.json", status)
            print(f"PRIMARY {index}/{len(phone_ids)}", flush=True)
    return cohorts


def score(image, pixels, xyz, supporters):
    from sphere_reconstruct.colmap.temporal_pose import reprojection_errors

    if not len(pixels):
        return {"correspondences": 0, "inliers": 0, "supported": False}
    errors = reprojection_errors(image.camera, image.cam_from_world(), pixels, xyz)
    inliers = errors <= 4
    coverage = (np.ptp(pixels[inliers], axis=0) / [image.camera.width, image.camera.height]).tolist() if inliers.any() else [0, 0]
    counts = defaultdict(int)
    for keep, captures in zip(inliers, supporters, strict=True):
        if keep:
            for capture in captures:
                counts[capture] += 1
    independent = sum(count >= 10 for count in counts.values())
    return {"correspondences": len(pixels), "inliers": int(inliers.sum()), "ratio": float(inliers.mean()),
            "coverage": coverage, "primary_captures_with_ten_inliers": independent,
            "clipped_error": float(np.minimum(errors, 4).mean()),
            "supported": bool(inliers.sum() >= 20 and inliers.mean() >= 0.25
                              and min(coverage) >= 0.15 and independent >= 2)}


def execute(args, status):
    sys.path.insert(0, str(args.backend_src))
    catalog = json.loads((args.source_run / "image_catalog.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in catalog["images"]}
    primary = pycolmap.Reconstruction(str(args.source_run / "primary"))
    baseline = pycolmap.Reconstruction(str(args.source_run / "baseline"))
    with sqlite3.connect((args.source_run / "matched.db").as_uri() + "?mode=ro", uri=True) as connection:
        cohorts = primary_correspondences(primary, baseline, connection, records, args.output, status)
    np.savez_compressed(args.output / "cohorts.npz", **{
        key: value for image_id, (pixels, xyz, _) in cohorts.items()
        for key, value in [(f"pixels_{image_id}", pixels), (f"xyz_{image_id}", xyz)]
    })
    write_json(args.output / "supporters.json", {str(i): [sorted(s) for s in v[2]] for i, v in cohorts.items()})
    report = {"reference": "primary_only_3d_tracks", "sources": {}, "images": [],
              "limitations": ["Primary geometry is a reference, not measured ground truth.",
                              "Correspondences may themselves contain repeated-object mismatches."]}
    candidates = {}
    candidate_models = []
    for path in args.candidate:
        model = pycolmap.Reconstruction(str(path))
        for image_id, image in model.images.items():
            if image_id in candidates:
                raise ValueError(f"duplicate candidate image ID: {image_id}")
            candidates[image_id] = image
        candidate_models.append(model)
    for image_id, (pixels, xyz, supporters) in cohorts.items():
        image = baseline.images[image_id]
        record = records[image.name]
        row = {"image_id": image_id, "source_id": record["source_id"], "capture": record["capture_index"],
               "baseline": score(image, pixels, xyz, supporters)}
        if image_id in candidates:
            candidate = candidates[image_id]
            if image.name != candidate.name or not np.array_equal(image.camera.params, candidate.camera.params):
                raise ValueError(f"candidate identity or calibration differs: {image_id}")
            row["candidate"] = score(candidate, pixels, xyz, supporters)
        report["images"].append(row)
    for source in catalog["sources"]:
        rows = [row for row in report["images"] if row["source_id"] == source["id"]]
        if not rows:
            continue
        summary = {"label": source["label"], "images": len(rows)}
        for version in ["baseline", "candidate"]:
            values = [row[version] for row in rows if version in row]
            summary[version] = {"images": len(values), "supported": sum(value["supported"] for value in values),
                                "inliers": sum(value["inliers"] for value in values)}
        report["sources"][source["id"]] = summary
    write_json(args.output / "report.json", report)
    print(json.dumps(report["sources"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-run", "backend-src", "output"]:
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "phase": "load",
              "started_at": datetime.now(UTC).isoformat(), "command": sys.argv}
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
