"""Native fisheye の不確実な既存点を除去し、隔離 LFStudio dataset を生成する。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from audit_fixed_pose import (
    fixed_pose_invariants,
    sparse_fingerprint,
)
from benchmark_fixed_pose import fingerprint, write_json
from sphere_reconstruct.colmap import model, web_preview
from sphere_reconstruct.domain.mask_artifact import MaskPurpose, mask_manifest_path
from sphere_reconstruct.stages.export_dataset import (
    _copy_masks,
    _copy_registered_images,
    _safe_relative_path,
    _validate_lf_dataset,
)
from supplement_stable_points import write_ply

from sphere_reconstruct.colmap.point_stability import (
    FULL_TRACK_COLUMNS,
    METRIC_COLUMNS,
    assess_point,
    geometry,
    retain_points,
    validate_tracks,
)


def execute(args):
    source = args.project / "reconstruct" / "sparse" / "0"
    spec_path = args.project / "extract_features" / "input_spec.json"
    watched = [spec_path, args.project / "rectify_fisheye" / "image_catalog.json",
               mask_manifest_path(args.project, MaskPurpose.TRAINING)]
    before = {str(path.relative_to(args.project)): fingerprint(path) for path in watched}
    source_fingerprint = sparse_fingerprint(source)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in spec["images"]}
    reconstruction = model.read_model(source)
    if not reconstruction.points3D:
        raise ValueError("source reconstruction has no points")
    unsupported = {camera.model for camera in reconstruction.cameras.values()} - {"OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"}
    if unsupported:
        raise ValueError(f"this experiment requires native fisheye cameras: {sorted(unsupported)}")
    for image in reconstruction.images.values():
        if image.name not in records or _safe_relative_path(image.name) is None:
            raise ValueError(f"missing capture metadata or unsafe image name: {image.name}")
    validate_tracks(reconstruction)
    views = geometry(reconstruction)
    points = sorted(reconstruction.points3D.values(), key=lambda point: point.point3D_id)
    metrics, reasons, retained = [], [], set()
    started = time.monotonic()
    counts = Counter()
    for index, point in enumerate(points, 1):
        if args.policy == "original":
            reason, values = "keep", np.full(len(METRIC_COLUMNS), np.nan)
        else:
            reason, values = assess_point(point, views, records, relative_budget=args.relative_error,
                                         pixel_sigma=args.pixel_sigma, cross_limit=args.max_cross_error,
                                         policy=args.policy)
        metrics.append(values)
        reasons.append(reason)
        counts[reason] += 1
        if reason == "keep":
            retained.add(point.point3D_id)
        if index % 1000 == 0 or index == len(points):
            state = {"status": "running", "phase": "assess_points", "current": index,
                     "total": len(points), "counts": dict(counts), "elapsed_seconds": time.monotonic() - started}
            write_json(args.output / "status.json", state)
            print(json.dumps(state), flush=True)
    np.savez_compressed(args.output / "point_assessment.npz", point_ids=[p.point3D_id for p in points],
                        values=metrics, reasons=reasons,
                        columns=METRIC_COLUMNS if args.policy == "split" else FULL_TRACK_COLUMNS,
                        model_fingerprint=json.dumps(source_fingerprint, sort_keys=True))
    if not retained:
        raise ValueError("no points satisfy the requested uncertainty budget; assessment has been saved")
    report = {"input_points": len(points), "retained_points": len(retained),
              "removed_points": len(points) - len(retained), "counts": dict(counts),
              "relative_error_budget": args.relative_error, "pixel_sigma_assumption": args.pixel_sigma,
              "cross_p95_limit_px": args.max_cross_error,
              "cross_max_limit_px": None if args.policy == "original" else 2 * args.max_cross_error,
              "policy": args.policy,
              "minimum_captures": {"original": None, "split": 4, "full_track": 3}[args.policy],
              "cross_validation": {"original": None, "split": "ordered and alternating halves",
                                   "full_track": "leave one capture out"}[args.policy],
              "uncertainty_observations": {"original": None, "split": "each half",
                                           "full_track": "full track"}[args.policy],
              "uncertainty": "conditional linearized 95% ellipsoid radius; fixed cameras and independent isotropic pixel errors",
              "model_fingerprint": source_fingerprint, "source_artifact_fingerprints": before,
              "original_geometry_retained": True, "added_points": 0}
    write_ply(args.output / "removed_points.ply", {p.point3D_id: p for p in points if p.point3D_id not in retained})
    retain_points(reconstruction, retained)
    dataset = args.output / "export_dataset"
    sparse = dataset / "sparse" / "0"
    sparse.mkdir(parents=True)
    model.write_cameras_bin(sparse / "cameras.bin", reconstruction.cameras)
    model.write_images_bin(sparse / "images.bin", reconstruction.images)
    model.write_points3D_bin(sparse / "points3D.bin", reconstruction.points3D)
    for name in ("rigs.bin", "frames.bin"):
        if (source / name).exists():
            shutil.copy2(source / name, sparse / name)
    write_ply(dataset / "points.ply", reconstruction.points3D)
    web_preview.write_web_preview(reconstruction, dataset / "preview")
    expected_sizes = {image.name: (reconstruction.cameras[image.camera_id].width,
                                   reconstruction.cameras[image.camera_id].height)
                      for image in reconstruction.images.values()}

    def progress(phase):
        def update(current, total):
            if current % 25 == 0 or current == total:
                state = {"status": "running", "phase": phase, "current": current, "total": total}
                write_json(args.output / "status.json", state)
                print(json.dumps(state), flush=True)
        return update

    sizes = _copy_registered_images(args.project / "extract_features" / spec["image_path"],
                                    dataset / "images", expected_sizes, progress=progress("export_images"))
    _, mask_sizes = _copy_masks(args.project, MaskPurpose.TRAINING, dataset / "masks", set(expected_sizes),
                                expected_sizes, progress=progress("export_training_masks"))
    loaded = model.read_model(sparse)
    validate_tracks(loaded)
    report["camera_invariants"] = fixed_pose_invariants(model.read_model(source), loaded)
    validation = _validate_lf_dataset(dataset, loaded, image_sizes=sizes, mask_sizes=mask_sizes,
                                      reference_prefix=f"sources/{spec['primary_source_id']}/")
    if not validation["training_ready"] or validation["matched_mask_count"] != len(loaded.images):
        raise ValueError(f"incomplete LFStudio dataset: {validation}")
    if source_fingerprint != sparse_fingerprint(source) or before != {
        str(path.relative_to(args.project)): fingerprint(path) for path in watched
    }:
        raise RuntimeError("source artifacts changed during export; comparison must be repeated")
    report["source_unchanged"] = True
    report["validation"] = validation
    write_json(args.output / "report.json", report)
    write_json(dataset / "export_manifest.json", {
        "format": "sphere-reconstruct-export", "version": 3, "load_in_lichtfeld_studio": ".",
        "camera_models": sorted({camera.model for camera in loaded.cameras.values()}),
        "images": len(loaded.images), "points3D": len(loaded.points3D), "masks": len(mask_sizes),
        "mask_source": "training", "image_source": "original", "training_crop": {"enabled": False},
        "model_source": "sparse_cleanup_experiment", "cleanup_policy": args.policy, "validation": validation,
        "strict_cleanup_report": "../report.json",
        "recommended_viewer_settings": {"gut": True, "undistort": False, "mask_mode": "segment"},
    })
    write_json(args.output / "status.json", {"status": "succeeded", "retained_points": len(retained),
                                              "source_unchanged": True, "export_dataset": "export_dataset"})
    print(json.dumps({"status": "succeeded", "counts": dict(counts), "validation": validation}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--relative-error", type=float)
    parser.add_argument("--pixel-sigma", type=float)
    parser.add_argument("--max-cross-error", type=float)
    parser.add_argument("--policy", choices=["original", "split", "full_track"], default="split")
    args = parser.parse_args()
    for value in (args.relative_error, args.pixel_sigma, args.max_cross_error):
        if args.policy == "original":
            if value is not None:
                raise ValueError("original baseline does not apply error budgets")
        elif value is None or not np.isfinite(value) or value <= 0:
            raise ValueError("all three error and uncertainty budgets must be specified, positive and finite")
    if args.policy != "original" and args.relative_error >= 1:
        raise ValueError("relative error budget must be less than one")
    args.project, args.output = args.project.resolve(), args.output.resolve()
    if args.output.is_relative_to(args.project):
        raise ValueError("experiment must be outside the source project")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        execute(args)
    except Exception as error:
        write_json(args.output / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
