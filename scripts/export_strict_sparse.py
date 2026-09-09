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
    geometry,
    sparse_fingerprint,
    triangulate_rays,
)
from benchmark_fixed_pose import fingerprint, write_json
from sphere_reconstruct.colmap import model, web_preview
from sphere_reconstruct.domain.mask_artifact import MaskPurpose, mask_manifest_path
from sphere_reconstruct.imaging.fisheye_camera import (
    camera_rays_to_pixels,
    pixels_to_camera_rays,
)
from sphere_reconstruct.stages.export_dataset import (
    _copy_masks,
    _copy_registered_images,
    _safe_relative_path,
    _validate_lf_dataset,
)
from supplement_stable_points import write_ply

METRIC_COLUMNS = ["captures", "split_relative_difference", "original_relative_difference",
                  "cross_p95_px", "cross_max_px", "conditional_radius95_relative",
                  "original_p95_px", "original_max_px"]
FULL_TRACK_COLUMNS = ["captures", "full_fit_relative_difference", "loo_relative_difference",
                      "cross_p95_px", "cross_max_px", "conditional_radius95_relative",
                      "original_p95_px", "original_max_px"]


def project_and_jacobian(xyz, centers, rotations, cameras):
    local = np.einsum("nij,nj->ni", rotations, np.asarray(xyz, dtype=np.float64) - centers)
    if not np.all(np.isfinite(local)) or np.any(local[:, 2] <= 0):
        return None
    pixels = np.empty((len(local), 2))
    jacobian = np.empty((len(local), 2, 3))
    for camera_id in sorted({camera.camera_id for camera in cameras}):
        indices = np.array([i for i, camera in enumerate(cameras) if camera.camera_id == camera_id])
        camera = cameras[indices[0]]
        points = local[indices]
        step = np.linalg.norm(points, axis=1) * 1e-5
        if np.any(step <= 0):
            return None
        pixels[indices] = camera_rays_to_pixels(camera.model, camera.params, points)
        for axis in range(3):
            offset = np.zeros_like(points)
            offset[:, axis] = step
            plus = camera_rays_to_pixels(camera.model, camera.params, points + offset)
            minus = camera_rays_to_pixels(camera.model, camera.params, points - offset)
            jacobian[indices, :, axis] = (plus - minus) / (2 * step[:, None])
    world_jacobian = np.einsum("nij,njk->nik", jacobian, rotations)
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(world_jacobian)):
        return None
    return pixels, world_jacobian


def relative_radius95(jacobian, sigma_px, distance):
    singular = np.linalg.svd(jacobian.reshape((-1, 3)), compute_uv=False)
    if singular[-1] <= singular[0] * 1e-10:
        return float("inf")
    # sqrt(chi2.ppf(0.95, 3)); 固定 camera・独立等方 pixel noise の線形近似。
    return float(2.7954834829151074 * sigma_px / singular[-1] / distance)


def assess_full_track(point, centers, rotations, cameras, pixels, directions, keys, captures,
                      distance, original_projection, values, *, relative_budget, pixel_sigma, cross_limit):
    values[5] = relative_radius95(original_projection[1], pixel_sigma, distance)
    if values[5] > relative_budget:
        return "conditional_uncertainty", values
    fit = triangulate_rays(centers, directions)
    if fit is None:
        return "degenerate_full_track", values
    values[1] = np.linalg.norm(fit - point.xyz) / distance
    if values[1] > relative_budget:
        return "full_fit_disagreement", values
    errors, differences = [], []
    for capture in captures:
        heldout = np.array([key == capture for key in keys])
        fit = triangulate_rays(centers[~heldout], directions[~heldout])
        if fit is None:
            return "degenerate_leave_one_out", values
        differences.append(float(np.linalg.norm(fit - point.xyz) / distance))
        local = np.einsum("nij,nj->ni", rotations[heldout], fit - centers[heldout])
        if not np.all(np.isfinite(local)) or np.any(local[:, 2] <= 0):
            return "invalid_leave_one_out_projection", values
        selected_cameras = [camera for camera, selected in zip(cameras, heldout, strict=True) if selected]
        predicted = np.array([camera_rays_to_pixels(camera.model, camera.params, ray[None])[0]
                              for camera, ray in zip(selected_cameras, local, strict=True)])
        errors.extend(np.linalg.norm(predicted - pixels[heldout], axis=1).tolist())
    values[2:5] = [max(differences), np.percentile(errors, 95), max(errors)]
    if not np.all(np.isfinite(values)):
        return "invalid_leave_one_out_projection", values
    if values[3] > cross_limit or values[4] > 2 * cross_limit:
        return "cross_reprojection", values
    return "keep", values


def assess_point(point, views, records, *, relative_budget, pixel_sigma, cross_limit, policy="split"):
    if policy not in {"split", "full_track"}:
        raise ValueError(f"unknown assessment policy: {policy}")
    values = np.full(len(METRIC_COLUMNS), np.nan)
    observations = sorted(point.track, key=lambda item: (
        records[views[item[0]]["image"].name]["source_id"],
        records[views[item[0]]["image"].name]["capture_index"]))
    keys = [(records[views[i]["image"].name]["source_id"],
             records[views[i]["image"].name]["capture_index"]) for i, _ in observations]
    captures = list(dict.fromkeys(keys))
    values[0] = len(captures)
    if len(captures) < (4 if policy == "split" else 3):
        return "insufficient_captures", values
    centers = np.array([views[i]["center"] for i, _ in observations])
    rotations = np.array([views[i]["rotation"] for i, _ in observations])
    cameras = [views[i]["camera"] for i, _ in observations]
    pixels = np.array([[views[i]["image"].points2D[j].x, views[i]["image"].points2D[j].y]
                       for i, j in observations])
    directions = np.array([pixels_to_camera_rays(camera.model, camera.params, pixel[None])[0] @ rotation
                           for camera, pixel, rotation in zip(cameras, pixels, rotations, strict=True)])
    distance = float(np.median(np.linalg.norm(np.asarray(point.xyz) - centers, axis=1)))
    if not np.isfinite(distance) or distance <= 0:
        return "invalid_geometry", values
    original_projection = project_and_jacobian(point.xyz, centers, rotations, cameras)
    if original_projection is None:
        return "invalid_original_projection", values
    original_errors = np.linalg.norm(original_projection[0] - pixels, axis=1)
    values[6:] = [np.percentile(original_errors, 95), original_errors.max()]
    if values[6] > cross_limit or values[7] > 2 * cross_limit:
        return "original_reprojection", values
    if policy == "full_track":
        return assess_full_track(point, centers, rotations, cameras, pixels, directions, keys, captures,
                                 distance, original_projection, values, relative_budget=relative_budget,
                                 pixel_sigma=pixel_sigma, cross_limit=cross_limit)
    if len({key[0] for key in captures}) != 1:
        capture_centers = np.array([centers[[key == capture for key in keys]].mean(axis=0)
                                    for capture in captures])
        centered = capture_centers - capture_centers.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        captures = [captures[i] for i in np.argsort(centered @ axis, kind="stable")]
    capture_order = {key: i for i, key in enumerate(captures)}
    order = np.array([capture_order[key] for key in keys])
    splits = [order < len(captures) // 2, order % 2 == 0]
    split_difference = original_difference = uncertainty = 0.0
    cross_errors = []
    for half in splits:
        fits = [triangulate_rays(centers[mask], directions[mask]) for mask in (half, ~half)]
        if any(fit is None for fit in fits):
            return "degenerate_split", values
        split_difference = max(split_difference, float(np.linalg.norm(fits[0] - fits[1]) / distance))
        for fit, mask in zip(fits, (half, ~half), strict=True):
            original_difference = max(original_difference, float(np.linalg.norm(fit - point.xyz) / distance))
            projection = project_and_jacobian(fit, centers, rotations, cameras)
            if projection is None:
                return "invalid_projection", values
            predicted, jacobian = projection
            cross_errors.extend(np.linalg.norm(predicted[~mask] - pixels[~mask], axis=1).tolist())
            uncertainty = max(uncertainty, relative_radius95(jacobian[mask], pixel_sigma, distance))
    values[1:6] = [split_difference, original_difference, np.percentile(cross_errors, 95),
                  max(cross_errors), uncertainty]
    if split_difference > relative_budget or original_difference > relative_budget:
        return "split_disagreement", values
    if values[3] > cross_limit or values[4] > 2 * cross_limit:
        return "cross_reprojection", values
    if uncertainty > relative_budget:
        return "conditional_uncertainty", values
    return "keep", values


def retain_points(reconstruction, retained):
    removed = set(reconstruction.points3D) - retained
    reconstruction.points3D = {pid: point for pid, point in reconstruction.points3D.items() if pid in retained}
    for image in reconstruction.images.values():
        for observation in image.points2D:
            if observation.point3D_id in removed:
                observation.point3D_id = 2**64 - 1
    validate_tracks(reconstruction)


def validate_tracks(reconstruction):
    for pid, point in reconstruction.points3D.items():
        if not np.all(np.isfinite(point.xyz)) or len(set(point.track)) != len(point.track):
            raise ValueError(f"invalid point geometry or duplicate track: {pid}")
        for image_id, index in point.track:
            if reconstruction.images[image_id].points2D[index].point3D_id != pid:
                raise ValueError(f"point-to-image track mismatch: {pid}")
    for image_id, image in reconstruction.images.items():
        for index, observation in enumerate(image.points2D):
            pid = observation.point3D_id
            if pid in {-1, 2**64 - 1}:
                continue
            if pid not in reconstruction.points3D or (image_id, index) not in reconstruction.points3D[pid].track:
                raise ValueError(f"image-to-point track mismatch: {image_id}:{index}")


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
