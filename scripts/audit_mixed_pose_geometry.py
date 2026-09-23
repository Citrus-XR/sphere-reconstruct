"""Compare mixed-source poses against an image-only primary reference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import traceback
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pycolmap
from audit_phone_primary_support import primary_correspondences, score
from experiment_support import launch_detached, write_json


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {"count": len(values), "nonfinite": int(len(values) - len(finite))}
    if len(finite):
        result.update(median=float(np.median(finite)), p95=float(np.percentile(finite, 95)),
                      maximum=float(np.max(finite)))
    return result


def align_to_primary(model, primary, records):
    from sphere_reconstruct.colmap.sequence_alignment import robust_similarity

    common = sorted(set(model.reg_image_ids()) & set(primary.reg_image_ids()))
    if len(common) < 40:
        raise ValueError("alignment requires at least forty common primary images")
    for image_id in common:
        if model.images[image_id].name != primary.images[image_id].name:
            raise ValueError(f"image identity changed: {image_id}")
    centers = np.array([model.images[i].projection_center() for i in common])
    target = np.array([primary.images[i].projection_center() for i in common])
    captures = np.array([records[model.images[i].name]["capture_index"] for i in common])
    capture_ids = sorted(set(captures.tolist()))
    capture_centers = np.array([target[captures == capture].mean(axis=0) for capture in capture_ids])
    steps = np.linalg.norm(np.diff(capture_centers, axis=0), axis=1)
    consecutive = np.diff(capture_ids) == 1
    if not consecutive.any():
        raise ValueError("primary alignment requires consecutive reference captures")
    normal_step = float(np.median(steps[consecutive]))
    fit = (captures // 20) % 2 == 0
    heldout = ~fit
    if min(fit.sum(), heldout.sum()) < 20:
        raise ValueError("alignment requires twenty fit and twenty heldout primary images")
    (scale, rotation, translation), _, inliers = robust_similarity(
        centers[fit], target[fit], normal_step * 2,
    )
    model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation.T), translation))
    center_errors = np.linalg.norm(scale * centers @ rotation + translation - target, axis=1)
    angles = [rotation_distance(model.images[i].cam_from_world(), primary.images[i].cam_from_world())
              for i in common]
    return {"scale": scale, "row_rotation": rotation.tolist(), "translation": translation.tolist(),
            "fit_images": int(fit.sum()), "fit_inliers": int(inliers.sum()),
            "heldout_images": int(heldout.sum()), "primary_step": normal_step,
            "heldout_center_errors": distribution(center_errors[heldout]),
            "heldout_rotation_degrees": distribution(np.asarray(angles)[heldout])}


def rotation_distance(first, second):
    relative = first.rotation.matrix() @ second.rotation.matrix().T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def correspondence_score(camera, pose, pixels, xyz):
    from sphere_reconstruct.colmap.temporal_pose import reprojection_errors

    if not len(pixels):
        return {"count": 0, "inliers": 0, "supported": False}
    errors = reprojection_errors(camera, pose, pixels, xyz)
    inliers = errors <= 4
    coverage = np.ptp(pixels[inliers], axis=0) / [camera.width, camera.height] if inliers.any() else np.zeros(2)
    return {"count": len(pixels), "inliers": int(inliers.sum()), "ratio": float(inliers.mean()),
            "coverage": coverage.tolist(), "clipped_error": float(np.minimum(errors, 4).mean()),
            "residual_px": distribution(errors),
            "supported": bool(inliers.sum() >= 20 and inliers.mean() >= 0.25 and min(coverage) >= 0.15)}


def heldout_pnp(camera, pixels, xyz, image_id, *, refine_focal=False):
    if len(pixels) < 80:
        return None, {"reason": "insufficient_correspondences", "count": len(pixels)}
    rng = np.random.default_rng(image_id)
    permutation = rng.permutation(len(pixels))
    fit = permutation[:len(pixels) * 2 // 3]
    heldout = permutation[len(pixels) * 2 // 3:]
    # A copy keeps PnP's camera mutation from changing the comparison models.
    calibration = pycolmap.Camera(camera.todict())
    result = pycolmap.estimate_and_refine_absolute_pose(
        pixels[fit], xyz[fit], calibration,
        {"estimate_focal_length": refine_focal, "ransac": {
            "max_error": 4.0, "min_inlier_ratio": 0.1, "min_num_trials": 200,
            "max_num_trials": 10000, "random_seed": 0,
        }},
        {"refine_focal_length": refine_focal, "refine_extra_params": False, "use_position_prior": False},
    )
    if result is None:
        return None, {"reason": "pnp_failed", "fit_count": len(fit), "heldout_count": len(heldout)}
    pose = result["cam_from_world"]
    return pose, {"reason": "estimated", "camera_params": calibration.params.tolist(),
                  "fit": correspondence_score(calibration, pose, pixels[fit], xyz[fit]),
                  "heldout": correspondence_score(calibration, pose, pixels[heldout], xyz[heldout]),
                  "heldout_indices": heldout.tolist(), "matrix": pose.matrix().tolist()}


def trajectory_comparison(models, records, ordered_sources):
    result = {}
    for source_id in ordered_sources:
        expected = sorted((record["capture_index"], name) for name, record in records.items()
                          if record["source_id"] == source_id)
        if len(expected) < 2 or len({capture for capture, _ in expected}) != len(expected):
            raise ValueError("ordered source must contain one image per capture and at least two captures")
        names_by_model = {key: {image.name: image for image in model.images.values() if image.has_pose}
                          for key, model in models.items()}
        pairs = [(first[1], second[1]) for first, second in pairwise(expected)
                 if all(first[1] in images and second[1] in images for images in names_by_model.values())]
        result[source_id] = {"expected_steps": len(expected) - 1, "common_steps": len(pairs), "models": {}}
        for label, images in names_by_model.items():
            steps = [{"from": first, "to": second,
                      "distance": float(np.linalg.norm(images[first].projection_center() - images[second].projection_center())),
                      "rotation_degrees": rotation_distance(images[first].cam_from_world(), images[second].cam_from_world())}
                     for first, second in pairs]
            result[source_id]["models"][label] = {
                "distance": distribution([row["distance"] for row in steps]),
                "rotation": distribution([row["rotation_degrees"] for row in steps]),
                "largest_steps": sorted(steps, key=lambda row: row["distance"], reverse=True)[:30],
                "steps": steps,
            }
    return result


def execute(args, status):
    sys.path.insert(0, str(args.backend_src))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in catalog["images"]}
    primary = pycolmap.Reconstruction(str(args.primary))
    if any(records[image.name]["source_id"] != catalog["primary_source_id"] for image in primary.images.values() if image.has_pose):
        raise ValueError("reference geometry must contain primary images only")
    baseline = pycolmap.Reconstruction(str(args.baseline))
    candidate = pycolmap.Reconstruction(str(args.candidate))
    report = {"strategy": "primary_geometry_image_only", "pycolmap": pycolmap.__version__,
              "primary_model": str(args.primary), "baseline_model": str(args.baseline), "candidate_model": str(args.candidate),
              "alignment": {"baseline": align_to_primary(baseline, primary, records),
                            "candidate": align_to_primary(candidate, primary, records)},
              "limitations": [
                  "Reference geometry is reconstructed from primary images, not surveyed truth.",
                  "Primary correspondences can contain repeated-texture mismatches.",
                  "PnP holds out correspondences; mapper may already have used those image matches.",
                  "Trajectory metrics use explicit ordered-source assumptions and common pairs only.",
              ], "images": []}
    if args.reference_report is not None:
        report["reference_provenance"] = json.loads(args.reference_report.read_text(encoding="utf-8"))
        report["limitations"].extend(report["reference_provenance"]["limitations"])
    write_json(args.output / "report.json", report)
    phone_ids = {i for i, image in baseline.images.items()
                 if image.has_pose and records[image.name]["source_id"] != catalog["primary_source_id"]}
    candidate_ids = set(candidate.reg_image_ids())
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        cohorts = primary_correspondences(primary, baseline, db, records, args.output, status, query_ids=phone_ids)
    np.savez_compressed(args.output / "cohorts.npz", **{
        key: array for image_id, (pixels, xyz, _) in cohorts.items()
        for key, array in [(f"pixels_{image_id}", pixels), (f"xyz_{image_id}", xyz)]
    })
    write_json(args.output / "supporters.json", {str(i): [sorted(s) for s in cohort[2]] for i, cohort in cohorts.items()})
    for index, (image_id, (pixels, xyz, supporters)) in enumerate(cohorts.items()):
        before = baseline.images[image_id]
        row = {"image_id": image_id, "name": before.name, "source_id": records[before.name]["source_id"],
               "baseline": score(before, pixels, xyz, supporters), "registered_candidate": image_id in candidate_ids}
        after = candidate.images[image_id] if image_id in candidate_ids else None
        if after is not None:
            if before.name != after.name:
                raise ValueError(f"candidate identity mismatch: {image_id}")
            row.update(candidate=score(after, pixels, xyz, supporters),
                       center_shift=float(np.linalg.norm(after.projection_center() - before.projection_center())),
                       rotation_shift_degrees=rotation_distance(after.cam_from_world(), before.cam_from_world()))
        camera = after.camera if after is not None else before.camera
        pose, hypothesis = heldout_pnp(camera, pixels, xyz, image_id)
        if pose is not None:
            heldout = np.asarray(hypothesis["heldout_indices"], dtype=int)
            hypothesis["baseline_heldout"] = correspondence_score(before.camera, before.cam_from_world(), pixels[heldout], xyz[heldout])
            if after is not None:
                hypothesis["candidate_heldout"] = correspondence_score(after.camera, after.cam_from_world(), pixels[heldout], xyz[heldout])
                hypothesis["candidate_center_difference"] = float(np.linalg.norm(pose.inverse().translation - after.projection_center()))
                hypothesis["candidate_rotation_difference"] = rotation_distance(pose, after.cam_from_world())
        row["pnp"] = hypothesis
        if args.probe_focal:
            _, row["pnp_focal"] = heldout_pnp(camera, pixels, xyz, image_id, refine_focal=True)
        report["images"].append(row)
        if index % 50 == 0:
            status.update(phase="score_and_pnp", processed=index, total=len(cohorts))
            write_json(args.output / "status.json", status)
            print(f"SCORE {index}/{len(cohorts)}", flush=True)
    report["trajectory"] = trajectory_comparison({"baseline": baseline, "candidate": candidate}, records, args.ordered_source)
    report["summary"] = {}
    for source in catalog["sources"]:
        rows = [row for row in report["images"] if row["source_id"] == source["id"]]
        if not rows:
            continue
        report["summary"][source["id"]] = {
            "label": source["label"], "baseline_images": len(rows),
            "candidate_images": sum(row["registered_candidate"] for row in rows),
            "common_images": {label: {"supported": sum(row[label]["supported"] for row in rows if row["registered_candidate"]),
                                       "inliers": sum(row[label]["inliers"] for row in rows if row["registered_candidate"])}
                              for label in ["baseline", "candidate"]},
            "pnp_heldout_supported": sum(row["pnp"].get("heldout", {}).get("supported", False) for row in rows),
        }
    write_json(args.output / "report.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["primary", "baseline", "candidate", "catalog", "database", "backend-src", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ordered-source", action="append", default=[], help="Explicit source ID with chronological single-camera images")
    parser.add_argument("--probe-focal", action="store_true", help="Compare per-image focal estimation on the same heldout features")
    parser.add_argument("--reference-report", type=Path, help="Provenance and limits of a conditional primary reference")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "phase": "load", "started_at": datetime.now(UTC).isoformat()}
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
