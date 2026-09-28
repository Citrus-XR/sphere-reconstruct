"""Refit primary-only points while preserving the existing primary camera trajectory."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pycolmap
from experiment_support import launch_detached, write_json
from sphere_reconstruct.colmap.model import Camera
from sphere_reconstruct.colmap.point_stability import assess_observations


def refit_primary(reconstruction, records, primary_source_id, *, max_error, relative_error=0.02, pixel_sigma=1.0, progress):
    primary_ids = {i for i in reconstruction.reg_image_ids()
                   if records[reconstruction.images[i].name]["source_id"] == primary_source_id}
    if not primary_ids:
        raise ValueError("the source model has no registered primary images")
    primary_frames = {reconstruction.images[i].frame_id for i in primary_ids}
    other_frames = {reconstruction.images[i].frame_id for i in reconstruction.reg_image_ids() if i not in primary_ids}
    if primary_frames & other_frames:
        raise ValueError("primary and supplemental images cannot share a frame in this experiment")
    original_poses = {i: reconstruction.images[i].cam_from_world().matrix().copy() for i in primary_ids}
    for frame_id in other_frames:
        reconstruction.deregister_frame(frame_id)
    before = len(reconstruction.points3D)
    rays = {}
    poses = {}
    for image_id in sorted(primary_ids):
        image = reconstruction.images[image_id]
        pixels = np.asarray([point.xy for point in image.points2D], dtype=float)
        normalized = image.camera.cam_from_img(pixels)
        directions = np.column_stack([normalized, np.ones(len(normalized))])
        rays[image_id] = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        poses[image_id] = image.cam_from_world().matrix()
    rejected = []
    triangulated = 0
    for index, (point_id, point) in enumerate(reconstruction.points3D.items(), 1):
        track = list(point.track.elements)
        if any(element.image_id not in primary_ids for element in track):
            raise RuntimeError(f"supplemental observation survived deregistration: {point_id}")
        captures = {records[reconstruction.images[element.image_id].name]["capture_index"] for element in track}
        if len(captures) < 3:
            rejected.append(point_id)
            continue
        directions = np.asarray([rays[element.image_id][element.point2D_idx] for element in track])
        if not np.isfinite(directions).all():
            rejected.append(point_id)
            continue
        xyz = pycolmap.triangulate_multi_view_point([poses[element.image_id] for element in track], directions)
        if xyz is None or not np.isfinite(xyz).all():
            rejected.append(point_id)
            continue
        point.xyz = xyz
        triangulated += 1
        if index % 10000 == 0:
            progress(index, before)
    for point_id in rejected:
        reconstruction.delete_point3D(point_id)
    del rays
    observation_manager = pycolmap.ObservationManager(reconstruction)
    removed_observations = observation_manager.filter_points3D_with_large_reprojection_error(
        max_error, set(reconstruction.points3D)
    )
    cameras = {
        i: Camera(i, camera.model.name, camera.width, camera.height, list(camera.params))
        for i, camera in reconstruction.cameras.items()
    }
    centers = {i: -pose[:, :3].T @ pose[:, 3] for i, pose in poses.items()}
    reasons = Counter()
    rejected = []
    for index, (point_id, point) in enumerate(reconstruction.points3D.items(), 1):
        track = sorted(point.track.elements, key=lambda element: records[
            reconstruction.images[element.image_id].name]["capture_index"])
        images = [reconstruction.images[element.image_id] for element in track]
        reason, _ = assess_observations(
            point.xyz,
            np.asarray([centers[element.image_id] for element in track]),
            np.asarray([poses[element.image_id][:, :3] for element in track]),
            [cameras[image.camera_id] for image in images],
            np.asarray([image.points2D[element.point2D_idx].xy
                        for image, element in zip(images, track, strict=True)]),
            [(primary_source_id, records[image.name]["capture_index"]) for image in images],
            relative_budget=relative_error, pixel_sigma=pixel_sigma,
            cross_limit=max_error, policy="full_track",
        )
        reasons[reason] += 1
        if reason != "keep":
            rejected.append(point_id)
        if index % 10000 == 0:
            progress(index, len(reconstruction.points3D))
    for point_id in rejected:
        removed_observations += reconstruction.points3D[point_id].track.length()
        reconstruction.delete_point3D(point_id)
    reconstruction.update_point_3d_errors()
    for image_id, matrix in original_poses.items():
        if not np.array_equal(reconstruction.images[image_id].cam_from_world().matrix(), matrix):
            raise RuntimeError(f"reference camera pose changed: {image_id}")
    return {"primary_images": len(primary_ids), "points_after_removing_phone_observations": before,
            "points_triangulated_from_primary_only": triangulated, "retained_points": len(reconstruction.points3D),
            "filtered_observations": removed_observations, "max_reprojection_error": max_error,
            "point_policy": "full_track", "relative_error": relative_error, "pixel_sigma": pixel_sigma,
            "reason_counts": dict(reasons), "primary_poses_unchanged": True}


def execute(args, status):
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    records = {row["name"]: row for row in catalog["images"]}
    reconstruction = pycolmap.Reconstruction(str(args.model))

    def progress(current, total):
        status.update(phase="primary_only_triangulation", current=current, total=total)
        write_json(args.output / "status.json", status)
        print(f"TRIANGULATE {current}/{total}", flush=True)

    report = {"strategy": "fixed_existing_primary_poses_with_primary_only_point_refit",
              "source_model": str(args.model), "pycolmap": pycolmap.__version__,
              "limitations": [
                  "Primary camera poses and track identities originate from the joint reconstruction.",
                  "Only the point coordinates are re-estimated using primary image observations alone.",
                  "This is a conditional reference, not a wholly independent primary reconstruction or ground truth.",
              ]}
    report.update(refit_primary(reconstruction, records, catalog["primary_source_id"], max_error=args.max_error,
                               relative_error=args.relative_error, pixel_sigma=args.pixel_sigma, progress=progress))
    sparse = args.output / "sparse"
    sparse.mkdir()
    reconstruction.write(str(sparse))
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "catalog", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-error", type=float, default=2.0)
    parser.add_argument("--relative-error", type=float, default=0.02)
    parser.add_argument("--pixel-sigma", type=float, default=1.0)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    thresholds = [args.max_error, args.relative_error, args.pixel_sigma]
    if not np.isfinite(thresholds).all() or min(thresholds) <= 0:
        parser.error("point quality thresholds must be positive and finite")
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
