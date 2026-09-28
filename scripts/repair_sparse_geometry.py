"""画像対応で検証済みの pose と track を適用し、影響点だけを再三角化する。"""

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
from experiment_support import file_hash, launch_detached, write_json
from sphere_reconstruct.colmap.model import Camera
from sphere_reconstruct.colmap.point_stability import (
    assess_observations,
    triangulate_rays,
)
from sphere_reconstruct.imaging.camera_geometry import pixels_to_camera_rays


def observations(reconstruction, track, records):
    elements = sorted(track, key=lambda e: (
        records[reconstruction.images[e.image_id].name]["source_id"],
        records[reconstruction.images[e.image_id].name]["capture_index"],
    ))
    images = [reconstruction.images[e.image_id] for e in elements]
    poses = [image.cam_from_world().matrix() for image in images]
    centers = np.array([-pose[:, :3].T @ pose[:, 3] for pose in poses])
    rotations = np.array([pose[:, :3] for pose in poses])
    cameras = [Camera(image.camera_id, image.camera.model.name, image.camera.width,
                      image.camera.height, list(image.camera.params)) for image in images]
    pixels = np.array([image.points2D[e.point2D_idx].xy for image, e in zip(images, elements, strict=True)])
    keys = [(records[image.name]["source_id"], records[image.name]["capture_index"]) for image in images]
    return centers, rotations, cameras, pixels, keys


def refit_track(reconstruction, track, records, *, relative_error, pixel_sigma, max_error):
    centers, rotations, cameras, pixels, keys = observations(reconstruction, track, records)
    if len(set(keys)) < 3:
        return None, "insufficient_captures"
    directions = np.array([pixels_to_camera_rays(camera, pixel[None])[0] @ rotation
                           for camera, pixel, rotation in zip(cameras, pixels, rotations, strict=True)])
    xyz = triangulate_rays(centers, directions)
    if xyz is None:
        return None, "degenerate_triangulation"
    reason, _ = assess_observations(
        xyz, centers, rotations, cameras, pixels, keys,
        relative_budget=relative_error, pixel_sigma=pixel_sigma, cross_limit=max_error, policy="full_track",
    )
    return xyz, reason


def repair(reconstruction, records, primary_source, poses, recovered, *, progress,
           relative_error=0.02, pixel_sigma=1.0, max_error=2.0):
    original = {i: image.cam_from_world().matrix().copy() for i, image in reconstruction.images.items()}
    original_cameras = {i: camera.params.copy() for i, camera in reconstruction.cameras.items()}
    changed = set()
    for row in poses:
        image_id = row["image_id"]
        if image_id in changed:
            raise ValueError(f"duplicate pose correction: {image_id}")
        image = reconstruction.images[image_id]
        if records[image.name]["source_id"] == primary_source:
            raise ValueError(f"cannot change primary camera: {image.name}")
        if not np.allclose(original[image_id], np.column_stack([row["old_R"], row["old_t"]]), rtol=0, atol=1e-8):
            raise ValueError(f"pose evidence belongs to a different model: {image.name}")
        rotation, translation = np.array(row["R"]), np.array(row["t"])
        if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
            raise ValueError(f"invalid corrected pose: {image.name}")
        shared = [i for i, other in reconstruction.images.items() if other.frame_id == image.frame_id]
        if shared != [image_id]:
            raise ValueError(f"pose correction requires a single-image frame: {image.name}")
        image.frame.set_cam_from_world(image.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation))
        changed.add(image_id)

    affected = [pid for pid, point in reconstruction.points3D.items()
                if any(e.image_id in changed for e in point.track.elements)]
    reasons = Counter()
    moved = []
    for index, pid in enumerate(affected, 1):
        point = reconstruction.points3D[pid]
        xyz, reason = refit_track(reconstruction, point.track.elements, records,
                                 relative_error=relative_error, pixel_sigma=pixel_sigma, max_error=max_error)
        reasons[reason] += 1
        if reason == "keep":
            moved.append(float(np.linalg.norm(xyz - point.xyz)))
            point.xyz = xyz
        else:
            reconstruction.delete_point3D(pid)
        if index % 1000 == 0:
            progress("refit", index, len(affected))

    restored = []
    for index, row in enumerate(recovered, 1):
        track = pycolmap.Track()
        entries = [tuple(entry) for entry in row["track"]]
        if len(set(entries)) != len(entries):
            raise ValueError(f"duplicate recovered observation: {row['old_id']}")
        color = np.asarray(row["rgb"])
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 255):
            raise ValueError(f"invalid recovered point color: {row['old_id']}")
        for image_id, point_index in entries:
            image = reconstruction.images[image_id]
            if records[image.name]["source_id"] != primary_source:
                raise ValueError("recovered track must use primary observations only")
            if not 0 <= point_index < len(image.points2D):
                raise ValueError(f"invalid recovered observation index: {image_id}:{point_index}")
            if image.points2D[point_index].has_point3D():
                raise ValueError(f"recovered observation already belongs to a point: {image_id}:{point_index}")
            track.add_element(image_id, point_index)
        xyz, reason = refit_track(reconstruction, track.elements, records,
                                 relative_error=relative_error, pixel_sigma=pixel_sigma, max_error=max_error)
        if reason != "keep":
            raise ValueError(f"recovered point no longer passes verification: {row['old_id']}: {reason}")
        restored.append(reconstruction.add_point3D(xyz, track, np.array(row["rgb"], dtype=np.uint8)))
        if index % 100 == 0:
            progress("restore", index, len(recovered))
    reconstruction.update_point_3d_errors()
    for image_id, before in original.items():
        if image_id not in changed and not np.array_equal(reconstruction.images[image_id].cam_from_world().matrix(), before):
            raise RuntimeError(f"unrequested image pose changed: {image_id}")
    for camera_id, before in original_cameras.items():
        if not np.array_equal(reconstruction.cameras[camera_id].params, before):
            raise RuntimeError(f"camera calibration changed: {camera_id}")
    return {"changed_images": sorted(changed), "affected_points": len(affected), "refit_reasons": dict(reasons),
            "point_displacement_percentiles": np.percentile(moved, [0, 50, 95, 100]).tolist() if moved else [],
            "restored_point_ids": restored, "preserved_images": len(original), "other_poses_unchanged": True,
            "intrinsics_unchanged": True, "point_policy": "full_track", "relative_error": relative_error,
            "pixel_sigma": pixel_sigma, "max_error": max_error}


def align_to_reference(reconstruction, reference, primary_ids):
    ids = sorted(primary_ids)
    if len(ids) < 3:
        raise ValueError("reference alignment requires at least three primary centers")
    a = np.array([reconstruction.images[i].projection_center() for i in ids])
    b = np.array([reference.images[i].projection_center() for i in ids])
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    if np.linalg.matrix_rank(ac) < 2:
        raise ValueError("reference alignment requires at least three non-collinear primary centers")
    u, singular, vt = np.linalg.svd(bc.T @ ac)
    sign = np.diag([1, 1, np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    scale = float((singular * np.diag(sign)).sum() / np.square(ac).sum())
    translation = b.mean(axis=0) - scale * rotation @ a.mean(axis=0)
    transform = pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation)
    reconstruction.transform(transform)
    for i in ids:
        if not np.allclose(reconstruction.images[i].cam_from_world().matrix(),
                           reference.images[i].cam_from_world().matrix(), atol=1e-7, rtol=0):
            raise ValueError(f"reference does not share the primary trajectory: {i}")
    return {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()}


def execute(args, status):
    catalog = json.loads(args.catalog.read_text(encoding="utf8"))
    records = {row["name"]: row for row in catalog["images"]}
    pose_report = json.loads(args.poses.read_text(encoding="utf8"))
    poses = pose_report["accepted"]
    recovered = json.loads(args.recovered.read_text(encoding="utf8"))["recovered"] if args.recovered else []
    hashes = {str(p.resolve()): file_hash(p) for p in [*args.model.glob("*.bin"), args.catalog, args.poses,
                                                     *([args.recovered] if args.recovered else [])]}
    model = pycolmap.Reconstruction(str(args.model))

    def progress(phase, current, total):
        status.update(phase=phase, current=current, total=total)
        write_json(args.output / "status.json", status)
        print(f"{phase} {current}/{total}", flush=True)

    result = repair(model, records, catalog["primary_source_id"], poses, recovered, progress=progress)
    if args.reference_model:
        reference = pycolmap.Reconstruction(str(args.reference_model))
        hashes.update({str(p.resolve()): file_hash(p) for p in args.reference_model.glob("*.bin")})
        primary_ids = {i for i, im in model.images.items() if records[im.name]["source_id"] == catalog["primary_source_id"]}
        result["transform_to_reference"] = align_to_reference(model, reference, primary_ids)
        del reference
    sparse = args.output / "sparse/0"
    sparse.mkdir(parents=True)
    model.write(str(sparse))
    reread = pycolmap.Reconstruction(str(sparse))
    if len(reread.images) != len(model.images) or len(reread.points3D) != len(model.points3D):
        raise RuntimeError("written candidate changed model counts")
    for i, image in reread.images.items():
        if not np.allclose(image.cam_from_world().matrix(), model.images[i].cam_from_world().matrix(), atol=1e-9, rtol=0):
            raise RuntimeError(f"frame/image pose mismatch after write: {i}")
    for name, digest in hashes.items():
        if file_hash(Path(name)) != digest:
            raise RuntimeError(f"source changed during repair: {name}")
    result.update(strategy="verified_local_pose_and_track_refit", source_hashes=hashes,
                  num_images=len(model.images), num_points3D=len(model.points3D),
                  output_hashes={p.name: file_hash(p) for p in sparse.glob("*.bin")})
    write_json(args.output / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "catalog", "poses", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-model", type=Path)
    parser.add_argument("--recovered", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()}
    write_json(args.output / "status.json", status)
    with (args.output / "worker.log").open("x", encoding="utf8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            execute(args, status)
            status.update(state="completed")
        except BaseException as error:
            status.update(state="failed", error=repr(error))
            traceback.print_exc()
            raise
        finally:
            status["finished_at"] = datetime.now(UTC).isoformat()
            write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
