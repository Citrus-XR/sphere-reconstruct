"""固定 pose の不変条件と、生成に使わない capture の再投影を検証する。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from sphere_reconstruct.colmap import model
from sphere_reconstruct.colmap.point_stability import geometry, triangulate_rays
from sphere_reconstruct.imaging.fisheye_camera import (
    camera_rays_to_pixels,
    pixels_to_camera_rays,
)

MAX_IMAGE_ID = 2_147_483_647


def sparse_fingerprint(directory):
    result = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        path = directory / name
        if name in {"rigs.bin", "frames.bin"} and not path.exists():
            continue
        with path.open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(zip(("p50", "p90", "p95", "p99", "max"), np.percentile(values, [50, 90, 95, 99, 100]).tolist())) if len(values) else None


def max_angle(directions):
    # 反対向きの ray も退化するため角度を 90 度で折り返す。
    # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/geometry/triangulation.cc#L217-L225
    return float(np.degrees(np.arccos(np.min(np.clip(np.abs(directions @ directions.T), 0, 1)))))


def reprojection(point, views, observations):
    errors = []
    for image_id, index in observations:
        view = views[image_id]
        camera = view["camera"]
        ray = view["rotation"] @ (point - view["center"])
        if ray[2] <= 0:
            errors.append(float("inf"))
            continue
        prediction = camera_rays_to_pixels(camera.model, camera.params, ray[None])[0]
        pixel = view["image"].points2D[index]
        errors.append(float(np.linalg.norm(prediction - [pixel.x, pixel.y])))
    return errors


def temporal_audit(reconstruction, records, sample_points):
    views = geometry(reconstruction)
    values = {"relative_xyz_disagreement": [], "cross_pixel_error": [],
              "half_minimum_angle_deg": [], "maximum_angle_deg": []}
    counts = Counter()
    point_rows = []
    points = sorted(reconstruction.points3D.values(), key=lambda p: p.point3D_id)
    stride = max(1, int(np.ceil(len(points) / sample_points)))
    for point in points[::stride]:
        counts["sampled_points"] += 1
        observations = sorted(point.track, key=lambda item: (
            records[views[item[0]]["image"].name]["source_id"],
            records[views[item[0]]["image"].name]["capture_index"]))
        keys = [(records[views[i]["image"].name]["source_id"], records[views[i]["image"].name]["capture_index"])
                for i, _ in observations]
        unique = list(dict.fromkeys(keys))
        if len({key[0] for key in keys}) != 1:
            counts["mixed_source_temporal_split_undefined"] += 1
            continue
        centers = np.array([views[i]["center"] for i, _ in observations])
        directions = []
        for image_id, index in observations:
            view = views[image_id]
            camera = view["camera"]
            pixel = view["image"].points2D[index]
            directions.append(pixels_to_camera_rays(camera.model, camera.params, [[pixel.x, pixel.y]])[0] @ view["rotation"])
        directions = np.array(directions)
        values["maximum_angle_deg"].append(max_angle(directions))
        if len(unique) < 4:
            counts["insufficient_captures"] += 1
            continue
        early_keys = set(unique[:len(unique) // 2])
        early = np.array([key in early_keys for key in keys])
        a = triangulate_rays(centers[early], directions[early])
        b = triangulate_rays(centers[~early], directions[~early])
        angle = min(max_angle(directions[early]), max_angle(directions[~early]))
        values["half_minimum_angle_deg"].append(angle)
        if a is None or b is None:
            counts["degenerate_halves"] += 1
            continue
        counts["solved_halves"] += 1
        distance = np.median(np.linalg.norm(np.asarray(point.xyz) - centers, axis=1))
        difference = float(np.linalg.norm(a - b) / distance)
        errors = reprojection(a, views, [obs for obs, flag in zip(observations, early) if not flag])
        errors += reprojection(b, views, [obs for obs, flag in zip(observations, early) if flag])
        if any(not np.isfinite(error) for error in errors):
            counts["cross_projection_behind_camera"] += 1
        values["relative_xyz_disagreement"].append(difference)
        values["cross_pixel_error"].extend(error for error in errors if np.isfinite(error))
        point_rows.append([point.point3D_id, len(unique), difference, float(np.median(errors)), angle])
        counts["xyz_disagreement_over_10pct"] += difference > 0.1
        counts["xyz_disagreement_over_25pct"] += difference > 0.25
    return ({"counts": dict(counts), "distributions": {key: quantiles(value) for key, value in values.items()}},
            np.asarray(point_rows, dtype=float).reshape((-1, 5)))


def load_links(database, records, validation_names):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        capture_codes = {key: number for number, key in enumerate(sorted({
            (record["source_id"], record["capture_index"]) for record in records.values()}))}
        blocks = []
        pixels = {}
        for pair_id, count, blob in db.execute("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows > 0"):
            a, b = divmod(pair_id, MAX_IMAGE_ID)
            hold_a, hold_b = names[a] in validation_names, names[b] in validation_names
            if hold_a == hold_b:
                continue
            rows = np.frombuffer(blob, dtype=np.uint32).reshape(count, 2)
            validation_id, training_id = (a, b) if hold_a else (b, a)
            validation_index, training_index = (rows[:, 0], rows[:, 1]) if hold_a else (rows[:, 1], rows[:, 0])
            record = records[names[training_id]]
            capture = capture_codes[(record["source_id"], record["capture_index"])]
            blocks.append(np.column_stack([np.full(count, validation_id), validation_index,
                                            np.full(count, training_id), training_index, np.full(count, capture)]))
        for image_id, rows, cols, blob in db.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            pixels[image_id] = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2].copy()
    if not blocks:
        raise ValueError("no held-out cross-capture correspondences")
    links = np.concatenate(blocks).astype(np.int64)
    witnesses = np.unique(links[:, [0, 1, 4]], axis=0)
    keys, counts = np.unique(witnesses[:, :2], axis=0, return_counts=True)
    eligible = keys[counts >= 2]
    return names, pixels, links, eligible


def fixed_pose_invariants(reference, candidate):
    original = {image.name: image for image in reference.images.values()}
    current = {image.name: image for image in candidate.images.values()}
    if original.keys() != current.keys():
        raise ValueError("registered image set changed")
    max_center = max_rotation = max_params = 0.0
    for name, before in original.items():
        after = current[name]
        max_center = max(max_center, float(np.linalg.norm(np.asarray(before.camera_center) - after.camera_center)))
        max_rotation = max(max_rotation, float(np.max(np.abs(np.asarray(model.qvec_to_rotation(before.qvec)) - model.qvec_to_rotation(after.qvec)))))
        ca, cb = reference.cameras[before.camera_id], candidate.cameras[after.camera_id]
        if (ca.model, ca.width, ca.height) != (cb.model, cb.width, cb.height):
            raise ValueError("camera model or image dimensions changed")
        max_params = max(max_params, float(np.max(np.abs(np.asarray(ca.params) - cb.params))))
    if max(max_center, max_rotation, max_params) > 1e-9:
        raise ValueError(f"fixed-pose violation: center={max_center}, rotation={max_rotation}, intrinsics={max_params}")
    return {"images": len(current), "max_center_change": max_center,
            "max_rotation_matrix_change": max_rotation, "max_intrinsic_change": max_params}


def heldout_audit(reconstruction, names, pixels, links, eligible, validation_names):
    by_name = {image.name: image for image in reconstruction.images.values()}
    db_by_name = {name: image_id for image_id, name in names.items()}
    arrays = {}
    max_xy_difference = 0.0
    for image in reconstruction.images.values():
        image_id = db_by_name[image.name]
        if len(image.points2D) != len(pixels[image_id]):
            raise ValueError(f"keypoint row count changed: {image.name}")
        xy = np.array([[point.x, point.y] for point in image.points2D])
        max_xy_difference = max(max_xy_difference, float(np.max(np.abs(xy - pixels[image_id]))))
        arrays[image_id] = np.fromiter((point.point3D_id if point.point3D_id in reconstruction.points3D else -1
                                       for point in image.points2D), dtype=np.int64)
    if max_xy_difference > 1e-4:
        raise ValueError(f"keypoint coordinates changed: {max_xy_difference}")
    point_ids = np.full(len(links), -1, dtype=np.int64)
    for image_id in np.unique(links[:, 2]):
        rows = links[:, 2] == image_id
        point_ids[rows] = arrays[image_id][links[rows, 3]]
    valid = point_ids >= 0
    votes = np.unique(np.column_stack([links[valid, 0], links[valid, 1], point_ids[valid], links[valid, 4]]), axis=0)
    proposals, support = np.unique(votes[:, :3], axis=0, return_counts=True)
    proposals = proposals[support >= 2]
    _, first, multiplicity = np.unique(proposals[:, :2], axis=0, return_index=True, return_counts=True)
    unambiguous = proposals[first[multiplicity == 1]]
    proposal_ranges = np.empty(len(proposals))
    proposal_errors = np.empty(len(proposals))
    for image_id in np.unique(proposals[:, 0]):
        rows = proposals[:, 0] == image_id
        image = by_name[names[image_id]]
        camera = reconstruction.cameras[image.camera_id]
        xyz = np.array([reconstruction.points3D[int(pid)].xyz for pid in proposals[rows, 2]])
        local = xyz @ np.asarray(model.qvec_to_rotation(image.qvec)).T + image.tvec
        proposal_ranges[rows] = np.linalg.norm(local, axis=1)
        predicted = camera_rays_to_pixels(camera.model, camera.params, local)
        proposal_errors[rows] = np.linalg.norm(predicted - pixels[image_id][proposals[rows, 1]], axis=1)
        proposal_errors[np.flatnonzero(rows)[local[:, 2] <= 0]] = np.inf
    plausible = proposal_errors <= 2.0
    plausible_count = np.add.reduceat(plausible.astype(int), first)
    minimum_range = np.minimum.reduceat(np.where(plausible, proposal_ranges, np.inf), first)
    maximum_range = np.maximum.reduceat(np.where(plausible, proposal_ranges, -np.inf), first)
    contested = plausible_count >= 2
    contested_ratios = maximum_range[contested] / minimum_range[contested]
    errors = np.full(len(unambiguous), np.inf)
    per_image = []
    for image_id in np.unique(unambiguous[:, 0]):
        rows = unambiguous[:, 0] == image_id
        image = by_name[names[image_id]]
        camera = reconstruction.cameras[image.camera_id]
        xyz = np.array([reconstruction.points3D[int(pid)].xyz for pid in unambiguous[rows, 2]])
        camera_xyz = xyz @ np.asarray(model.qvec_to_rotation(image.qvec)).T + image.tvec
        predicted = camera_rays_to_pixels(camera.model, camera.params, camera_xyz)
        expected = pixels[image_id][unambiguous[rows, 1]]
        distance = np.linalg.norm(predicted - expected, axis=1)
        distance[camera_xyz[:, 2] <= 0] = np.inf
        errors[rows] = distance
        per_image.append({"name": names[image_id], "predictions": len(distance),
                          "within_2px": int(np.count_nonzero(distance <= 2)), "error_px": quantiles(distance)})
    leaks = sum(point.point3D_id in reconstruction.points3D for name, image in by_name.items()
                if name in validation_names for point in image.points2D)
    report = {"eligible_validation_keypoints": len(eligible), "predicted_keypoints": len(errors),
            "ambiguous_keypoints": int(np.count_nonzero(multiplicity > 1)),
            "unsupported_or_ambiguous_keypoints": len(eligible) - len(errors),
            "prediction_coverage": len(errors) / len(eligible), "error_px": quantiles(errors),
            "invalid_projections": int(np.count_nonzero(~np.isfinite(errors))),
            "within_px": {str(t): int(np.count_nonzero(errors <= t)) for t in [1, 2, 4, 8]},
            "within_px_ratio_of_all_eligible": {str(t): float(np.count_nonzero(errors <= t) / len(eligible)) for t in [1, 2, 4, 8]},
            "max_keypoint_xy_difference": max_xy_difference, "validation_observations_in_model": leaks,
            "ambiguous_with_two_plausible_depths": int(np.count_nonzero(contested)),
            "plausible_depth_ratio": quantiles(contested_ratios),
            "ambiguous_depths_differ_over_10pct": int(np.count_nonzero(contested_ratios > 1.1)),
            "ambiguous_depths_differ_over_25pct": int(np.count_nonzero(contested_ratios > 1.25)),
            "per_image": per_image}
    return report, unambiguous, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--sample-points", default=10000, type=int)
    parser.add_argument("--cases", nargs="+", default=["fixed_all", "heldout_all", "heldout_wide2"])
    parser.add_argument("--skip-reference-audit", action="store_true")
    args = parser.parse_args()
    spec = json.loads((args.experiment / "input_spec.json").read_text(encoding="utf-8"))
    records = {item["name"]: item for item in spec["images"]}
    validation_names = set(json.loads((args.experiment / "validation_names.json").read_text()))
    names, pixels, links, eligible = load_links(args.database, records, validation_names)
    reference = model.read_model(args.experiment / "reference")
    results_path = args.experiment / "audit.json"
    results = json.loads(results_path.read_text()) if args.skip_reference_audit and results_path.exists() else {}
    for case in args.cases if args.skip_reference_audit else ["reference", *args.cases]:
        print(f"Auditing {case}", flush=True)
        candidate = reference if case == "reference" else model.read_model(args.experiment / case / "sparse" / "0")
        heldout, predictions, errors = heldout_audit(candidate, names, pixels, links, eligible, validation_names)
        np.savez_compressed(args.experiment / f"validation_{case}.npz", predictions=predictions, error_px=errors)
        temporal, point_rows = temporal_audit(candidate, records, args.sample_points)
        candidate_path = args.experiment / "reference" if case == "reference" else args.experiment / case / "sparse" / "0"
        np.savez_compressed(args.experiment / f"stability_{case}.npz", values=point_rows,
                            model_fingerprint=json.dumps(sparse_fingerprint(candidate_path), sort_keys=True),
                            columns=np.array(["point_id", "captures", "relative_xyz_disagreement", "cross_median_px", "half_minimum_angle_deg"]))
        result = {"summary": candidate.summary(), "invariants": fixed_pose_invariants(reference, candidate),
                  "temporal_split": temporal, "heldout": heldout}
        if case.startswith("heldout_") and result["heldout"]["validation_observations_in_model"]:
            raise ValueError("validation observations leaked into triangulation")
        result["heldout"]["independent_point_observations"] = case.startswith("heldout_")
        results[case] = result
        temporary = args.experiment / "audit.tmp"
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.experiment / "audit.json")
        print(json.dumps({"case": case, "points": len(candidate.points3D),
                          "prediction_coverage": result["heldout"]["prediction_coverage"],
                          "heldout_error": result["heldout"]["error_px"]}), flush=True)
        if candidate is not reference:
            del candidate
            gc.collect()


if __name__ == "__main__":
    main()
