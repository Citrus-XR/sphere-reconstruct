"""訓練側 temporal split の許容誤差を明示して、新規観測だけを補完する。"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from audit_fixed_pose import fixed_pose_invariants, sparse_fingerprint
from sphere_reconstruct.colmap import model


def write_ply(path, points):
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for point in points.values():
            stream.write(struct.pack("<fffBBB", *point.xyz, *point.rgb))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("stability", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experiment-reference", type=Path, required=True,
                        help="reference/, input_spec.json, validation_names.json を含む実験 directory")
    parser.add_argument("--relative-error-budgets", type=float, nargs="+", required=True)
    parser.add_argument("--max-cross-error-px", type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.max_cross_error_px) or args.max_cross_error_px <= 0:
        raise ValueError("cross error budget must be positive and finite")
    if any(not 0 < budget < 1 for budget in args.relative_error_budgets):
        raise ValueError("relative error budgets must lie between zero and one")
    cases = [f"heldout_supplement_{round(budget * 100):02d}pct" for budget in args.relative_error_budgets]
    if len(set(cases)) != len(cases):
        raise ValueError("budgets must have distinct whole-percentage case names")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = model.read_model(args.base)
    candidate = model.read_model(args.candidate)
    fixed_pose_invariants(reference, candidate)
    by_name = {image.name: image for image in reference.images.values()}
    image_map = {}
    for image in candidate.images.values():
        original = by_name[image.name]
        if (len(original.points2D) != len(image.points2D)
                or any((a.x, a.y) != (b.x, b.y) for a, b in zip(original.points2D, image.points2D))):
            raise ValueError("keypoint ordering differs")
        image_map[image.image_id] = original.image_id
    with np.load(args.stability) as archive:
        if "model_fingerprint" not in archive:
            raise ValueError("stability archive has no model fingerprint; rerun audit_fixed_pose.py")
        if json.loads(archive["model_fingerprint"].item()) != sparse_fingerprint(args.candidate):
            raise ValueError("stability archive belongs to a different candidate model")
        stability = archive["values"]
        if archive["columns"].tolist() != ["point_id", "captures", "relative_xyz_disagreement",
                                          "cross_median_px", "half_minimum_angle_deg"]:
            raise ValueError("unknown temporal stability schema")
    if stability.ndim != 2 or stability.shape[1] != 5:
        raise ValueError("temporal stability must have five columns")
    shutil.copytree(args.experiment_reference / "reference", args.output / "reference")
    for name in ("input_spec.json", "validation_names.json"):
        shutil.copy2(args.experiment_reference / name, args.output / name)
    results = []
    for budget, case in zip(args.relative_error_budgets, cases):
        eligible = set(stability[(stability[:, 2] <= budget) & (stability[:, 3] <= args.max_cross_error_px), 0].astype(int))
        destination = args.output / case
        target = destination / "sparse" / "0"
        target.mkdir(parents=True)
        next_id = max(reference.points3D, default=0) + 1
        added = []
        restored = []
        conflicts = 0
        for candidate_id in sorted(eligible):
            point = candidate.points3D[candidate_id]
            track = [(image_map[i], index) for i, index in point.track]
            if any(reference.images[i].points2D[index].point3D_id in reference.points3D for i, index in track):
                conflicts += 1
                continue
            identifier = next_id
            next_id += 1
            reference.points3D[identifier] = model.Point3D(identifier, point.xyz, point.rgb, point.error, track)
            for image_id, index in track:
                image_point = reference.images[image_id].points2D[index]
                restored.append((image_point, image_point.point3D_id))
                image_point.point3D_id = identifier
            added.append(identifier)
        model.write_cameras_bin(target / "cameras.bin", reference.cameras)
        model.write_images_bin(target / "images.bin", reference.images)
        model.write_points3D_bin(target / "points3D.bin", reference.points3D)
        for name in ("frames.bin", "rigs.bin"):
            if (args.base / name).exists():
                shutil.copy2(args.base / name, target / name)
        write_ply(destination / "points.ply", reference.points3D)
        result = {"case": case, "base": str(args.base), "candidate": str(args.candidate),
                  "stability": str(args.stability), "relative_error_budget": budget, "max_cross_error_px": args.max_cross_error_px,
                  "train_temporal_eligible_points": len(eligible), "overlap_with_base": conflicts,
                  "added_points": len(added), "total_points": len(reference.points3D)}
        results.append(result)
        (destination / "supplement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        for image_point, old_identifier in restored:
            image_point.point3D_id = old_identifier
        for identifier in added:
            del reference.points3D[identifier]
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
