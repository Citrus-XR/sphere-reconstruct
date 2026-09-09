"""同一 camera / point identity の cleanup 間で画像上の初期点 coverage を比較する。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from audit_fixed_pose import fixed_pose_invariants, model, sparse_fingerprint
from benchmark_fixed_pose import write_json
from export_strict_sparse import validate_tracks


def coverage(reconstruction):
    rows = {}
    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        pixels = np.array([(point.x, point.y) for point in image.points2D
                           if point.point3D_id in reconstruction.points3D], dtype=float).reshape((-1, 2))
        cells = np.floor(pixels / [camera.width, camera.height] * 32).astype(int)
        rows[image.name] = {"observations": len(pixels), "cells": {tuple(cell) for cell in cells}}
    return rows


def execute(source, candidates):
    original = model.read_model(source)
    validate_tracks(original)
    reference = coverage(original)
    results = []
    for candidate in candidates:
        filtered = model.read_model(candidate)
        validate_tracks(filtered)
        invariants = fixed_pose_invariants(original, filtered)
        if set(filtered.points3D) - set(original.points3D):
            raise ValueError("cleanup contains added points")
        for pid, point in filtered.points3D.items():
            old = original.points3D[pid]
            if point.xyz != old.xyz or point.rgb != old.rgb or point.track != old.track:
                raise ValueError(f"cleanup changed point geometry or track: {pid}")
        rows = coverage(filtered)
        images = [{"name": name, "original_observations": reference[name]["observations"],
                   "retained_observations": row["observations"],
                   "original_cells": len(reference[name]["cells"]), "retained_cells": len(row["cells"]),
                   "retained_cell_fraction": (len(row["cells"]) / len(reference[name]["cells"]))
                   if reference[name]["cells"] else None} for name, row in sorted(rows.items())]
        fractions = [row["retained_cell_fraction"] for row in images if row["retained_cell_fraction"] is not None]
        results.append({"candidate": candidate.parent.parent.parent.name,
                        "points": len(filtered.points3D), "source_points": len(original.points3D),
                        "camera_invariants": invariants, "unchanged_surviving_points": True,
                        "empty_images": sum(row["retained_observations"] == 0 for row in images),
                        "retained_cell_fraction_quantiles": dict(zip(["p05", "p50", "p95"],
                                                                     np.percentile(fractions, [5, 50, 95]).tolist())),
                        "model_fingerprint": sparse_fingerprint(candidate), "images": images})
        del filtered
        gc.collect()
    return {"grid": [32, 32], "interpretation": "occupied sparse observation cells, not surface completeness or accuracy",
            "source_fingerprint": sparse_fingerprint(source), "candidates": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.source, args.candidates)
    write_json(args.output, result)
    print(json.dumps([{key: value for key, value in row.items() if key != "images"}
                      for row in result["candidates"]]), flush=True)


if __name__ == "__main__":
    main()
