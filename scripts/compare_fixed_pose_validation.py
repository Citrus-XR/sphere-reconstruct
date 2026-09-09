"""同一留保 keypoint の誤差と、追加・消失した予測を分離して比較する。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def read_predictions(path):
    with np.load(path) as archive:
        rows, errors = archive["predictions"], archive["error_px"]
    keys = np.rec.fromarrays(rows[:, :2].T, names="image,keypoint")
    if len(np.unique(keys)) != len(keys) or len(rows) != len(errors):
        raise ValueError("prediction archive must have one error per unique image/keypoint")
    return keys, errors


def summarize(errors):
    finite = errors[np.isfinite(errors)]
    return {"predictions": len(errors), "within_2px": int(np.count_nonzero(errors <= 2)),
            "invalid": int(np.count_nonzero(~np.isfinite(errors))),
            "median_px": float(np.median(finite)) if len(finite) else None,
            "p95_px": float(np.percentile(finite, 95)) if len(finite) else None}


def compare(base_keys, base_errors, keys, errors):
    _, before, after = np.intersect1d(base_keys, keys, return_indices=True)
    common_base, common_candidate = base_errors[before], errors[after]
    added = ~np.isin(keys, base_keys)
    lost = ~np.isin(base_keys, keys)
    finite = np.isfinite(common_base) & np.isfinite(common_candidate)
    changes = common_candidate[finite] - common_base[finite]
    return {"common_base": summarize(common_base), "common_candidate": summarize(common_candidate),
            "common_max_absolute_error_change": float(np.max(np.abs(changes))) if len(changes) else None,
            "common_worsened_over_1px": int(np.count_nonzero(common_candidate > common_base + 1)),
            "common_improved_over_1px": int(np.count_nonzero(common_base > common_candidate + 1)),
            "added": summarize(errors[added]), "lost": summarize(base_errors[lost]),
            "net_within_2px": int(np.count_nonzero(errors <= 2) - np.count_nonzero(base_errors <= 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    records = {item["name"]: item for item in json.loads(args.spec.read_text(encoding="utf-8"))["images"]}
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as database:
        names = dict(database.execute("SELECT image_id, name FROM images"))
    groups = {}
    for image_id, name in names.items():
        record = records[name]
        key = (record["source_id"], record["sensor_id"])
        groups.setdefault(key, []).append((record["capture_index"], image_id))
    bins = {}
    for (source, sensor), members in groups.items():
        for quarter, part in enumerate(np.array_split(sorted(members), 4)):
            bins[f"{source}/{sensor}/quarter{quarter + 1}"] = part[:, 1] if len(part) else []
    base_keys, base_errors = read_predictions(args.base)
    results = {}
    for path in args.candidates:
        keys, errors = read_predictions(path)
        result = compare(base_keys, base_errors, keys, errors)
        result["by_source_sensor_quarter"] = {}
        for label, ids in bins.items():
            base_mask, mask = np.isin(base_keys.image, ids), np.isin(keys.image, ids)
            result["by_source_sensor_quarter"][label] = compare(
                base_keys[base_mask], base_errors[base_mask], keys[mask], errors[mask])
        results[path.stem] = result
        print(json.dumps({"case": path.stem, **{k: v for k, v in result.items()
                                              if k != "by_source_sensor_quarter"}}), flush=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
