"""特徴 database の各 ray が consumer 有効視野に収まるかを読み取り専用で検証する。"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from sphere_reconstruct.imaging import fisheye_camera
from sphere_reconstruct.pipeline import prepared_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    catalog = prepared_images.load_catalog(args.project)
    by_name = {record["name"]: record for record in catalog["images"]}
    summary = {}
    with sqlite3.connect(f"{(args.project / 'extract_features' / 'database.db').as_uri()}?mode=ro", uri=True) as db:
        for name, rows, cols, blob in db.execute(
            "SELECT images.name, keypoints.rows, keypoints.cols, keypoints.data "
            "FROM keypoints JOIN images USING(image_id)"
        ):
            record = by_name[name]
            region = record["valid_region"]
            if region["kind"] != "fisheye":
                continue
            key = f"{record['source_id']}:{record['sensor_id']}"
            report = summary.setdefault(key, {"images": 0, "keypoints": 0, "outside_hemisphere": 0,
                                              "outside_valid_angle": 0, "maximum_angle_deg": 0.0,
                                              "valid_angle_deg": math.degrees(region["max_theta_rad"])})
            pixels = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2]
            theta = fisheye_camera.pixels_to_theta(region["camera_model"], region["params"], pixels)
            report["images"] += 1
            report["keypoints"] += rows
            report["outside_hemisphere"] += int(np.count_nonzero(theta >= math.pi / 2))
            report["outside_valid_angle"] += int(np.count_nonzero(theta >= region["max_theta_rad"]))
            if len(theta):
                report["maximum_angle_deg"] = max(report["maximum_angle_deg"], math.degrees(float(theta.max())))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
