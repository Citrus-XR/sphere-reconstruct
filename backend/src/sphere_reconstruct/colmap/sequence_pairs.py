"""Fill retrieval gaps with explicit within-sensor sequence neighbors."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path


def missing_sequence_pairs(database_path: Path, records: list[dict], source_ids: set[str], *, overlap: int) -> tuple[list[tuple[str, str]], dict]:
    if overlap < 1:
        raise ValueError("sequence overlap must be positive")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["source_id"] in source_ids:
            groups[record["source_id"], record["sensor_id"]].append(record)
    missing_sources = source_ids - {source_id for source_id, _ in groups}
    if missing_sources:
        raise ValueError(f"ordered sources are absent from the image catalog: {sorted(missing_sources)}")
    with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        ids = {name: image_id for image_id, name in db.execute("SELECT image_id,name FROM images")}
        attempted = {pair_id for (pair_id,) in db.execute("SELECT pair_id FROM matches UNION SELECT pair_id FROM two_view_geometries")}
    pairs = []
    report = {"enabled": bool(source_ids), "overlap": overlap, "expected_pairs": 0, "missing_pairs": 0, "sources": {}}
    for (source_id, _sensor), images in sorted(groups.items()):
        ordered = sorted(images, key=lambda row: row["capture_index"])
        if len({row["capture_index"] for row in ordered}) != len(ordered):
            raise ValueError(f"multiple images for one ordered sensor capture: {source_id}")
        absent = [row["name"] for row in ordered if row["name"] not in ids]
        if absent:
            raise ValueError(f"ordered images missing from feature database: {absent[:5]}")
        summary = report["sources"].setdefault(source_id, {"expected_pairs": 0, "missing_pairs": 0})
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:index + 1 + overlap]:
                summary["expected_pairs"] += 1
                low, high = sorted((ids[first["name"]], ids[second["name"]]))
                if low * 2147483647 + high not in attempted:
                    pairs.append((first["name"], second["name"]))
                    summary["missing_pairs"] += 1
    report["expected_pairs"] = sum(row["expected_pairs"] for row in report["sources"].values())
    report["missing_pairs"] = len(pairs)
    return pairs, report
