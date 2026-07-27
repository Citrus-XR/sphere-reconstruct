"""抽出済み特徴から image pair と幾何検証済み matching を作る独立ステージ."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from ..colmap import runner as colmap_runner
from ..colmap.input_workspace import InputSpec
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings
from .colmap_progress import counted_progress


@register
class MatchFeatures(Stage):
    name = StageName.MATCH_FEATURES
    impl_version = "2.0"

    def normalize_params(self, raw: dict) -> dict:
        feature_type = str(raw.get("feature_type", "SIFT")).upper()
        matcher = str(raw.get("matcher_type", "bruteforce")).lower()
        if matcher not in {"bruteforce", "lightglue"}:
            raise ValueError(f"unsupported matcher type: {matcher}")
        pairing = str(raw.get("pairing", "auto")).lower()
        if pairing not in {"auto", "sequential", "exhaustive", "vocab_tree"}:
            raise ValueError(f"unsupported pairing strategy: {pairing}")
        return {
            "feature_type": feature_type,
            "matcher_type": matcher,
            "pairing": pairing,
            "overlap": int(raw.get("overlap", 4)),
            "loop_closure": bool(raw.get("loop_closure", False)),
            "use_gpu": bool(raw.get("use_gpu", True)),
            "max_num_matches": int(raw.get("max_num_matches", 16384)),
            "guided_matching": bool(raw.get("guided_matching", False)),
            "min_num_inliers": int(raw.get("min_num_inliers", 15)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "extract_features.json",
            ctx.project_dir / "extract_features" / "database.db",
            ctx.project_dir / "extract_features" / "input_spec.json",
        ]
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.exists()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params
        source_database = ctx.project_dir / "extract_features" / "database.db"
        if not source_database.exists():
            raise RuntimeError("extract_features must run before matching")
        database_path = ctx.stage_out_dir / "database.db"
        shutil.copy2(source_database, database_path)
        logs_dir = ctx.stage_out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        settings = get_settings()
        spec = InputSpec.read(ctx.project_dir / "extract_features" / "input_spec.json")
        colmap_bin = colmap_runner.resolve_colmap_bin(settings.binaries.colmap or None)
        matching_type = _matching_type(ctx.params["feature_type"], ctx.params["matcher_type"])
        extra_args = [
            "--FeatureMatching.max_num_matches",
            str(ctx.params["max_num_matches"]),
            "--FeatureMatching.guided_matching",
            "1" if ctx.params["guided_matching"] else "0",
            "--TwoViewGeometry.min_num_inliers",
            str(ctx.params["min_num_inliers"]),
        ]
        if matching_type == "ALIKED_LIGHTGLUE" and settings.aliked.matcher_path:
            extra_args += ["--AlikedMatching.lightglue_model_path", settings.aliked.matcher_path]

        ctx.progress.info(
            f"matching: {matching_type}, pairing={ctx.params['pairing']}",
            progress=0.05,
            key="log.matching_start",
            args={"type": matching_type, "pairing": ctx.params["pairing"]},
        )
        common = {
            "database_path": database_path,
            "use_gpu": ctx.params["use_gpu"],
            "matching_type": matching_type,
            "extra_args": extra_args,
            "log_path": logs_dir / "matcher.log",
            "on_line": counted_progress(
                ctx,
                "matching",
                r"Processing image \[(\d+)/(\d+)\]",
                low=0.05,
                high=0.95,
            ),
        }
        pairing = _resolve_pairing(ctx.params["pairing"], spec, settings, ctx.params["feature_type"])
        if pairing == "exhaustive":
            colmap_runner.exhaustive_matcher(colmap_bin, **common)
        elif pairing == "vocab_tree":
            if not ctx.params["feature_type"].startswith("SIFT"):
                raise ValueError("the configured vocabulary tree is only valid for SIFT")
            if not settings.binaries.vocab_tree:
                raise RuntimeError("vocab_tree pairing requires binaries.vocab_tree")
            colmap_runner.vocab_tree_matcher(
                colmap_bin,
                vocab_tree_path=Path(settings.binaries.vocab_tree),
                **common,
            )
        else:
            loop = ctx.params["loop_closure"]
            if loop and not settings.binaries.vocab_tree:
                raise RuntimeError("loop closure requires binaries.vocab_tree")
            colmap_runner.sequential_matcher(
                colmap_bin,
                overlap=ctx.params["overlap"],
                loop_detection=loop,
                vocab_tree_path=Path(settings.binaries.vocab_tree) if loop else None,
                **common,
            )

        summary = _matching_summary(database_path, spec)
        summary.update(
            {
                "matching_type": matching_type,
                "pairing": pairing,
                "requested_pairing": ctx.params["pairing"],
                "loop_closure": ctx.params["loop_closure"] if pairing == "sequential" else False,
                "gpu_enabled": ctx.params["use_gpu"],
            }
        )
        summary_path = ctx.stage_out_dir / "matching_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.outputs = [_file_ref(database_path, ctx), _file_ref(summary_path, ctx)]
        manifest.extra = summary
        ctx.progress.info(
            f"matching done: {summary['verified_pairs']} verified pairs",
            progress=1.0,
            key="log.matching_done",
            args={"pairs": summary["verified_pairs"]},
        )
        return manifest


def _matching_type(feature_type: str, matcher_type: str) -> str:
    family = "ALIKED" if feature_type.startswith("ALIKED") else "SIFT"
    suffix = "LIGHTGLUE" if matcher_type == "lightglue" else "BRUTEFORCE"
    return f"{family}_{suffix}"


def _resolve_pairing(requested: str, spec: InputSpec, settings, feature_type: str) -> str:
    if requested != "auto":
        return requested
    if spec.source_count <= 1:
        return "sequential"
    if spec.image_count <= 500:
        return "exhaustive"
    if feature_type.startswith("SIFT") and settings.binaries.vocab_tree:
        return "vocab_tree"
    raise RuntimeError(
        "500 枚を超える mixed-source dataset には SIFT vocab tree を設定するか、"
        "pairing strategy を明示してください"
    )


def _matching_summary(database_path: Path, spec: InputSpec) -> dict:
    with sqlite3.connect(database_path) as connection:
        raw_pairs = connection.execute("SELECT COUNT(*) FROM matches WHERE rows > 0").fetchone()[0]
        row = connection.execute(
            "SELECT COUNT(*), MIN(rows), AVG(rows), MAX(rows), SUM(rows) "
            "FROM two_view_geometries WHERE rows > 0"
        ).fetchone()
        image_names = dict(connection.execute("SELECT image_id, name FROM images").fetchall())
        verified_pair_ids = [
            int(value[0])
            for value in connection.execute(
                "SELECT pair_id FROM two_view_geometries WHERE rows > 0"
            ).fetchall()
        ]
    source_by_name = {image["name"]: image["source_id"] for image in spec.images}
    source_edges: dict[str, int] = {}
    cross_source_pairs = 0
    for pair_id in verified_pair_ids:
        first_id, second_id = _pair_ids(pair_id)
        first_source = source_by_name.get(image_names.get(first_id, ""))
        second_source = source_by_name.get(image_names.get(second_id, ""))
        if first_source is None or second_source is None or first_source == second_source:
            continue
        cross_source_pairs += 1
        key = "|".join(sorted((first_source, second_source)))
        source_edges[key] = source_edges.get(key, 0) + 1
    return {
        "raw_pairs": int(raw_pairs),
        "verified_pairs": int(row[0]),
        "minimum_inliers": int(row[1] or 0),
        "average_inliers": float(row[2] or 0.0),
        "maximum_inliers": int(row[3] or 0),
        "total_inliers": int(row[4] or 0),
        "cross_source_verified_pairs": cross_source_pairs,
        "source_pair_counts": source_edges,
    }


def _pair_ids(pair_id: int) -> tuple[int, int]:
    maximum_image_id = 2_147_483_647
    second = pair_id % maximum_image_id
    first = (pair_id - second) // maximum_image_id
    return first, second


def _file_ref(path: Path, ctx: StageContext) -> FileRef:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    relative = Path(final_name) / path.relative_to(ctx.stage_out_dir)
    return FileRef(
        path=str(relative),
        size=path.stat().st_size,
        sha256=sha256_file(path),
        mime="application/json" if path.suffix == ".json" else "application/octet-stream",
    )
