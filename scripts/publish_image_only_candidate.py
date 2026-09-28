"""検証済み sparse model をバックアップ付きで公開し、通常の後段 stage を再実行する。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import sys
import traceback
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

from experiment_support import launch_detached, validate_primary_preserved, write_json


def validate_feature_identity(current, candidate):
    counts = {}
    with (sqlite3.connect(current.resolve().as_uri() + "?mode=ro", uri=True) as first,
          sqlite3.connect(candidate.resolve().as_uri() + "?mode=ro", uri=True) as second):
        for table, columns in [("images", "image_id,name,camera_id"), ("keypoints", "*"), ("descriptors", "*")]:
            query = f"SELECT {columns} FROM {table} ORDER BY image_id"
            count = 0
            for left, right in zip_longest(first.execute(query), second.execute(query)):
                if left != right:
                    image_id = left[0] if left is not None else right[0]
                    raise ValueError(f"current {table} differ from the candidate database at image {image_id}")
                count += 1
            counts[table] = count
    return counts


def restore_backup(project, affected, moves, output, *, backup_completed):
    errors = []
    blocked_outputs = set()
    if backup_completed:
        failed = output / "failed_publication"
        for stage in affected:
            for relative in (Path(stage.value), Path("manifests") / f"{stage.value}.json",
                             Path(".pipeline/stale") / f"{stage.value}.json"):
                source = project / relative
                if source.exists():
                    destination = failed / relative
                    try:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(destination)
                    except OSError as error:
                        errors.append(f"cannot preserve failed artifact {source}: {error}")
                        if relative == Path(stage.value):
                            blocked_outputs.add(stage)
    backed_up = dict(moves)

    def restore(source):
        destination = backed_up[source]
        try:
            source.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                raise FileExistsError(f"restore target still exists: {source}")
            destination.rename(source)
        except OSError as error:
            errors.append(f"cannot restore {destination} to {source}: {error}")
            return False
        return True

    for stage in reversed(affected):
        stage_output = project / stage.value
        if stage_output in backed_up and not restore(stage_output):
            blocked_outputs.add(stage)
        for relative in (Path("manifests") / f"{stage.value}.json",
                         Path(".pipeline/stale") / f"{stage.value}.json"):
            source = project / relative
            if stage in blocked_outputs:
                # A partial backup may leave metadata beside a directory that cannot be restored.
                if source not in backed_up and source.exists():
                    destination = output / "backup" / relative
                    try:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(destination)
                    except OSError as error:
                        errors.append(f"cannot withhold metadata {source}: {error}")
                continue
            if source in backed_up:
                restore(source)
    return errors


def execute(args, status):
    os.environ["SPHERE_CONFIG"] = str(args.config)
    sys.path.insert(0, str(args.backend_src))
    from sphere_reconstruct import stages  # noqa: F401
    from sphere_reconstruct.colmap import model as colmap_model
    from sphere_reconstruct.domain.artifacts import StageManifest
    from sphere_reconstruct.domain.pipeline_state import StageName, downstream_of
    from sphere_reconstruct.infrastructure.filesystem import sha256_bytes, sha256_file
    from sphere_reconstruct.pipeline.prepared_images import load_catalog

    project = args.project.resolve()
    affected = downstream_of(StageName.RECONSTRUCT)
    old_manifests = {
        stage: StageManifest.load(project / "manifests" / f"{stage.value}.json")
        for stage in affected if (project / "manifests" / f"{stage.value}.json").exists()
    }
    required = [StageName.RECONSTRUCT, StageName.ALIGN_RECONSTRUCTION, StageName.RESTORE_METRIC_SCALE,
                StageName.SCENE_ALIGNMENT, StageName.CLEANUP_SPARSE, StageName.EXPORT_DATASET]
    if any(stage not in old_manifests for stage in required if stage != StageName.CLEANUP_SPARSE):
        raise ValueError("publication requires an existing reconstruction and complete downstream stage parameters")
    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    if json.loads((project / "manifests/reconstruct.json").read_text(encoding="utf-8")) != baseline_manifest:
        raise ValueError("project reconstruction manifest changed since validation")
    references = {Path(row["path"]): row for row in baseline_manifest["outputs"]}
    for filename in ("cameras.bin", "images.bin", "frames.bin", "rigs.bin", "points3D.bin"):
        relative = Path("reconstruct/sparse/0") / filename
        if sha256_file(project / relative) != references[relative]["sha256"]:
            raise ValueError(f"project baseline changed since candidate experiment: {filename}")
    feature_identity = None
    for reference in baseline_manifest["inputs"]:
        if Path(reference["path"]) == Path("match_features/database.db"):
            # The candidate intentionally adds matches and updates calibration.
            # Verify the feature indices used by the model instead of SQLite layout.
            feature_identity = validate_feature_identity(project / reference["path"], args.candidate / "database.db")
            continue
        if sha256_file(project / reference["path"]) != reference["sha256"]:
            raise ValueError(f"reconstruction input changed: {reference['path']}")
    catalog = load_catalog(project)
    report_bytes = (args.candidate / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report["strategy"] != "image_sequence_completion_then_primary_anchored_registration":
        raise ValueError("expected image-only sequence registration report")
    if json.loads((args.candidate / "status.json").read_text())["state"] != "completed":
        raise ValueError("candidate experiment is not completed")
    if json.loads((args.source_run / "status.json").read_text())["state"] != "completed":
        raise ValueError("geometry and paired training pipeline is not completed")
    evidence_paths = {name: args.source_run / relative for name, relative in {
        "training_comparison": "training-comparison/report.json",
        "geometry_audit": "audit/report.json",
        "trajectory_audit": "final-trajectory.json",
    }.items()}
    evidence_bytes = {name: path.read_bytes() for name, path in evidence_paths.items()}
    training = json.loads(evidence_bytes["training_comparison"])
    if not training["official_metrics"]["candidate"] or not training["views"]:
        raise ValueError("paired training evidence is incomplete")
    geometry_audit = json.loads(evidence_bytes["geometry_audit"])
    if Path(geometry_audit["baseline_model"]).resolve() != project / "reconstruct/sparse/0":
        raise ValueError("audit did not compare this project baseline")
    candidate_model = args.candidate / "sparse/0"
    if Path(geometry_audit["candidate_model"]).resolve() != candidate_model.resolve():
        raise ValueError("audit did not compare this candidate")
    files = {path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted(candidate_model.glob("*.bin"))}
    model = colmap_model.read_model(candidate_model)
    primary = colmap_model.read_model(args.source_run / "primary/sparse")
    preservation = validate_primary_preserved(primary, model)
    primary_names = {row["name"] for row in catalog["images"] if row["source_role"] == "primary"}
    registered_names = {image.name for image in model.images.values()}
    if not primary_names <= registered_names or len(primary_names) != preservation["images"]:
        raise ValueError("candidate does not preserve every primary image")
    if len(model.images) != report["summary"]["num_images"] or len(model.points3D) != report["summary"]["num_points3D"]:
        raise ValueError("candidate model counts differ from its report")
    if not registered_names <= {row["name"] for row in catalog["images"]}:
        raise ValueError("candidate contains images outside this project")
    receipt = {"strategy": report["strategy"], "files": files, "final_images": len(model.images),
               "primary_images": len(primary_names), "phone_images": len(model.images) - len(primary_names),
               "final_points": len(model.points3D), "unregistered_images": report["unregistered_images"],
               "primary_preservation": preservation, "quality": "pose_comparison_completed_with_remaining_geometry_limits",
               "feature_identity": feature_identity,
               "evidence_sha256": {name:sha256_bytes(data) for name,data in evidence_bytes.items()},
               "training_validation_scope": {
                   "kind": "common_phone_cohort_and_primary_seed_points",
                   "primary_images": training["comparison"]["primary_images"],
                   "phone_images": training["comparison"]["common_phone_images"],
                   "seed_points": training["comparison"]["primary_seed_points"],
                   "covers_full_export": False},
               "report_sha256": sha256_bytes(report_bytes)}
    del model, primary
    write_json(args.output / "validated_candidate.json", receipt)
    if args.validate_only:
        return

    publish_validated_candidate(args, status, receipt, report_bytes, evidence_bytes, old_manifests, catalog)


def publish_validated_candidate(args, status, receipt, report_bytes, evidence_bytes, old_manifests, catalog,
                                *, candidate_database=None):
    from sphere_reconstruct.colmap import model as colmap_model
    from sphere_reconstruct.colmap.input_workspace import InputSpec
    from sphere_reconstruct.domain.artifacts import FileRef, StageManifest
    from sphere_reconstruct.domain.pipeline_state import StageName, downstream_of
    from sphere_reconstruct.infrastructure.filesystem import sha256_file
    from sphere_reconstruct.pipeline.engine import Engine
    from sphere_reconstruct.pipeline.invalidation import derive_pipeline_state
    from sphere_reconstruct.pipeline.manifest import register
    from sphere_reconstruct.pipeline.stage import new_manifest
    from sphere_reconstruct.stages import similarity_transform
    from sphere_reconstruct.stages.reconstruct import (
        Reconstruct,
        _file_ref,
        _select_largest_model,
    )

    project = args.project.resolve()
    affected = downstream_of(StageName.RECONSTRUCT)
    required = [StageName.RECONSTRUCT, StageName.ALIGN_RECONSTRUCTION, StageName.RESTORE_METRIC_SCALE,
                StageName.SCENE_ALIGNMENT, StageName.CLEANUP_SPARSE, StageName.EXPORT_DATASET]
    candidate_model = args.candidate / "sparse/0"
    candidate_database = candidate_database or args.candidate / "database.db"
    files = receipt["files"]
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    database = project.parents[1] / "state.db"
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("BEGIN IMMEDIATE")
    try:
        if connection.execute("SELECT id FROM project WHERE id=?", (project.name,)).fetchone() is None:
            raise ValueError("project is absent from workspace state database")
        if connection.execute("SELECT id FROM job WHERE project_id=? AND status IN ('queued','running')", (project.name,)).fetchone():
            raise RuntimeError("project has an active job")
        connection.execute("INSERT INTO job(id,project_id,kind,stage,status,created_at,started_at,pid) "
                           "VALUES(?,?,'rerun_stage','reconstruct','running',?,?,?)",
                           (job_id, project.name, now, now, os.getpid()))
        connection.commit()
    except BaseException:
        connection.rollback()
        connection.close()
        raise
    backup = args.output / "backup"
    moves = []
    backup_completed = False
    engine = None
    try:
        status.update(job_id=job_id, phase="backup")
        write_json(args.output / "status.json", status)
        backup.mkdir()
        for stage in affected:
            for relative in (Path(stage.value), Path("manifests") / f"{stage.value}.json",
                             Path(".pipeline/stale") / f"{stage.value}.json"):
                source, destination = project / relative, backup / relative
                if source.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(destination)
                    moves.append((source, destination))
                    write_json(args.output / "backup_moves.json", [{"project_path": str(a.relative_to(project)),
                               "backup_path": str(b.relative_to(args.output))} for a, b in moves])
        backup_completed = True
        import_dir = project / ".pipeline/imports" / job_id
        import_dir.mkdir(parents=True)
        write_json(import_dir / "candidate.json", receipt)
        (import_dir / "quality_report.json").write_bytes(report_bytes)
        for name, data in evidence_bytes.items():
            (import_dir / (name + ".json")).write_bytes(data)

        @register
        class CandidateReconstruction(Reconstruct):
            impl_version = "verified-sparse-import-2"

            def collect_inputs(self, ctx):
                return [*super().collect_inputs(ctx), *[
                    FileRef(path=str(path.relative_to(project)), size=path.stat().st_size, sha256=sha256_file(path))
                    for path in sorted(import_dir.glob("*.json"))]]

            def execute(self, ctx):
                manifest = new_manifest(self.name, self.impl_version)
                manifest.inputs, manifest.params = ctx.inputs_for(self), ctx.params
                target = ctx.stage_out_dir / "sparse/0"
                shutil.copytree(candidate_model, target)
                for filename, expected in files.items():
                    if sha256_file(target / filename) != expected["sha256"]:
                        raise RuntimeError(f"candidate changed during publication: {filename}")
                spec = InputSpec.read(project / "extract_features/input_spec.json")
                spec = replace(spec, sources=catalog["sources"])
                _, summary = _select_largest_model(ctx.stage_out_dir / "sparse", spec,
                                                  max_step_ratio=ctx.params["max_adjacent_step_ratio"])
                summary.update(candidate_import=receipt, mapper="incremental", requested_mapper=ctx.params["mapper"],
                               registered_ratio=1.0, registered_total_ratio=summary["num_images"] / spec.image_count,
                               input_images=spec.image_count, view_graph_calibration=False)
                preview = similarity_transform.write_preview(ctx, colmap_model.read_model(target),
                          metadata_key="reconstruction", metadata=summary, max_points=ctx.params["max_preview_points"])
                summary["preview_points"] = preview.num_points_written
                write_json(ctx.stage_out_dir / "model_summary.json", summary)
                (ctx.stage_out_dir / "quality_report.json").write_bytes(report_bytes)
                shutil.copy2(candidate_database, ctx.stage_out_dir / "database.db")
                manifest.outputs = [_file_ref(path, ctx) for path in sorted(ctx.stage_out_dir.rglob("*")) if path.is_file()]
                manifest.extra = summary
                return manifest

        engine = Engine(database, project.parents[1], project.name, job_id)
        for stage in required:
            status.update(phase=stage.value)
            write_json(args.output / "status.json", status)
            print(f"STAGE {stage.value}", flush=True)
            params = dict(old_manifests[stage].params) if stage in old_manifests else {}
            if stage == StageName.CLEANUP_SPARSE:
                params.update(enabled=True, remove_trajectory_outliers=False)
            engine.run_stage(stage, params)
        cleanup = StageManifest.load(project / "manifests/cleanup_sparse.json")
        exported = StageManifest.load(project / "manifests/export_dataset.json")
        if (cleanup.extra["removed_images"] != 0
                or exported.extra["model_source"] != "cleanup_sparse"
                or exported.extra["num_points3D"] != cleanup.extra["output_points"]):
            raise RuntimeError("publication must export the cleaned points without deleting candidate cameras")
        if exported.extra["num_images"] != receipt["final_images"] or not exported.extra["validation"]["training_ready"]:
            raise RuntimeError("published export did not preserve the candidate or is not training-ready")
        write_json(args.output / "published.json", {"job_id": job_id, "candidate": receipt,
                   "export_validation": exported.extra["validation"], "sparse_cleanup": cleanup.extra,
                   "export_points": exported.extra["num_points3D"], "backup": "backup"})
        connection.execute("UPDATE job SET status='succeeded',finished_at=? WHERE id=?", (datetime.now(UTC).isoformat(), job_id))
    except BaseException as error:
        errors = restore_backup(project, affected, moves, args.output, backup_completed=backup_completed)
        diagnostic = repr(error) + (f"; rollback errors: {errors}" if errors else "")
        connection.execute("UPDATE job SET status='failed',finished_at=?,error_text=? WHERE id=?",
                           (datetime.now(UTC).isoformat(), diagnostic, job_id))
        connection.execute("UPDATE project SET state=?,updated_at=? WHERE id=?",
                           (derive_pipeline_state(project).value, datetime.now(UTC).isoformat(), project.name))
        if errors:
            raise RuntimeError(diagnostic) from error
        raise
    finally:
        if engine is not None:
            engine.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "candidate", "source-run", "baseline-manifest", "config", "backend-src", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.detach:
        launch_detached(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "phase": "validate", "started_at": datetime.now(UTC).isoformat()}
    with (args.output / "worker.log").open("x", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        write_json(args.output / "status.json", status)
        try:
            execute(args, status)
        except BaseException as error:
            status.update(state="failed", error=repr(error), exit_code=1)
            traceback.print_exc()
            raise
        else:
            status.update(state="validated" if args.validate_only else "completed", exit_code=0)
        finally:
            status["finished_at"] = datetime.now(UTC).isoformat()
            write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
