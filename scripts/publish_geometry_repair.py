"""局所 pose / track 修正を検証し、通常の cleanup と export を含めて公開する。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from experiment_support import launch_detached, write_json
from publish_image_only_candidate import publish_validated_candidate


def execute(args, status):
    os.environ["SPHERE_CONFIG"] = str(args.config)
    sys.path.insert(0, str(args.backend_src))
    from sphere_reconstruct import stages  # noqa: F401
    from sphere_reconstruct.colmap import model as colmap_model
    from sphere_reconstruct.colmap.point_stability import validate_tracks
    from sphere_reconstruct.domain.artifacts import StageManifest
    from sphere_reconstruct.domain.pipeline_state import StageName, downstream_of
    from sphere_reconstruct.infrastructure.filesystem import sha256_bytes, sha256_file
    from sphere_reconstruct.pipeline.prepared_images import load_catalog

    project = args.project.resolve()
    if json.loads((args.candidate / "status.json").read_text())["state"] != "completed":
        raise ValueError("geometry repair is not completed")
    report_bytes = (args.candidate / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report["strategy"] != "verified_local_pose_and_track_refit":
        raise ValueError("expected a verified local geometry repair")
    if not report["other_poses_unchanged"] or not report["intrinsics_unchanged"]:
        raise ValueError("repair changed unrelated camera geometry")
    hashes = {Path(name).resolve(): digest for name, digest in report["source_hashes"].items()}
    binaries = ("cameras.bin", "images.bin", "frames.bin", "rigs.bin", "points3D.bin")
    required_sources = {project / stage / "sparse/0" / name
                        for stage in ("reconstruct", "cleanup_sparse") for name in binaries}
    required_sources.add(project / "extract_features/input_spec.json")
    if not required_sources <= hashes.keys():
        raise ValueError("repair provenance does not cover this project and its cleaned model")
    for path, digest in hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"repair input changed since validation: {path}")
    candidate_model = args.candidate / "sparse/0"
    if set(report["output_hashes"]) != set(binaries):
        raise ValueError("repair output must include cameras, frames, rigs, images, and points")
    files = {}
    for name, digest in report["output_hashes"].items():
        path = candidate_model / name
        if sha256_file(path) != digest:
            raise ValueError(f"repair output changed: {name}")
        files[name] = {"size": path.stat().st_size, "sha256": digest}
    catalog = load_catalog(project)
    model = colmap_model.read_model(candidate_model)
    validate_tracks(model)
    registered = {image.name for image in model.images.values()}
    primary = {row["name"] for row in catalog["images"] if row["source_role"] == "primary"}
    expected = {row["name"] for row in catalog["images"]}
    if not primary <= registered or not registered <= expected:
        raise ValueError("repair image set does not match the project catalog")
    if len(model.images) != report["num_images"] or len(model.points3D) != report["num_points3D"]:
        raise ValueError("repair output counts differ from the report")
    if len(model.images) != report["preserved_images"]:
        raise ValueError("repair changed the camera set")
    old_manifests = {stage: StageManifest.load(project / "manifests" / f"{stage.value}.json")
                     for stage in downstream_of(StageName.RECONSTRUCT)
                     if (project / "manifests" / f"{stage.value}.json").exists()}
    for stage in (StageName.RECONSTRUCT, StageName.ALIGN_RECONSTRUCTION, StageName.RESTORE_METRIC_SCALE,
                  StageName.SCENE_ALIGNMENT, StageName.CLEANUP_SPARSE, StageName.EXPORT_DATASET):
        if stage not in old_manifests:
            raise ValueError(f"missing downstream stage parameters: {stage.value}")
    for reference in old_manifests[StageName.RECONSTRUCT].inputs:
        if sha256_file(project / reference.path) != reference.sha256:
            raise ValueError(f"reconstruction input changed: {reference.path}")
    receipt = {"strategy": report["strategy"], "files": files, "final_images": len(model.images),
               "primary_images": len(primary), "phone_images": len(registered - primary),
               "final_points": len(model.points3D), "changed_images": report["changed_images"],
               "restored_points": len(report["restored_point_ids"]), "report_sha256": sha256_bytes(report_bytes),
               "quality": "local_image_geometry_validated", "training_validation_scope": "not_evaluated"}
    del model
    write_json(args.output / "validated_candidate.json", receipt)
    if args.validate_only:
        return
    # The project reconstruction directory is moved into the rollback backup during publication.
    database = args.output / "database.db"
    shutil.copy2(project / "reconstruct/database.db", database)
    publish_validated_candidate(args, status, receipt, report_bytes, {}, old_manifests, catalog,
                                candidate_database=database)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "candidate", "config", "backend-src", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
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
            status.update(state="validated" if args.validate_only else "completed")
        except BaseException as error:
            status.update(state="failed", error=repr(error))
            traceback.print_exc()
            raise
        finally:
            status["finished_at"] = datetime.now(UTC).isoformat()
            write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
