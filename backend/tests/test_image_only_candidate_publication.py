"""候補の公開失敗時に既存成果物と job の状態を復元する。"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from sphere_reconstruct.colmap import model as colmap_model
from sphere_reconstruct.domain.artifacts import StageManifest
from sphere_reconstruct.domain.pipeline_state import StageName, downstream_of
from sphere_reconstruct.infrastructure.database import _SCHEMA
from sphere_reconstruct.infrastructure.filesystem import sha256_bytes
from sphere_reconstruct.pipeline import manifest as stage_registry


@pytest.fixture
def publication(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository / "scripts"))
    spec = importlib.util.spec_from_file_location("candidate_publication", repository / "scripts/publish_image_only_candidate.py")
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    monkeypatch.setattr(stage_registry, "_REGISTRY", dict(stage_registry._REGISTRY))
    monkeypatch.setenv("SPHERE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(sys, "path", list(sys.path))

    workspace = tmp_path / "workspace"
    project = workspace / "projects" / "project"
    candidate, source, output = (tmp_path / name for name in ("candidate", "source", "publication"))
    for directory in (project, candidate / "sparse/0", source / "primary/sparse", output,
                      source / "audit", source / "training-comparison"):
        directory.mkdir(parents=True)
    affected = downstream_of(StageName.RECONSTRUCT)
    for stage in affected:
        directory = project / stage.value
        directory.mkdir()
        (directory / "original.txt").write_text(f"original {stage.value}")
        StageManifest(stage=stage.value, impl_version="original", started_at=datetime.now(UTC)).dump(
            project / "manifests" / f"{stage.value}.json"
        )
        stale = project / ".pipeline/stale" / f"{stage.value}.json"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(json.dumps({"original": stage.value}))
    (project / "reconstruct/sparse/0").mkdir(parents=True)
    for filename in ("cameras.bin", "images.bin", "frames.bin", "rigs.bin", "points3D.bin"):
        for directory in (project / "reconstruct/sparse/0", source / "primary/sparse", candidate / "sparse/0"):
            (directory / filename).write_bytes(filename.encode())
    (project / "reconstruct/database.db").write_bytes(b"original database")
    (candidate / "database.db").write_bytes(b"completed matching database")
    baseline_path = project / "manifests/reconstruct.json"
    baseline_manifest = json.loads(baseline_path.read_text())
    baseline_manifest['outputs'] = [dict(path=str(path.relative_to(project)), size=path.stat().st_size,
                                        sha256=sha256_bytes(path.read_bytes()))
                                    for path in (project / 'reconstruct/sparse/0').iterdir()]
    baseline_path.write_text(json.dumps(baseline_manifest))
    (source / 'baseline_manifest.json').write_bytes(baseline_path.read_bytes())
    (source / 'status.json').write_text(json.dumps({'state':'completed'}))
    (source / 'training-comparison/report.json').write_text(json.dumps({'official_metrics':{'candidate':[{}]},'views':[{}], 'comparison':{'primary_images':1,'common_phone_images':1,'primary_seed_points':0}}))
    (source / 'audit/report.json').write_text(json.dumps({'baseline_model':str(project / 'reconstruct/sparse/0'),
                                                         'candidate_model':str(candidate / 'sparse/0')}))
    (source / 'final-trajectory.json').write_text('{}')
    monkeypatch.setattr(publisher, 'validate_primary_preserved', lambda *_args: {'images':1})
    (project / "rectify_fisheye").mkdir()
    catalog = {"images": [{"name": "primary.jpg", "source_role": "primary"},
                           {"name": "phone.jpg", "source_role": "detail"}], "sources": []}
    for path in (project / "rectify_fisheye/image_catalog.json", source / "image_catalog.json"):
        path.write_text(json.dumps(catalog))
    (project / "source-video.mp4").write_bytes(b"source footage")
    report = {"strategy": "image_sequence_completion_then_primary_anchored_registration",
              "primary_images": 1, "phone_images": 1, "final_images": 2, "final_points": 0,
              "missing_from_original_baseline": [], 'unregistered_images':[], 'summary':{'num_images':2, 'num_points3D':0}}
    (candidate / "report.json").write_text(json.dumps(report))
    (candidate / "status.json").write_text(json.dumps({"state": "completed"}))
    monkeypatch.setattr(colmap_model, "read_model", lambda _path: SimpleNamespace(
        images={1: SimpleNamespace(name="primary.jpg"), 2: SimpleNamespace(name="phone.jpg")}, points3D={}
    ))
    database = workspace / "state.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(_SCHEMA)
        connection.execute("INSERT INTO project(id,name,created_at,updated_at,state) "
                           "VALUES('project','project','2026-09-18','2026-09-18','exported')")
    originals = {path.relative_to(project): path.read_bytes() for path in project.rglob("*") if path.is_file()}
    args = SimpleNamespace(project=project, candidate=candidate, source_run=source, output=output,
                           baseline_manifest=source / 'baseline_manifest.json',
                           config=tmp_path / "config.toml", backend_src=repository / "backend/src", validate_only=False)
    return SimpleNamespace(publisher=publisher, args=args, originals=originals, database=database)


def _assert_originals_restored(publication):
    for relative, contents in publication.originals.items():
        assert (publication.args.project / relative).read_bytes() == contents, str(relative)


def _job(publication):
    with sqlite3.connect(publication.database) as connection:
        return connection.execute("SELECT status,finished_at,error_text FROM job").fetchone()


def test_partial_backup_failure_restores_moved_and_untouched_artifacts(publication, monkeypatch):
    original_rename = Path.rename
    failure = PermissionError("alignment output is locked")

    def fail_second_stage(path, target):
        if path == publication.args.project / "align_reconstruction":
            raise failure
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second_stage)
    with pytest.raises(PermissionError) as raised:
        publication.publisher.execute(publication.args, {})
    assert raised.value is failure
    _assert_originals_restored(publication)
    assert _job(publication)[0] == "failed"
    assert not (publication.args.output / "failed_publication").exists()


def test_backup_journal_failure_restores_already_renamed_directory(publication, monkeypatch):
    write_json = publication.publisher.write_json
    failure = OSError("backup journal volume is full")

    def fail_journal(path, value):
        if path == publication.args.output / "backup_moves.json":
            assert not (publication.args.project / "reconstruct").exists()
            raise failure
        write_json(path, value)

    monkeypatch.setattr(publication.publisher, "write_json", fail_journal)
    with pytest.raises(OSError) as raised:
        publication.publisher.execute(publication.args, {})
    assert raised.value is failure
    _assert_originals_restored(publication)
    assert _job(publication)[0] == "failed"


def _failing_engine(publication, monkeypatch):
    failure = RuntimeError("downstream export failed")

    class FailingEngine:
        def __init__(self, *_args):
            pass

        def run_stage(self, stage, _params):
            directory = publication.args.project / stage.value
            directory.mkdir()
            (directory / "new.txt").write_text("new candidate output")
            manifest = publication.args.project / "manifests" / f"{stage.value}.json"
            manifest.write_text("failed new manifest")
            raise failure

        def close(self):
            pass

    monkeypatch.setattr("sphere_reconstruct.pipeline.engine.Engine", FailingEngine)
    return failure


def test_downstream_failure_preserves_failed_outputs_and_restores_originals(publication, monkeypatch):
    failure = _failing_engine(publication, monkeypatch)
    with pytest.raises(RuntimeError) as raised:
        publication.publisher.execute(publication.args, {})
    assert raised.value is failure
    _assert_originals_restored(publication)
    assert (publication.args.output / "failed_publication/reconstruct/new.txt").read_text() == "new candidate output"
    status, finished, diagnostic = _job(publication)
    assert status == "failed" and finished is not None
    assert str(failure) in diagnostic


def test_initial_status_write_failure_finalizes_created_job(publication, monkeypatch):
    write_json = publication.publisher.write_json
    failure = OSError("status volume is full")

    def fail_status(path, value):
        if path == publication.args.output / "status.json":
            raise failure
        write_json(path, value)

    monkeypatch.setattr(publication.publisher, "write_json", fail_status)
    with pytest.raises(OSError) as raised:
        publication.publisher.execute(publication.args, {})
    assert raised.value is failure
    _assert_originals_restored(publication)
    status, finished, diagnostic = _job(publication)
    assert status == "failed" and finished is not None
    assert str(failure) in diagnostic


def test_rollback_lock_preserves_backup_and_restores_unrelated_stages(publication, monkeypatch):
    failure = _failing_engine(publication, monkeypatch)
    original_rename = Path.rename

    def fail_quarantine(path, target):
        if path == publication.args.project / "reconstruct" and "failed_publication" in Path(target).parts:
            raise PermissionError("failed candidate output remains locked")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_quarantine)
    with pytest.raises(RuntimeError) as raised:
        publication.publisher.execute(publication.args, {})
    assert raised.value.__cause__ is failure
    assert (publication.args.output / "backup/reconstruct/original.txt").read_text() == "original reconstruct"
    for relative, content in publication.originals.items():
        if relative in (Path("manifests/reconstruct.json"), Path(".pipeline/stale/reconstruct.json")):
            assert not (publication.args.project / relative).exists()
            assert (publication.args.output / "backup" / relative).read_bytes() == content
        elif relative.parts[0] != "reconstruct":
            assert (publication.args.project / relative).read_bytes() == content
    status, finished, diagnostic = _job(publication)
    assert status == "failed" and finished is not None
    assert str(failure) in diagnostic
    assert "locked" in diagnostic


def test_restore_output_before_publishing_its_original_manifest(publication, monkeypatch):
    _failing_engine(publication, monkeypatch)
    original_rename = Path.rename

    def check_restore_order(path, target):
        if path == publication.args.output / "backup/manifests/reconstruct.json":
            assert (publication.args.project / "reconstruct/original.txt").read_text() == "original reconstruct"
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", check_restore_order)
    with pytest.raises(RuntimeError, match="downstream export failed"):
        publication.publisher.execute(publication.args, {})
    _assert_originals_restored(publication)


def test_partial_backup_restore_lock_withholds_unmoved_metadata(publication, monkeypatch):
    write_json = publication.publisher.write_json
    original_rename = Path.rename

    def fail_journal(path, value):
        if path == publication.args.output / "backup_moves.json":
            raise OSError("journal write failed")
        write_json(path, value)

    def fail_restore(path, target):
        if path == publication.args.output / "backup/reconstruct":
            raise PermissionError("old reconstruction cannot be restored")
        return original_rename(path, target)

    monkeypatch.setattr(publication.publisher, "write_json", fail_journal)
    monkeypatch.setattr(Path, "rename", fail_restore)
    with pytest.raises(RuntimeError, match="rollback errors"):
        publication.publisher.execute(publication.args, {})
    for relative in (Path("manifests/reconstruct.json"), Path(".pipeline/stale/reconstruct.json")):
        assert not (publication.args.project / relative).exists()
        assert (publication.args.output / "backup" / relative).read_bytes() == publication.originals[relative]
    assert _job(publication)[0] == "failed"


@pytest.mark.parametrize("cleanup_state", ["present", "missing", "disabled"])
def test_publication_pins_report_bytes_before_validation_and_copy(publication, monkeypatch, cleanup_state):
    cleanup_path = publication.args.project / "manifests/cleanup_sparse.json"
    if cleanup_state == "missing":
        cleanup_path.unlink()
    elif cleanup_state == "disabled":
        cleanup_manifest = StageManifest.load(cleanup_path)
        cleanup_manifest.params = {"enabled": False, "remove_trajectory_outliers": True}
        cleanup_manifest.dump(cleanup_path)
    report_path = publication.args.candidate / "report.json"
    expected_bytes = report_path.read_bytes()
    original_read_model = colmap_model.read_model

    def replace_report_during_model_validation(path):
        report_path.write_text('{"replaced": true}')
        return original_read_model(path)

    monkeypatch.setattr(colmap_model, "read_model", replace_report_during_model_validation)

    @dataclass
    class MinimalInputSpec:
        sources: list
        image_count: int = 2

    monkeypatch.setattr("sphere_reconstruct.colmap.input_workspace.InputSpec.read",
                        lambda _path: MinimalInputSpec([]))
    monkeypatch.setattr("sphere_reconstruct.stages.reconstruct._select_largest_model",
                        lambda *_args, **_kwargs: (None, {"num_images": 2}))
    monkeypatch.setattr("sphere_reconstruct.stages.similarity_transform.write_preview",
                        lambda *_args, **_kwargs: SimpleNamespace(num_points_written=0))

    stage_calls = []

    class ImportingEngine:
        def __init__(self, *_args):
            pass

        def run_stage(self, stage, _params):
            stage_calls.append((stage, _params))
            directory = publication.args.project / stage.value
            directory.mkdir()
            if stage == StageName.RECONSTRUCT:
                ctx = SimpleNamespace(project_dir=publication.args.project, stage_out_dir=directory,
                                      inputs_for=lambda _stage: [], params={"mapper": "global",
                                      "max_preview_points": 10, "max_adjacent_step_ratio": 10})
                manifest = stage_registry.get(stage)().execute(ctx)
                manifest.dump(publication.args.project / "manifests/reconstruct.json")
            elif stage == StageName.CLEANUP_SPARSE:
                StageManifest(stage=stage.value, impl_version="stub-cleanup", started_at=datetime.now(UTC),
                              extra={"removed_images":0,"output_points":0}).dump(
                    publication.args.project / "manifests/cleanup_sparse.json")
            elif stage == StageName.EXPORT_DATASET:
                StageManifest(stage=stage.value, impl_version="stub-export", started_at=datetime.now(UTC),
                              extra={"num_images": 2, "num_points3D":0, "model_source":"cleanup_sparse", "validation": {"training_ready": True}}).dump(
                    publication.args.project / "manifests/export_dataset.json")

        def close(self):
            pass

    monkeypatch.setattr("sphere_reconstruct.pipeline.engine.Engine", ImportingEngine)
    status = {}
    publication.publisher.execute(publication.args, status)
    cleanup_calls = [params for stage, params in stage_calls if stage == StageName.CLEANUP_SPARSE]
    assert cleanup_calls == [{"enabled": True, "remove_trajectory_outliers": False}]
    assert [stage for stage, _ in stage_calls].index(StageName.CLEANUP_SPARSE) < [stage for stage, _ in stage_calls].index(StageName.EXPORT_DATASET)
    receipt = json.loads((publication.args.output / "validated_candidate.json").read_bytes())
    assert receipt["report_sha256"] == sha256_bytes(expected_bytes)
    for path in (publication.args.project / ".pipeline/imports" / status["job_id"] / "quality_report.json",
                 publication.args.project / "reconstruct/quality_report.json"):
        assert path.read_bytes() == expected_bytes
    assert report_path.read_bytes() != expected_bytes
    assert _job(publication)[0] == "succeeded"


def test_changed_baseline_rejected_before_job_or_backup(publication):
    changed = publication.args.project / "reconstruct/sparse/0/images.bin"
    changed.write_bytes(b"newer published reconstruction")
    with pytest.raises(ValueError, match="baseline changed"):
        publication.publisher.execute(publication.args, {})
    assert _job(publication) is None
    assert changed.read_bytes() == b"newer published reconstruction"
    assert not (publication.args.output / "backup").exists()


def test_active_project_job_rejected_without_touching_originals(publication):
    with sqlite3.connect(publication.database) as connection:
        connection.execute("INSERT INTO job(id,project_id,kind,status,created_at) "
                           "VALUES('existing','project','rerun_stage','running','2026-09-18')")
    with pytest.raises(RuntimeError, match="active job"):
        publication.publisher.execute(publication.args, {})
    _assert_originals_restored(publication)
    assert not (publication.args.output / "backup").exists()
    with sqlite3.connect(publication.database) as connection:
        assert connection.execute("SELECT id,status FROM job").fetchall() == [("existing", "running")]


@pytest.mark.parametrize('change_features', [False, True])
def test_feature_identity_checks_content_not_sqlite_layout(publication, tmp_path, change_features):
    paths = [tmp_path / 'first.db', tmp_path / 'second.db']
    for index,path in enumerate(paths):
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)')
            db.execute("INSERT INTO images VALUES(1,'primary.jpg',1)")
            for table in ['keypoints','descriptors']:
                db.execute(f'CREATE TABLE {table}(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)')
                data = b'changed' if change_features and index == 1 else b'original'
                db.execute(f'INSERT INTO {table} VALUES(1,1,2,?)', (data,))
            if index:
                db.execute('CREATE TABLE extra_metadata(value TEXT)')
    if change_features:
        with pytest.raises(ValueError, match='keypoints differ'):
            publication.publisher.validate_feature_identity(*paths)
    else:
        assert publication.publisher.validate_feature_identity(*paths) == {'images':1, 'keypoints':1, 'descriptors':1}


@pytest.mark.parametrize(
    ("removed_images", "model_source", "export_points"),
    [(1, "cleanup_sparse", 0), (0, "scene_alignment", 0), (0, "cleanup_sparse", 1)],
)
def test_publication_rejects_camera_deletion_or_uncleaned_export(
    publication, monkeypatch, removed_images, model_source, export_points
):
    class IncorrectExportEngine:
        def __init__(self, *_args):
            pass

        def run_stage(self, stage, _params):
            (publication.args.project / stage.value).mkdir()
            if stage == StageName.CLEANUP_SPARSE:
                extra = {"removed_images": removed_images, "output_points": 0}
            elif stage == StageName.EXPORT_DATASET:
                extra = {"num_images": 2, "num_points3D": export_points,
                         "model_source": model_source, "validation": {"training_ready": True}}
            else:
                return
            StageManifest(stage=stage.value, impl_version="stub", started_at=datetime.now(UTC),
                          extra=extra).dump(publication.args.project / "manifests" / f"{stage.value}.json")

        def close(self):
            pass

    monkeypatch.setattr("sphere_reconstruct.pipeline.engine.Engine", IncorrectExportEngine)
    with pytest.raises(RuntimeError, match="must export the cleaned points"):
        publication.publisher.execute(publication.args, {})
    _assert_originals_restored(publication)
    assert _job(publication)[0] == "failed"
