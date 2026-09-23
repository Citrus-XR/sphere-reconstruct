from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from sphere_reconstruct.colmap import runner
from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.colmap.sequence_pairs import missing_sequence_pairs
from sphere_reconstruct.domain.source import MediaKind
from sphere_reconstruct.pipeline.stage import ProgressReporter
from sphere_reconstruct.stages.match_features import MatchFeatures


def database(tmp_path, rows, *, attempted=(), verified=()):
    path = tmp_path / "database.db"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE images(image_id INTEGER, name TEXT); CREATE TABLE matches(pair_id INTEGER,rows INTEGER); CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER);")
        db.executemany("INSERT INTO images VALUES (?, ?)", rows)
        db.executemany("INSERT INTO matches VALUES (?, ?)", attempted)
        db.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", verified)
    return path


def test_retrieval_gaps_use_selected_rank_and_keep_sensors_and_sources_separate(tmp_path):
    records = [{"name": f"{source}/{sensor}/{capture}.png", "source_id": source,
                "sensor_id": sensor, "capture_index": capture}
               for source in ["video", "photos"] for sensor in ["front", "back"]
               for capture in [1000, 1004, 1020]]
    ids = {row["name"]: index for index, row in enumerate(records, 1)}
    # An attempted failed match and a verified-only pair must not be retried.
    path = database(tmp_path, [(value, name) for name, value in ids.items()],
                    attempted=[(1 * 2147483647 + 2, 0)], verified=[(4 * 2147483647 + 5, 50)])

    pairs, report = missing_sequence_pairs(path, records[::-1], {"video"}, overlap=1)

    assert pairs == [("video/back/1004.png", "video/back/1020.png"),
                     ("video/front/1004.png", "video/front/1020.png")]
    assert report["expected_pairs"] == 4
    assert report["missing_pairs"] == 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM matches").fetchall() == [(2147483649, 0)]


def test_ordered_photo_source_requires_explicit_selection(tmp_path):
    records = [{"name": str(i), "source_id": "photos", "sensor_id": "main", "capture_index": i} for i in range(3)]
    path = database(tmp_path, [(i + 1, str(i)) for i in range(3)])
    assert missing_sequence_pairs(path, records, set(), overlap=2)[0] == []
    assert len(missing_sequence_pairs(path, records, {"photos"}, overlap=2)[0]) == 3
    with pytest.raises(ValueError, match="absent"):
        missing_sequence_pairs(path, records, {"unknown"}, overlap=2)


def test_sequence_matching_preserves_calculation_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_command", lambda binary, arguments, **kwargs: calls.append((binary, arguments, kwargs)))
    runner.explicit_pairs_matcher("colmap", database_path=tmp_path / "database.db",
                                  match_list_path=tmp_path / "pairs.txt", use_gpu=False,
                                  matching_type="SIFT_LIGHTGLUE",
                                  extra_args=["--FeatureMatching.guided_matching", "1"])
    command = calls[0][1]
    assert command[0] == "matches_importer"
    assert command[command.index("--match_type") + 1] == "pairs"
    assert command[command.index("--FeatureMatching.use_gpu") + 1] == "0"
    assert command[command.index("--FeatureMatching.type") + 1] == "SIFT_LIGHTGLUE"
    assert command[command.index("--FeatureMatching.guided_matching") + 1] == "1"


@pytest.mark.parametrize("invalid", ["phone", [1], [""]])
def test_ordered_source_parameter_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="ordered_image_source_ids"):
        MatchFeatures().normalize_params({"ordered_image_source_ids": invalid})


@pytest.mark.parametrize("media_kind,explicit", [(MediaKind.VIDEO, []), (MediaKind.IMAGES, ["source"])])
def test_retrieval_stage_adds_missing_neighbors_after_global_matching(tmp_path, monkeypatch, media_kind, explicit):
    from sphere_reconstruct.stages import match_features as module

    features = tmp_path / "extract_features"
    features.mkdir()
    database(features, [(1, "a.png"), (2, "b.png"), (3, "c.png")])
    output = tmp_path / "output"
    output.mkdir()
    records = [{"name": name, "source_id": "source", "sensor_id": "main", "capture_index": index, "timestamp_sec": index}
               for index, name in enumerate(["a.png", "b.png", "c.png"])]
    spec = InputSpec(version=3, reconstruction_mode="native_fisheye", image_count=3, source_count=1,
                     primary_source_id="source", primary_image_names=[row["name"] for row in records],
                     sources=[{"id": "source", "label": "Source", "role": "primary"}], images=records,
                     feature_batches=[], image_path="images", mask_path=None, feature_masks_enabled=False,
                     rig_config_path=None, refine_intrinsics=False, refine_rig=False, multiple_models=False)
    monkeypatch.setattr(module.InputSpec, "read", lambda _path: spec)
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(binaries=SimpleNamespace(colmap="colmap", vocab_tree="tree")))
    monkeypatch.setattr(runner, "resolve_colmap_bin", lambda value: value)
    monkeypatch.setattr(runner, "resolve_vocab_tree_path", lambda value: tmp_path / value)
    calls = []

    def retrieval(_binary, **kwargs):
        with sqlite3.connect(kwargs["database_path"]) as db:
            db.execute("INSERT INTO matches VALUES (?,100)", (1 * 2147483647 + 2,))
            db.execute("INSERT INTO two_view_geometries VALUES (?,80)", (1 * 2147483647 + 2,))
        calls.append("retrieval")

    def neighbors(_binary, **kwargs):
        assert kwargs["match_list_path"].read_text() == "b.png c.png\n"
        assert kwargs["matching_type"] == "SIFT_BRUTEFORCE"
        calls.append("neighbors")

    monkeypatch.setattr(runner, "vocab_tree_matcher", retrieval)
    monkeypatch.setattr(runner, "explicit_pairs_matcher", neighbors)
    context = SimpleNamespace(project_dir=tmp_path, stage_out_dir=output,
                              sources=(SimpleNamespace(id="source", media_kind=media_kind),),
                              params=MatchFeatures().normalize_params({"pairing": "vocab_tree", "overlap": 1,
                                                                        "ordered_image_source_ids": explicit}),
                              progress=ProgressReporter(lambda *_args: None), inputs_for=lambda _stage: [])

    result = MatchFeatures().execute(context)

    assert calls == ["retrieval", "neighbors"]
    assert result.extra["sequence_matching"]["missing_pairs"] == 1
