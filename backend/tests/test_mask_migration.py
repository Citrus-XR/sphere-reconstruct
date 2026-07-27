"""単一 mask artifact を Training-mask Step へ移す one-shot migration を検証する。"""

from __future__ import annotations

import json

import pytest

from sphere_reconstruct.infrastructure.database import Database
from sphere_reconstruct.infrastructure.migrations import _upgrade_mask_manifest


@pytest.mark.parametrize(
    ("kind", "frame_payload", "expected_names"),
    [
        (
            "sam3_fisheye_masks",
            {"index": 2, "lenses": [
                {"lens": 0, "path": "generate_masks/lens0/frame_000002.png", "coverage": 0.1},
                {"lens": 1, "path": "generate_masks/lens1/frame_000002.png", "coverage": 0.2},
            ]},
            ["front/frame_000002.jpg", "back/frame_000002.jpg"],
        ),
        (
            "sam3_pinhole_masks",
            {"index": 2, "views": [
                {"view": "front", "lens": 0, "path": "generate_masks/frame_000002/front_lens0.png"},
                {"view": "right", "lens": 1, "path": "generate_masks/frame_000002/right_lens1.png"},
            ]},
            ["front_lens0/frame_000002.jpg", "right_lens1/frame_000002.jpg"],
        ),
        (
            "sam3_erp_masks",
            {"index": 2, "path": "generate_masks/frame_000002.png", "coverage": 0.3},
            ["frame_000002.jpg"],
        ),
    ],
)
def test_pre_catalog_mask_layouts_are_flattened(kind, frame_payload, expected_names):
    result = _upgrade_mask_manifest(
        {
            "kind": kind,
            "prompt": ["person"],
            "max_inference_size": 1024,
            "dilate_px": 8,
            "frames": [frame_payload],
        },
        "primary",
    )

    assert result["version"] == 3
    assert result["purpose"] == "training"
    assert [record["name"] for record in result["images"]] == expected_names
    assert all(record["source_id"] == "primary" for record in result["images"])
    assert all("generate_training_masks" in record["path"] for record in result["images"])


@pytest.mark.asyncio
async def test_legacy_mask_artifact_and_ui_state_are_migrated(tmp_path):
    db_path = tmp_path / "state.db"
    database = Database(db_path)
    await database.connect()
    metadata = {
        "ui": {
            "params": {
                "maskSize": 1024,
                "downsampleOn": True,
                "dilate": 8,
                "dilateOn": True,
                "prompt": "person",
                "featureType": "SIFT",
            },
            "disabled": ["generate_masks"],
        }
    }
    await database.conn.execute(
        "INSERT INTO project (id,name,created_at,updated_at,state,metadata_json) VALUES (?,?,?,?,?,?)",
        ("p", "Project", "2026-01-01", "2026-01-01", "masked", json.dumps(metadata)),
    )
    await database.conn.execute(
        "INSERT INTO job (id,project_id,kind,stage,status,created_at) VALUES (?,?,?,?,?,?)",
        ("j", "p", "rerun_stage", "generate_masks", "succeeded", "2026-01-01"),
    )
    await database.conn.execute(
        """INSERT INTO stage_run
        (id,project_id,job_id,stage,impl_version,params_hash,inputs_hash,status,started_at,manifest_path)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            "r",
            "p",
            "j",
            "generate_masks",
            "1.0",
            "params",
            "inputs",
            "succeeded",
            "2026-01-01",
            str(tmp_path / "projects/p/manifests/generate_masks.json"),
        ),
    )
    await database.conn.execute(
        "INSERT INTO event (job_id,project_id,stage,level,message,kind,ts) VALUES (?,?,?,?,?,?,?)",
        ("j", "p", "generate_masks", "info", "done", "log", "2026-01-01"),
    )
    await database.conn.commit()
    await database.close()

    project = tmp_path / "projects" / "p"
    old = project / "generate_masks"
    mask = old / "images/frame.jpg.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"mask")
    (old / "manifest_masks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "prompt": ["person"],
                "images": [
                    {
                        "name": "images/frame.jpg",
                        "path": "generate_masks/images/frame.jpg.png",
                    }
                ],
            }
        )
    )
    manifests = project / "manifests"
    manifests.mkdir()
    (manifests / "generate_masks.json").write_text(
        json.dumps(
            {
                "stage": "generate_masks",
                "outputs": [{"path": "generate_masks/images/frame.jpg.png"}],
            }
        )
    )
    stale = project / ".pipeline/stale"
    stale.mkdir(parents=True)
    (stale / "generate_masks.json").write_text(
        json.dumps({"stage": "generate_masks", "invalidated_by": "prepare_images"})
    )

    migrated = Database(db_path)
    await migrated.connect()

    assert not old.exists()
    training = project / "generate_training_masks"
    document = json.loads((training / "manifest_masks.json").read_text())
    assert document["version"] == 3
    assert document["purpose"] == "training"
    assert document["images"][0]["path"] == "generate_training_masks/images/frame.jpg.png"
    assert (manifests / "generate_training_masks.json").is_file()
    assert not (manifests / "generate_masks.json").exists()
    assert (stale / "generate_training_masks.json").is_file()

    project_row = await (
        await migrated.conn.execute("SELECT metadata_json FROM project WHERE id='p'")
    ).fetchone()
    ui = json.loads(project_row["metadata_json"])["ui"]
    assert "disabled" not in ui
    assert ui["params"]["featureMaskEnabled"] is False
    assert ui["params"]["trainingMaskEnabled"] is False
    assert "maskSize" not in ui["params"]
    assert "prompt" not in ui["params"]
    assert ui["params"]["featureType"] == "SIFT"

    stage_run = await (
        await migrated.conn.execute("SELECT stage, manifest_path FROM stage_run WHERE id='r'")
    ).fetchone()
    assert stage_run["stage"] == "generate_training_masks"
    assert "generate_training_masks.json" in stage_run["manifest_path"]
    event = await (await migrated.conn.execute("SELECT stage FROM event")).fetchone()
    job = await (await migrated.conn.execute("SELECT stage FROM job WHERE id='j'")).fetchone()
    assert event["stage"] == "generate_training_masks"
    assert job["stage"] == "generate_training_masks"
    await migrated.close()
