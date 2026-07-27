"""永続 DB の状態値 migration を検証する。"""

from sphere_reconstruct.infrastructure.database import Database


async def test_removed_branch_state_is_migrated_to_aligned(tmp_path):
    path = tmp_path / "state.db"
    database = Database(path)
    await database.connect()
    await database.conn.execute(
        "INSERT INTO project (id, name, created_at, updated_at, state) VALUES (?, ?, ?, ?, ?)",
        ("legacy", "legacy", "2026-01-01", "2026-01-01", "denoised"),
    )
    await database.conn.commit()
    await database.close()

    migrated = Database(path)
    await migrated.connect()
    row = await (await migrated.conn.execute("SELECT state FROM project WHERE id='legacy'")).fetchone()
    await migrated.close()

    assert row["state"] == "aligned"


async def test_legacy_single_source_is_migrated_to_primary_source(tmp_path):
    path = tmp_path / "state.db"
    database = Database(path)
    await database.connect()
    await database.conn.execute(
        "INSERT INTO project "
        "(id, name, created_at, updated_at, state, source_kind, source_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-source",
            "legacy",
            "2026-01-01",
            "2026-01-01",
            "created",
            "insv",
            "D:/capture.insv",
        ),
    )
    await database.conn.commit()
    await database.close()

    migrated = Database(path)
    await migrated.connect()
    row = await (
        await migrated.conn.execute(
            "SELECT role, adapter, media_kind, projection, path "
            "FROM project_source WHERE project_id='legacy-source'"
        )
    ).fetchone()
    legacy = await (
        await migrated.conn.execute("SELECT source_kind, source_path FROM project WHERE id='legacy-source'")
    ).fetchone()
    await migrated.close()

    assert tuple(row) == (
        "primary",
        "insta360_insv",
        "video",
        "dual_fisheye",
        "D:/capture.insv",
    )
    assert tuple(legacy) == (None, None)
