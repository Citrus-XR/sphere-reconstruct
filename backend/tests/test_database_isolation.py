"""明示 transaction と独立 SQL 呼び出しの隔離を検証する。"""

import asyncio

import pytest

from sphere_reconstruct.infrastructure.database import Database


@pytest.fixture
async def database(tmp_path):
    database = Database(tmp_path / "state.db")
    await database.connect()
    await database.conn.execute("CREATE TABLE isolation_probe (value TEXT NOT NULL)")
    await database.conn.commit()
    try:
        yield database
    finally:
        await database.close()


async def test_independent_commit_cannot_publish_transaction_changes(database):
    with pytest.raises(ValueError, match="rollback"):
        async with database.transaction() as connection:
            await connection.execute("INSERT INTO isolation_probe VALUES ('private')")
            await database.conn.commit()
            rows = await (await database.conn.execute("SELECT * FROM isolation_probe")).fetchall()
            assert rows == []
            raise ValueError("rollback")
    rows = await (await database.conn.execute("SELECT * FROM isolation_probe")).fetchall()
    assert rows == []


async def test_independent_rollback_cannot_erase_transaction_changes(database):
    async with database.transaction() as connection:
        await connection.execute("INSERT INTO isolation_probe VALUES ('committed')")
        await database.conn.rollback()
    rows = await (await database.conn.execute("SELECT value FROM isolation_probe")).fetchall()
    assert [row["value"] for row in rows] == ["committed"]


async def test_overlapping_transactions_keep_independent_outcomes(database):
    first_entered = asyncio.Event()
    second_attempted = asyncio.Event()
    second_entered = asyncio.Event()

    async def failing_transaction():
        with pytest.raises(ValueError, match="rollback"):
            async with database.transaction() as connection:
                await connection.execute("INSERT INTO isolation_probe VALUES ('discarded')")
                first_entered.set()
                await second_attempted.wait()
                await asyncio.sleep(0)
                assert not second_entered.is_set()
                raise ValueError("rollback")

    async def successful_transaction():
        await first_entered.wait()
        second_attempted.set()
        async with database.transaction() as connection:
            second_entered.set()
            await connection.execute("INSERT INTO isolation_probe VALUES ('committed')")

    await asyncio.wait_for(asyncio.gather(failing_transaction(), successful_transaction()), timeout=5)
    rows = await (await database.conn.execute("SELECT value FROM isolation_probe")).fetchall()
    assert [row["value"] for row in rows] == ["committed"]


async def test_cancelled_transaction_rolls_back_and_releases_connection(database):
    entered = asyncio.Event()

    async def cancelled_transaction():
        async with database.transaction() as connection:
            await connection.execute("INSERT INTO isolation_probe VALUES ('discarded')")
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(cancelled_transaction())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with database.transaction() as connection:
        await connection.execute("INSERT INTO isolation_probe VALUES ('committed')")
    rows = await (await database.conn.execute("SELECT value FROM isolation_probe")).fetchall()
    assert [row["value"] for row in rows] == ["committed"]


async def test_transaction_connection_enforces_foreign_keys(database):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        async with database.transaction() as connection:
            await connection.execute(
                "INSERT INTO job (id, project_id, kind, status, created_at) "
                "VALUES ('job', 'missing', 'run_pipeline', 'queued', '2026-09-07')"
            )
