"""パイプライン全体の結合テスト.

lavfi で合成した 2 stream MP4 を「INSV 相当」として渡し, inspect_source (最低限,
mp4 box 走査だけ動く) → extract_frames (paired) の 2 段が回ることを確認する.

INSV footer は存在しないので inspect_source は footer=None を報告するはず (但し
failed にはならない). extract_frames は 2 stream 分の JPEG を書く.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sphere_reconstruct.domain import project as project_domain
from sphere_reconstruct.domain import source as source_domain
from sphere_reconstruct.infrastructure.database import close_db, init_db
from sphere_reconstruct.job_supervisor import init_supervisor, shutdown_supervisor

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg / ffprobe not found in PATH",
)


def _make_dual_stream_mp4(dst: Path, duration: float = 2.0, fps: int = 10) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=160x160:rate={fps}",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=duration={duration}:size=160x160:rate={fps}",
        "-map",
        "0:v",
        "-map",
        "1:v",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        # .insv 拡張子でも MP4 として書きたいので明示指定.
        "-f",
        "mp4",
        str(dst),
    ]
    subprocess.run(args, check=True, capture_output=True)


async def _wait_job(db, jid: str, timeout: int = 60) -> tuple[str, str | None]:
    for _ in range(timeout * 4):
        await asyncio.sleep(0.25)
        cur = await db.conn.execute("SELECT status, error_text FROM job WHERE id=?", (jid,))
        row = await cur.fetchone()
        if row and row["status"] in ("succeeded", "failed", "cancelled"):
            return row["status"], row["error_text"]
    return "timeout", None


async def _run_pipeline(tmp_workspace: Path, src: Path):
    """FastAPI レイヤは経由せず, domain + supervisor 直接使う."""
    import os

    os.environ["SPHERE_WORKSPACE__ROOT"] = str(tmp_workspace)
    # lru_cache をクリアして env を再読ませる.
    from sphere_reconstruct.settings import get_settings

    get_settings.cache_clear()

    db_path = tmp_workspace / "state.db"
    db = await init_db(db_path)
    try:
        sup = init_supervisor(db, db_path)
        p = await project_domain.create_project(db, "integ")
        await source_domain.add_source(
            db,
            p.id,
            label="synthetic INSV",
            role=source_domain.SourceRole.PRIMARY,
            adapter=source_domain.SourceAdapter.INSTA360_INSV,
            media_kind=source_domain.MediaKind.VIDEO,
            projection=source_domain.Projection.DUAL_FISHEYE,
            path=str(src),
        )

        j1 = await sup.enqueue_run_pipeline(project_id=p.id, stage="inspect_source")
        st1, err1 = await _wait_job(db, j1, 30)
        assert st1 == "succeeded", f"inspect failed: {err1}"

        j2 = await sup.enqueue_run_pipeline(project_id=p.id, stage="extract_frames")
        st2, err2 = await _wait_job(db, j2, 60)
        assert st2 == "succeeded", f"extract failed: {err2}"

        project_dir = tmp_workspace / "projects" / p.id
        assert (project_dir / "inspect_source" / "sources.json").exists()
        mf_path = project_dir / "extract_frames" / "manifest_frames.json"
        assert mf_path.exists()
        mf = json.loads(mf_path.read_text())
        assert mf["sources"][0]["kind"] == "insv_dual"
        assert mf["count"] > 0
        # 相対パスが実際に解決できる (原子置換後の名前になっている) ことを確認.
        first = mf["frames"][0]
        assert (project_dir / first["lens0"]).exists()
        assert (project_dir / first["lens1"]).exists()
    finally:
        await shutdown_supervisor()
        await close_db()
        # lru_cache をクリアして次のテストに影響しないように.
        get_settings.cache_clear()


def test_full_pipeline_on_synthetic_dual_stream(tmp_path: Path):
    fake_insv = tmp_path / "fake.insv"
    _make_dual_stream_mp4(fake_insv, duration=2.0, fps=10)

    asyncio.run(_run_pipeline(tmp_path / "workspace", fake_insv))
