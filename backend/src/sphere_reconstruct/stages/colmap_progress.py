"""COLMAP の逐次ログを UI の低頻度進捗へ変換する共通 callback."""

from __future__ import annotations

import re

from ..pipeline.stage import StageContext


def hidden_log(ctx: StageContext, prefix: str):
    def callback(line: str) -> None:
        if line.strip():
            ctx.progress.tick(message=f"[{prefix}] {line}")

    return callback


def counted_progress(ctx: StageContext, prefix: str, pattern: str, *, low: float, high: float):
    compiled = re.compile(pattern)

    def callback(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[{prefix}] {line}")
        match = compiled.search(line)
        if match is None:
            return
        current, total = int(match.group(1)), int(match.group(2))
        fraction = min(1.0, current / max(1, total))
        ctx.progress.tick(
            progress=low + (high - low) * fraction,
            message=f"{prefix}: {current}/{total}",
            args={"done": current, "total": total},
        )

    return callback


def global_mapper_progress(ctx: StageContext):
    phases = (
        ("Decomposing relative poses", 0.15),
        ("rotation averaging", 0.25),
        ("track establishment", 0.4),
        ("global positioning", 0.55),
        ("bundle adjustment", 0.72),
        ("retriangulation", 0.88),
        ("Extracting colors", 0.96),
    )

    def callback(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[global-mapper] {line}")
        lowered = line.lower()
        for phase, progress in phases:
            if phase.lower() in lowered:
                ctx.progress.tick(progress=progress, message=phase)
                break

    return callback


def mapper_progress(
    ctx: StageContext,
    image_count: int,
    *,
    low: float = 0.1,
    high: float = 0.95,
):
    pattern = re.compile(r"Registering image")
    state = {"registered": 0}

    def callback(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[mapper] {line}")
        if pattern.search(line):
            state["registered"] += 1
            fraction = min(1.0, state["registered"] / max(1, image_count))
            ctx.progress.tick(
                progress=low + (high - low) * fraction,
                message=f"mapper: registered {state['registered']}/{image_count}",
                key="log.recon_mapper_progress",
                args={"done": state["registered"], "total": image_count},
            )

    return callback
