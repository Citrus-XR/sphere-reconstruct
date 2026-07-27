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


def matching_progress(ctx: StageContext, *, low: float, high: float):
    image_pattern = re.compile(r"(?:Processing|Matching) (?:image|file) \[(\d+)/(\d+)\]", re.I)
    block_pattern = re.compile(
        r"(?:Processing|Matching) block \[(\d+)/(\d+)\s*,\s*(\d+)/(\d+)\]", re.I
    )
    state = {"fraction": 0.0}

    def emit_fraction(fraction: float, message: str) -> None:
        state["fraction"] = max(state["fraction"], min(1.0, fraction))
        ctx.progress.tick(
            progress=low + (high - low) * state["fraction"],
            message=message,
            key="log.matching_progress",
            args={"percent": round(state["fraction"] * 100, 1)},
        )

    def callback(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[matching] {line}")
        block = block_pattern.search(line)
        if block is not None:
            first, first_total, second, second_total = map(int, block.groups())
            linear = (first - 1) * second_total + second
            emit_fraction(linear / max(1, first_total * second_total), line)
            return
        image = image_pattern.search(line)
        if image is not None:
            current, total = map(int, image.groups())
            emit_fraction(current / max(1, total), line)

    return callback


def global_mapper_progress(ctx: StageContext, *, low: float = 0.15, high: float = 0.9):
    phases = (
        ("Decomposing relative poses", 0.05),
        ("rotation averaging", 0.2),
        ("track establishment", 0.38),
        ("global positioning", 0.55),
        ("bundle adjustment", 0.72),
        ("retriangulation", 0.88),
        ("Extracting colors", 1.0),
    )
    state = {"fraction": 0.0}

    def callback(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[global-mapper] {line}")
        lowered = line.lower()
        for phase, fraction in phases:
            if phase.lower() in lowered:
                state["fraction"] = max(state["fraction"], fraction)
                ctx.progress.tick(
                    progress=low + (high - low) * state["fraction"], message=phase
                )
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
