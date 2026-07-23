"""パイプライン全体の状態遷移.

CREATED -> INSPECTED -> EXTRACTED -> REPROJECTED -> MASKED -> RECONSTRUCTED -> EXPORTED.

各ステージは冪等. 途中でパラメータや入力ハッシュが変わったら, そのステージから下は
invalidate する (呼び出し側の責任).
"""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    CREATED = "created"
    INSPECTED = "inspected"
    EXTRACTED = "extracted"
    REPROJECTED = "reprojected"
    MASKED = "masked"
    RECONSTRUCTED = "reconstructed"
    EXPORTED = "exported"

    def order(self) -> int:
        return _ORDER[self]

    def is_after(self, other: "PipelineState") -> bool:
        return self.order() > other.order()


_ORDER: dict[PipelineState, int] = {
    PipelineState.CREATED: 0,
    PipelineState.INSPECTED: 1,
    PipelineState.EXTRACTED: 2,
    PipelineState.REPROJECTED: 3,
    PipelineState.MASKED: 4,
    PipelineState.RECONSTRUCTED: 5,
    PipelineState.EXPORTED: 6,
}


class StageName(str, Enum):
    """ステージ名. state と 1:1 対応する (CREATED は入力状態なのでステージなし)."""

    INSPECT_SOURCE = "inspect_source"
    EXTRACT_FRAMES = "extract_frames"
    REPROJECT_VIEWS = "reproject_views"
    GENERATE_MASKS = "generate_masks"
    RECONSTRUCT = "reconstruct"
    EXPORT_DATASET = "export_dataset"


STAGE_TO_STATE: dict[StageName, PipelineState] = {
    StageName.INSPECT_SOURCE: PipelineState.INSPECTED,
    StageName.EXTRACT_FRAMES: PipelineState.EXTRACTED,
    StageName.REPROJECT_VIEWS: PipelineState.REPROJECTED,
    StageName.GENERATE_MASKS: PipelineState.MASKED,
    StageName.RECONSTRUCT: PipelineState.RECONSTRUCTED,
    StageName.EXPORT_DATASET: PipelineState.EXPORTED,
}

# 実行順序.
STAGE_ORDER: tuple[StageName, ...] = (
    StageName.INSPECT_SOURCE,
    StageName.EXTRACT_FRAMES,
    StageName.REPROJECT_VIEWS,
    StageName.GENERATE_MASKS,
    StageName.RECONSTRUCT,
    StageName.EXPORT_DATASET,
)


def downstream_of(stage: StageName) -> list[StageName]:
    """stage を含めた, それ以降のステージ (invalidate 対象)."""
    idx = STAGE_ORDER.index(stage)
    return list(STAGE_ORDER[idx:])
