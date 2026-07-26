"""パイプライン全体の状態遷移.

CREATED -> INSPECTED -> EXTRACTED -> REPROJECTED -> MASKED -> FEATURES_EXTRACTED ->
MATCHED -> RECONSTRUCTED -> ALIGNED -> EXPORTED.

各ステージは冪等. 途中でパラメータや入力ハッシュが変わったら, そのステージから下は
invalidate する (呼び出し側の責任).
"""

from __future__ import annotations

from enum import StrEnum


class PipelineState(StrEnum):
    CREATED = "created"
    INSPECTED = "inspected"
    EXTRACTED = "extracted"
    REPROJECTED = "reprojected"
    MASKED = "masked"
    FEATURES_EXTRACTED = "features_extracted"
    MATCHED = "matched"
    RECONSTRUCTED = "reconstructed"
    ALIGNED = "aligned"
    EXPORTED = "exported"

    def order(self) -> int:
        return _ORDER[self]

    def is_after(self, other: PipelineState) -> bool:
        return self.order() > other.order()


_ORDER: dict[PipelineState, int] = {
    PipelineState.CREATED: 0,
    PipelineState.INSPECTED: 1,
    PipelineState.EXTRACTED: 2,
    PipelineState.REPROJECTED: 3,
    PipelineState.MASKED: 4,
    PipelineState.FEATURES_EXTRACTED: 5,
    PipelineState.MATCHED: 6,
    PipelineState.RECONSTRUCTED: 7,
    PipelineState.ALIGNED: 8,
    PipelineState.EXPORTED: 9,
}


class StageName(StrEnum):
    """ステージ名. state と 1:1 対応する (CREATED は入力状態なのでステージなし)."""

    INSPECT_SOURCE = "inspect_source"
    EXTRACT_FRAMES = "extract_frames"
    REPROJECT_VIEWS = "reproject_views"
    GENERATE_MASKS = "generate_masks"
    EXTRACT_FEATURES = "extract_features"
    MATCH_FEATURES = "match_features"
    RECONSTRUCT = "reconstruct"
    ALIGN_RECONSTRUCTION = "align_reconstruction"
    EXPORT_DATASET = "export_dataset"


STAGE_TO_STATE: dict[StageName, PipelineState] = {
    StageName.INSPECT_SOURCE: PipelineState.INSPECTED,
    StageName.EXTRACT_FRAMES: PipelineState.EXTRACTED,
    StageName.REPROJECT_VIEWS: PipelineState.REPROJECTED,
    StageName.GENERATE_MASKS: PipelineState.MASKED,
    StageName.EXTRACT_FEATURES: PipelineState.FEATURES_EXTRACTED,
    StageName.MATCH_FEATURES: PipelineState.MATCHED,
    StageName.RECONSTRUCT: PipelineState.RECONSTRUCTED,
    StageName.ALIGN_RECONSTRUCTION: PipelineState.ALIGNED,
    StageName.EXPORT_DATASET: PipelineState.EXPORTED,
}

# 実行順序.
STAGE_ORDER: tuple[StageName, ...] = (
    StageName.INSPECT_SOURCE,
    StageName.EXTRACT_FRAMES,
    StageName.REPROJECT_VIEWS,
    StageName.GENERATE_MASKS,
    StageName.EXTRACT_FEATURES,
    StageName.MATCH_FEATURES,
    StageName.RECONSTRUCT,
    StageName.ALIGN_RECONSTRUCTION,
    StageName.EXPORT_DATASET,
)


def downstream_of(stage: StageName, *, include_self: bool = True) -> list[StageName]:
    """線形化した artifact DAG 上で stage 以降の consumer を返す."""
    idx = STAGE_ORDER.index(stage)
    return list(STAGE_ORDER[idx if include_self else idx + 1 :])
