"""パイプライン全体の状態遷移.

Main reconstruction branch は EXTRACTED から FEATURES_EXTRACTED -> MATCHED -> RECONSTRUCTED ->
ALIGNED -> EXPORTED へ進む。PREPARED / MASKED は camera adapter と optional stage に応じて
main branch の途中へ入る。

各ステージは冪等。入力が変わった場合は consumer graph だけを transitive invalidate し、独立
branch の再生成可能 artifact を保持する。PipelineState は UI / DB 用の要約で、artifact の真偽は
manifest を基準にする。
"""

from __future__ import annotations

from enum import StrEnum


class PipelineState(StrEnum):
    CREATED = "created"
    INSPECTED = "inspected"
    EXTRACTED = "extracted"
    PREPARED = "prepared"
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
    PipelineState.PREPARED: 3,
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
    PREPARE_IMAGES = "prepare_images"
    GENERATE_MASKS = "generate_masks"
    EXTRACT_FEATURES = "extract_features"
    MATCH_FEATURES = "match_features"
    RECONSTRUCT = "reconstruct"
    ALIGN_RECONSTRUCTION = "align_reconstruction"
    EXPORT_DATASET = "export_dataset"


STAGE_TO_STATE: dict[StageName, PipelineState] = {
    StageName.INSPECT_SOURCE: PipelineState.INSPECTED,
    StageName.EXTRACT_FRAMES: PipelineState.EXTRACTED,
    StageName.PREPARE_IMAGES: PipelineState.PREPARED,
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
    StageName.PREPARE_IMAGES,
    StageName.GENERATE_MASKS,
    StageName.EXTRACT_FEATURES,
    StageName.MATCH_FEATURES,
    StageName.RECONSTRUCT,
    StageName.ALIGN_RECONSTRUCTION,
    StageName.EXPORT_DATASET,
)


_STAGE_CONSUMERS: dict[StageName, tuple[StageName, ...]] = {
    StageName.INSPECT_SOURCE: (StageName.EXTRACT_FRAMES,),
    StageName.EXTRACT_FRAMES: (StageName.PREPARE_IMAGES,),
    StageName.PREPARE_IMAGES: (StageName.GENERATE_MASKS, StageName.EXTRACT_FEATURES),
    StageName.GENERATE_MASKS: (StageName.EXTRACT_FEATURES,),
    StageName.EXTRACT_FEATURES: (StageName.MATCH_FEATURES,),
    StageName.MATCH_FEATURES: (StageName.RECONSTRUCT,),
    StageName.RECONSTRUCT: (StageName.ALIGN_RECONSTRUCTION,),
    StageName.ALIGN_RECONSTRUCTION: (StageName.EXPORT_DATASET,),
    StageName.EXPORT_DATASET: (),
}


def downstream_of(stage: StageName, *, include_self: bool = True) -> list[StageName]:
    """Artifact DAG 上で stage の transitive consumer を実行順に返す。"""
    pending = list(_STAGE_CONSUMERS[stage])
    found = {stage} if include_self else set()
    while pending:
        consumer = pending.pop()
        if consumer in found:
            continue
        found.add(consumer)
        pending.extend(_STAGE_CONSUMERS[consumer])
    return [item for item in STAGE_ORDER if item in found]
