"""inspect_source ステージ.

ソース (INSV / ERP video / ERP images) を検査し, 統一メタデータ JSON を出力する.
このステージは軽量: ffprobe は使わず, ファイル構造直接解析 + 拡張子判定のみ.

出力:
- <project>/inspect_source/source.json   統一メタデータ
- <project>/inspect_source/footer_records.json  INSV フッタの record type 一覧 (debug)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind, SourceAdapter
from ..infrastructure.filesystem import sha256_file
from ..insta360 import calibration as calib
from ..insta360 import imu as insv_imu
from ..insta360 import insv
from ..insta360 import metadata as insv_metadata
from ..insta360 import protobuf as pb
from ..pipeline.manifest import register
from ..pipeline.source_inputs import IMAGE_EXTENSIONS, collect_source_inputs
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class InspectSource(Stage):
    name = StageName.INSPECT_SOURCE
    impl_version = "2.0"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        return collect_source_inputs(ctx.sources)

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        if not ctx.sources:
            raise RuntimeError("project has no enabled sources")

        out_dir = ctx.stage_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for source in ctx.sources:
            source_out = out_dir / "sources" / source.id
            source_out.mkdir(parents=True, exist_ok=True)
            if source.adapter == SourceAdapter.INSTA360_INSV:
                summary = self._inspect_insv(source.path, source_out, ctx)
            elif source.media_kind == MediaKind.VIDEO:
                summary = self._inspect_video(source.path, ctx)
            else:
                summary = self._inspect_images(source.path, ctx)
            summaries.append(
                {
                    "id": source.id,
                    "label": source.label,
                    "role": source.role.value,
                    "adapter": source.adapter.value,
                    "media_kind": source.media_kind.value,
                    "projection": source.projection.value,
                    **summary,
                }
            )

        summary_path = out_dir / "sources.json"
        summary_path.write_text(
            json.dumps({"version": 2, "sources": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.outputs = [
            FileRef(
                path=str(summary_path.relative_to(ctx.project_dir)),
                size=summary_path.stat().st_size,
                sha256=sha256_file(summary_path),
                mime="application/json",
            )
        ]
        manifest.extra = _source_statistics(summaries)
        return manifest

    def _inspect_insv(self, path: Path, out_dir: Path, ctx: StageContext) -> dict:
        ctx.progress.info("scanning MP4 boxes", progress=0.1, key="log.inspect_scan_mp4")
        layout = insv.layout(path)

        summary: dict = {
            "kind": "insv",
            "path": str(path),
            "file_size": layout.file_size,
            "mp4_boxes": [
                {"type": b.box_type.decode("latin1", "replace"), "offset": b.offset, "size": b.total_size}
                for b in layout.boxes
            ],
            "footer_offset": layout.footer_offset,
            "footer_size": layout.footer_size,
            "footer": None,
            "offset_v3": None,
            "gravity": None,
            "pb": None,
            "calibration_source": None,
        }

        if layout.footer_offset is not None:
            ctx.progress.info(
                "scanning Insta360 footer (inst box)", progress=0.4, key="log.inspect_scan_footer"
            )
            try:
                view = insv_metadata.read_footer(path, layout.footer_offset)
                summary["footer"] = {
                    "inst_box_offset": view.inst_box_offset,
                    "inst_box_total_size": view.inst_box_total_size,
                    "inst_box_data_size": view.inst_box_data_size,
                    "trailer_offset": view.trailer_offset,
                    "trailer_size": view.trailer_size,
                    "trailer_header_offset": view.trailer_header_offset,
                    "reported_inst_data_size": view.reported_inst_data_size,
                    "version": view.version,
                    "signature_offset": view.signature_offset,
                    "signature_valid": view.signature_valid,
                }

                # inst box を丸ごと読んで offset_v3 ASCII 校准串を拾う.
                ctx.progress.info(
                    "parsing offset_v3 (ascii)", progress=0.55, key="log.inspect_parse_offset_v3"
                )
                inst_bytes = insv_metadata.read_inst_box_bytes(view)
                cands = calib.find_ascii_calibrations(inst_bytes)
                chosen = calib.pick_offset_v3(cands)
                if chosen is not None:
                    parsed = calib.parse_offset_v3_ascii(chosen)
                    summary["offset_v3"] = {
                        "found": True,
                        "candidate_count": len(cands),
                        "chosen_inst_offset": chosen.inst_offset,
                        "chosen_items": len(chosen.values),
                        "valid": parsed.is_valid(),
                        "calibration_id": parsed.raw.get("calibration_id"),
                        "lenses": [lens.to_dict() for lens in parsed.lenses] if parsed.is_valid() else [],
                        "text": chosen.text,
                    }
                    if parsed.is_valid():
                        summary["calibration_source"] = "offset_v3"
                else:
                    summary["offset_v3"] = {"found": False}

                # IMU（Gyro record）から重力方向（IMU 座標）を抽出し、再構成の重力整列に使う。
                grav = insv_imu.extract_gravity(view)
                if grav is not None:
                    summary["gravity"] = {
                        "imu": list(grav.gravity_imu),
                        "samples": grav.sample_count,
                        "mean_magnitude": round(grav.mean_magnitude, 4),
                    }
                    ctx.progress.info(
                        f"IMU gravity: {grav.gravity_imu} ({grav.sample_count} samples, "
                        f"|a|~{grav.mean_magnitude:.2f})",
                        progress=0.6,
                        key="log.inspect_gravity",
                        args={
                            "gx": round(grav.gravity_imu[0], 3),
                            "gy": round(grav.gravity_imu[1], 3),
                            "gz": round(grav.gravity_imu[2], 3),
                            "n": grav.sample_count,
                            "mag": round(grav.mean_magnitude, 2),
                        },
                    )
                else:
                    ctx.progress.info(
                        "IMU gravity not found (no Gyro record)",
                        progress=0.6,
                        key="log.inspect_gravity_none",
                    )
            except insv_metadata.FooterNotFoundError as e:
                ctx.progress.warn(
                    f"footer not detected: {e}",
                    key="log.inspect_footer_not_detected",
                    args={"error": str(e)},
                )

        ctx.progress.info("looking for external .insv.pb", progress=0.7, key="log.inspect_look_pb")
        pb_path = pb.find_pb_for(path)
        if pb_path is not None:
            probe = pb.probe(pb_path)
            summary["pb"] = {"path": str(pb_path), "size": probe.size}
            # PB は最優先ソース. offset_v3 で埋めた calibration_source を上書きする.
            summary["calibration_source"] = "pb"
        elif summary.get("calibration_source") is None and layout.footer_offset is not None:
            # offset_v3 では埋まらなかったが footer は存在する -> 内蔵 profile への降級待ち.
            summary["calibration_source"] = "builtin_profile"

        ctx.progress.info("inspect done", progress=1.0, key="log.inspect_done")
        return summary

    # -- ERP video / images (最小実装) ---------------------------------------------
    def _inspect_video(self, path: Path, ctx: StageContext) -> dict:
        from ..imaging import ffprobe as _ffprobe
        from ..settings import get_settings

        ctx.progress.info("ffprobe on video source", progress=0.5, key="log.inspect_video")
        pr = _ffprobe.probe(path, ffprobe_bin=get_settings().binaries.ffprobe or None)
        vs = pr.video_streams[0] if pr.video_streams else None
        ctx.progress.info("inspect done", progress=1.0, key="log.inspect_done")
        return {
            "path": str(path),
            "file_size": path.stat().st_size,
            "duration_sec": pr.duration,
            "video": {
                "width": vs.width if vs else 0,
                "height": vs.height if vs else 0,
                "fps": vs.fps if vs else 0.0,
                "codec": vs.codec_name if vs else "",
                "nb_frames": vs.nb_frames if vs else None,
            }
            if vs
            else None,
        }

    def _inspect_images(self, path: Path, ctx: StageContext) -> dict:
        if not path.is_dir():
            raise NotADirectoryError(f"image source must be a directory: {path}")
        images = sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        ctx.progress.info(
            f"image collection: {len(images)} files",
            progress=1.0,
            key="log.inspect_images",
            args={"count": len(images)},
        )
        return {
            "path": str(path),
            "image_count": len(images),
            "first_files": [p.name for p in images[:5]],
        }


def _source_statistics(summaries: list[dict]) -> dict:
    return {
        "sources": len(summaries),
        "videos": sum(source["media_kind"] == "video" for source in summaries),
        "image_collections": sum(source["media_kind"] == "images" for source in summaries),
        "source_images": sum(int(source.get("image_count", 0)) for source in summaries),
        "gravity_sources": sum(bool(source.get("gravity")) for source in summaries),
        "calibrated_dual_fisheye_sources": sum(
            bool((source.get("offset_v3") or {}).get("valid")) for source in summaries
        ),
    }
