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
from ..infrastructure.filesystem import sha256_file
from ..insta360 import calibration as calib
from ..insta360 import imu as insv_imu
from ..insta360 import insv
from ..insta360 import metadata as insv_metadata
from ..insta360 import protobuf as pb
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class InspectSource(Stage):
    name = StageName.INSPECT_SOURCE
    impl_version = "0.2"  # gravity 抽出を追加

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        if ctx.source_path is None:
            return []
        p = ctx.source_path
        if not p.exists():
            raise FileNotFoundError(f"source not found: {p}")
        return [
            FileRef(
                path=str(p),
                size=p.stat().st_size,
                sha256=sha256_file(p),
            )
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        if ctx.source_path is None or ctx.source_kind is None:
            raise RuntimeError("source is not set. call POST /api/projects/{id}/source first.")

        out_dir = ctx.stage_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        source_kind = ctx.source_kind
        source_path = ctx.source_path

        if source_kind == "insv":
            summary = self._inspect_insv(source_path, out_dir, ctx)
        elif source_kind == "erp_video":
            summary = self._inspect_erp_video(source_path, ctx)
        elif source_kind == "erp_images":
            summary = self._inspect_erp_images(source_path, ctx)
        else:
            raise ValueError(f"unknown source_kind: {source_kind}")

        (out_dir / "source.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.outputs = [
            FileRef(
                path=str((out_dir / "source.json").relative_to(ctx.project_dir)),
                size=(out_dir / "source.json").stat().st_size,
                sha256=sha256_file(out_dir / "source.json"),
                mime="application/json",
            )
        ]
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
                        "lenses": [l.to_dict() for l in parsed.lenses] if parsed.is_valid() else [],
                        "text": chosen.text,
                    }
                    if parsed.is_valid():
                        summary["calibration_source"] = "offset_v3"
                else:
                    summary["offset_v3"] = {"found": False}

                # IMU (Gyro record) から重力方向 (IMU 座標) を抽出. 再構成の重力対齐に使う.
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
    def _inspect_erp_video(self, path: Path, ctx: StageContext) -> dict:
        from ..imaging import ffprobe as _ffprobe
        from ..settings import get_settings

        ctx.progress.info("ffprobe on erp video", progress=0.5, key="log.inspect_ffprobe_erp")
        pr = _ffprobe.probe(path, ffprobe_bin=get_settings().binaries.ffprobe or None)
        vs = pr.video_streams[0] if pr.video_streams else None
        ctx.progress.info("inspect done", progress=1.0, key="log.inspect_done")
        return {
            "kind": "erp_video",
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

    def _inspect_erp_images(self, path: Path, ctx: StageContext) -> dict:
        if not path.is_dir():
            raise NotADirectoryError(f"erp_images source must be a directory: {path}")
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        ctx.progress.info(
            f"erp images: {len(images)} files",
            progress=1.0,
            key="log.inspect_erp_images",
            args={"count": len(images)},
        )
        return {
            "kind": "erp_images",
            "path": str(path),
            "image_count": len(images),
            "first_files": [p.name for p in images[:5]],
        }
