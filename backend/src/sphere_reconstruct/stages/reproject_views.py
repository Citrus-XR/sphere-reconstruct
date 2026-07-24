"""reproject_views ステージ.

extract_frames の出力 (lens0/lens1 の JPEG 対) + inspect_source の offset_v3 を
使い, 各時刻について pinhole rig 画像を書き出す.

V1 は固定で 6-view cubemap (90 deg FoV, 1024x1024). 出力:
  <project>/reproject_views/frame_XXXXXX/<view_name>_lens<idx>.jpg
  <project>/reproject_views/manifest_rig.json

呼び出し前提:
  ctx.source_kind == "insv"

パラメータ:
  size: int    (default 1024)
  fov_deg: float (default 90.0)
  max_frames: int (0 = 全部)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import projection, rendering
from ..infrastructure.filesystem import sha256_file
from ..insta360 import calibration as calib
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class ReprojectViews(Stage):
    name = StageName.REPROJECT_VIEWS
    impl_version = "0.2"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        # 入力ハッシュ: inspect_source の source.json + extract_frames の manifest_frames.json
        # (JPEG 全部 hash するとコスト高なので, これら 2 ファイルのハッシュだけで invalidate 判定).
        candidates = [
            ctx.project_dir / "inspect_source" / "source.json",
            ctx.project_dir / "extract_frames" / "manifest_frames.json",
        ]
        return [
            FileRef(path=str(p.relative_to(ctx.project_dir)), size=p.stat().st_size, sha256=sha256_file(p))
            for p in candidates
            if p.exists()
        ]

    def normalize_params(self, raw: dict) -> dict:
        return {
            "size": int(raw.get("size", 1024)),
            "fov_deg": float(raw.get("fov_deg", 90.0)),
            "max_frames": int(raw.get("max_frames", 0)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        source_json = ctx.project_dir / "inspect_source" / "source.json"
        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if not source_json.exists():
            raise RuntimeError("inspect_source must run first")
        if not frames_json.exists():
            raise RuntimeError("extract_frames must run first")

        src = json.loads(source_json.read_text())
        frames_mf = json.loads(frames_json.read_text())

        ov3 = src.get("offset_v3") or {}
        if not ov3.get("valid"):
            raise RuntimeError(
                "offset_v3 not valid; PB / builtin calibration paths not implemented yet"
            )
        lens_dicts = ov3["lenses"]
        # dict -> MeiLensCalibration.
        lenses = [_lens_from_dict(d) for d in lens_dicts]

        # ネイティブ単眼参照解像度 (X5 では 10752/2 = 5376).
        combined_w = lenses[0].ref_image_width
        single_lens_native_w = combined_w // 2

        # 抽出画像の実解像度 (paired なので lens0/lens1 で同じ).
        target_w = int(frames_mf["width"])
        target_h = int(frames_mf["height"])

        intrs = [
            projection.lens_to_intrinsics(
                lens,
                lens_index=i,
                single_lens_native_width=single_lens_native_w,
                target_width=target_w,
                target_height=target_h,
            )
            for i, lens in enumerate(lenses)
        ]

        views = projection.cubemap_views(size=ctx.params["size"], fov_deg=ctx.params["fov_deg"])

        # 各 view について, 「rig -> lens A/B」の回転を lens_local_rotation で作る.
        lens_rotations = [rendering.lens_local_rotation(l) for l in lenses]

        # frame ごとに rig を書き出す. 大きなデータになるので max_frames で絞れる.
        frames = frames_mf["frames"]
        if ctx.params["max_frames"] > 0:
            frames = frames[: ctx.params["max_frames"]]
        n = len(frames)
        ctx.progress.info(f"reproject {n} frames x 6 views x 2 lenses = {n*12} renders", progress=0.05)

        rig_records = []
        outputs: list[FileRef] = []

        for fi, fr in enumerate(frames):
            frame_dir = ctx.stage_out_dir / f"frame_{fr['index']:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_rec: dict = {
                "index": fr["index"],
                "source_frame": fr.get("source_frame"),
                "timestamp_sec": fr.get("timestamp_sec"),
                "views": [],
            }

            for view in views:
                for li, (intr, R_lens) in enumerate(zip(intrs, lens_rotations, strict=True)):
                    src_path = ctx.project_dir / fr[f"lens{li}"]
                    if not src_path.exists():
                        raise RuntimeError(f"missing extract frame: {src_path}")
                    img, stats = rendering.render_pinhole(
                        src_path, view, intr, extra_rotation=R_lens
                    )
                    out_path = frame_dir / f"{view.name}_lens{li}.jpg"
                    rendering.write_jpeg(img, out_path, quality=92)
                    outputs.append(
                        FileRef(
                            path=_final_relpath(out_path, ctx),
                            size=out_path.stat().st_size,
                            sha256="",  # 大量ファイルなので個別 hash 省略.
                            mime="image/jpeg",
                        )
                    )
                    frame_rec["views"].append(
                        {
                            "view": view.name,
                            "lens": li,
                            "path": _final_relpath(out_path, ctx),
                            "valid_ratio": stats.valid_ratio,
                        }
                    )

            rig_records.append(frame_rec)
            ctx.progress.info(
                f"reproject frame {fi + 1}/{n}", progress=0.05 + 0.9 * ((fi + 1) / n)
            )

        rig_manifest = {
            "kind": "insv_pinhole_cubemap",
            "view_count": len(views),
            "views": [
                {"name": v.name, "fov_deg": v.fov_deg, "size": v.width, "yaw_deg": v.yaw_deg, "pitch_deg": v.pitch_deg}
                for v in views
            ],
            "lens_count": len(lenses),
            # 各レンズの光学中心オフセット (offset_v3 の tx/ty/tz, 単位 m). rig 拘束の
            # cam_from_rig 計算に使う. lens0 は原点, lens1 は物理ベースライン分ずれる.
            "lenses": [
                {"index": i, "tx": l.tx, "ty": l.ty, "tz": l.tz}
                for i, l in enumerate(lenses)
            ],
            "frames": rig_records,
        }
        rig_path = ctx.stage_out_dir / "manifest_rig.json"
        rig_path.write_text(json.dumps(rig_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(
            FileRef(
                path=_final_relpath(rig_path, ctx),
                size=rig_path.stat().st_size,
                sha256=sha256_file(rig_path),
                mime="application/json",
            )
        )

        manifest.outputs = outputs
        ctx.progress.info("reproject_views done", progress=1.0)
        return manifest


def _lens_from_dict(d: dict) -> calib.MeiLensCalibration:
    return calib.MeiLensCalibration(**d)


def _final_relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
