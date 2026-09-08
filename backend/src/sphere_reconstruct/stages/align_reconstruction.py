"""IMU 重力で sparse model 全体を LFStudio dataset 座標へ整列するステージ."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..colmap import gravity_align, web_preview
from ..colmap import model as colmap_model
from ..colmap import runner as colmap_runner
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import SourceAdapter
from ..infrastructure.filesystem import sha256_file
from ..insta360 import imu
from ..pipeline import prepared_images
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest
from ..settings import get_settings
from .colmap_progress import hidden_log


@register
class AlignReconstruction(Stage):
    name = StageName.ALIGN_RECONSTRUCTION
    impl_version = "2.2"

    def normalize_params(self, raw: dict) -> dict:
        method = str(raw.get("method", "auto")).lower()
        if method not in {"auto", "imu", "none"}:
            raise ValueError(f"unsupported alignment method: {method}")
        return {
            "method": method,
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "reconstruct.json",
            ctx.project_dir / "manifests" / "extract_frames.json",
            ctx.project_dir / "inspect_source" / "sources.json",
            prepared_images.catalog_path(ctx.project_dir),
        ]
        candidates += list((ctx.project_dir / "reconstruct" / "sparse" / "0").glob("*"))
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.is_file()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        input_model = ctx.project_dir / "reconstruct" / "sparse" / "0"
        if not (input_model / "cameras.bin").exists():
            raise RuntimeError("reconstruct must run before alignment")
        output_model = ctx.stage_out_dir / "sparse" / "0"
        ctx.progress.info(
            "estimating reconstruction alignment",
            progress=0.0,
            key="log.alignment_start",
        )
        alignment = self._estimate(ctx, input_model, ProgressSpan(ctx.progress, 0.0, 0.55))

        rotation = alignment.pop("rotation")
        ctx.progress.info(
            "applying reconstruction transform",
            progress=0.55,
            key="log.alignment_transform",
        )
        if alignment["applied"]:
            quaternion = gravity_align._R_to_quat(rotation)
            transform_path = ctx.stage_out_dir / "transform.txt"
            transform_path.write_text(
                "1 " + " ".join(f"{value:.17g}" for value in quaternion) + " 0 0 0\n",
                encoding="utf-8",
            )
            colmap_bin = colmap_runner.resolve_colmap_bin(get_settings().binaries.colmap or None)
            colmap_runner.model_transformer(
                colmap_bin,
                input_path=input_model,
                output_path=output_model,
                transform_path=transform_path,
                log_path=ctx.stage_out_dir / "model_transformer.log",
                on_line=hidden_log(ctx, "alignment"),
            )
        else:
            shutil.copytree(input_model, output_model)

        ctx.progress.tick(
            0.75,
            message="alignment transform complete",
            key="log.alignment_transform_done",
        )
        aligned_reconstruction = colmap_model.read_model(output_model)
        aligned_summary = aligned_reconstruction.summary()
        alignment["model_summary"] = aligned_summary
        alignment_path = ctx.stage_out_dir / "alignment.json"
        alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")

        # Dataset -Y up を Three.js +Y up へ明示変換した preview を同じステージで生成する.
        ctx.progress.info(
            "building aligned web preview",
            progress=0.82,
            key="log.alignment_preview",
        )
        gravity_align.apply_alignment(aligned_reconstruction, gravity_align.DATASET_TO_VIEWER)
        preview_dir = ctx.stage_out_dir / "preview"
        preview = web_preview.write_web_preview(
            aligned_reconstruction,
            preview_dir,
            max_points=ctx.params["max_preview_points"],
        )
        preview_json_path = preview_dir / "reconstruction.json"
        preview_json = json.loads(preview_json_path.read_text(encoding="utf-8"))
        preview_json["alignment"] = alignment
        preview_json["coordinate_system"] = {
            "dataset_up": "-Y",
            "viewer_up": "+Y",
            "dataset_to_viewer": [1, 0, 0, 0, -1, 0, 0, 0, -1],
        }
        preview_json_path.write_text(json.dumps(preview_json, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs = [_file_ref(alignment_path, ctx), _file_ref(preview_json_path, ctx)]
        outputs.append(_file_ref(preview_dir / "points.bin", ctx))
        for path in output_model.iterdir():
            if path.is_file():
                outputs.append(_file_ref(path, ctx))
        manifest.outputs = outputs
        manifest.extra = {
            **alignment,
            "preview_points": preview.num_points_written,
        }
        ctx.progress.info(
            f"alignment done: applied={alignment['applied']}, spread={alignment.get('spread_deg', 0)}deg",
            progress=0.99,
            key="log.alignment_done",
            args={
                "applied": alignment["applied"],
                "spread": alignment.get("spread_deg", 0),
            },
        )
        return manifest

    def _estimate(
        self, ctx: StageContext, input_model: Path, progress_span: ProgressSpan
    ) -> dict:
        reconstruction = colmap_model.read_model(input_model)
        primary = ctx.primary_source
        reference_prefix = f"sources/{primary.id}/" if primary else None
        diameter = gravity_align.reference_trajectory_diameter(reconstruction, reference_prefix)
        progress_span.tick(
            0.05,
            message="alignment model loaded",
            key="log.alignment_model_loaded",
        )
        if ctx.params["method"] == "none":
            progress_span.tick(1.0, message="alignment disabled", key="log.alignment_estimate_done")
            return {
                "applied": False,
                "method": "none",
                "reason": "disabled",
                "source_trajectory_diameter": diameter,
                "rotation": gravity_align._rotation_aligning(
                    gravity_align.TARGET_UP, gravity_align.TARGET_UP
                ),
            }
        if primary is None or primary.adapter != SourceAdapter.INSTA360:
            if ctx.params["method"] == "imu":
                raise RuntimeError("IMU alignment requires an INSV source")
            progress_span.tick(
                1.0,
                message="source has no IMU alignment",
                key="log.alignment_estimate_done",
            )
            return {
                "applied": False,
                "method": "none",
                "reason": "source_has_no_imu",
                "source_trajectory_diameter": diameter,
                "rotation": gravity_align._rotation_aligning(
                    gravity_align.TARGET_UP, gravity_align.TARGET_UP
                ),
            }

        frames_path = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        frames = [
            frame
            for frame in json.loads(frames_path.read_text(encoding="utf-8"))["frames"]
            if frame["source_id"] == primary.id
        ]
        frame_times = {
            int(frame["index"]): float(frame["timestamp_sec"])
            for frame in frames
            if frame.get("timestamp_sec") is not None
        }
        progress_span.tick(
            0.1,
            message="reading IMU recording",
            key="log.alignment_read_imu",
        )
        recording = imu.read_imu_recording(primary.path)
        if recording is None:
            raise RuntimeError("INSV contains no IMU samples")
        progress_span.tick(
            0.25,
            message=f"IMU recording loaded: {len(recording.samples)} samples",
            key="log.alignment_imu_loaded",
            args={"samples": len(recording.samples)},
        )

        def alignment_progress(phase: str, current: int, total: int) -> None:
            phase_span = (
                progress_span.child(0.25, 0.8)
                if phase == "coarse"
                else progress_span.child(0.8, 1.0)
            )
            phase_span.tick(
                current / max(1, total),
                message=f"gravity offset {phase} {current}/{total}",
                key="log.alignment_offset_progress",
                args={"phase": phase, "cur": current, "tot": total},
            )

        rotation, diagnostics = gravity_align.compute_timed_align_rotation(
            reconstruction,
            frame_times,
            recording.samples,
            imu_timestamps_sec=recording.timestamps_sec,
            image_prefix=f"sources/{primary.id}/",
            progress=alignment_progress,
        )
        if rotation is None:
            raise RuntimeError(f"IMU gravity alignment failed: {diagnostics}")
        return {
            "applied": True,
            "method": "imu_timed_consensus",
            "camera_type": recording.camera_type,
            "target_dataset_up": [0, -1, 0],
            "source_trajectory_diameter": diameter,
            "reference_image_prefix": f"sources/{primary.id}/",
            "rotation": rotation,
            **diagnostics,
        }


def _file_ref(path: Path, ctx: StageContext) -> FileRef:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    relative = Path(final_name) / path.relative_to(ctx.stage_out_dir)
    return FileRef(
        path=str(relative),
        size=path.stat().st_size,
        sha256=sha256_file(path),
        mime="application/json" if path.suffix == ".json" else "application/octet-stream",
    )
