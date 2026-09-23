"""Build paired LFS datasets with identical primary cameras, images, and seed points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pycolmap
from audit_mixed_pose_geometry import align_to_primary
from experiment_support import file_hash, write_json


def execute(args):
    sys.path.insert(0, str(args.backend_src))
    from sphere_reconstruct.colmap import model as cm
    from sphere_reconstruct.domain.mask_artifact import (
        MaskPurpose,
        load_mask_manifest,
        records_by_name,
    )
    from sphere_reconstruct.stages.export_dataset import _safe_relative_path

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    records = {record["name"]: record for record in catalog["images"]}
    primary = pycolmap.Reconstruction(str(args.primary))
    models = {"baseline": pycolmap.Reconstruction(str(args.baseline)),
              "candidate": pycolmap.Reconstruction(str(args.candidate))}
    primary_model = cm.read_model(args.primary)
    used_primary_cameras = {image.camera_id for image in primary_model.images.values()}
    primary_model.cameras = {key: camera for key, camera in primary_model.cameras.items() if key in used_primary_cameras}
    cohort = cm.read_model(args.cohort)
    if any(records[image.name]["source_id"] != catalog["primary_source_id"] for image in primary_model.images.values()):
        raise ValueError("seed model must be primary-only")
    alignments = {label: align_to_primary(model, primary, records) for label, model in models.items()}
    phone_names = {image.name for image in cohort.images.values()
                   if records[image.name]["source_id"] != catalog["primary_source_id"]}
    by_name = {label: {image.name: image for image in model.images.values() if image.has_pose}
               for label, model in models.items()}
    shared = phone_names.intersection(*(set(images) for images in by_name.values()))
    if not shared:
        raise ValueError("no common supplemental images in the requested cohort")
    spec = json.loads((args.project / "extract_features/input_spec.json").read_text(encoding="utf-8"))
    image_root = args.project / "extract_features" / spec["image_path"]
    masks = records_by_name(load_mask_manifest(args.project, MaskPurpose.TRAINING))
    report = {"strategy": "paired_phone_pose_training_comparison", "alignment": alignments,
              "primary_images": len(primary_model.images), "requested_phone_images": len(phone_names),
              "common_phone_images": len(shared), "omitted_phone_images": sorted(phone_names - shared),
              "primary_seed_points": len(primary_model.points3D),
              "limitations": ["Common cohort comparison does not measure recovery of omitted images.",
                              "Primary geometry and primary camera poses are identical in both datasets.",
                              "Phone poses and their estimated calibration are the treatment variables.",
                              "Training is stochastic; a single run per condition is diagnostic evidence."],
              "datasets": {}}
    for label, reconstruction in models.items():
        dataset = args.output / label
        sparse = dataset / "sparse/0"
        sparse.mkdir(parents=True)
        images = dict(primary_model.images)
        cameras = dict(primary_model.cameras)
        for name in sorted(shared):
            image = by_name[label][name]
            if image.image_id in images or image.camera_id in primary_model.cameras:
                raise ValueError("supplemental image/camera identity conflicts with primary model")
            calibration = reconstruction.cameras[image.camera_id]
            pose = image.cam_from_world()
            xyzw = pose.rotation.quat.tolist()
            cameras[image.camera_id] = cm.Camera(image.camera_id, calibration.model_name,
                                                calibration.width, calibration.height,
                                                calibration.params.tolist(), int(calibration.model))
            images[image.image_id] = cm.Image(image.image_id, (xyzw[3], *xyzw[:3]),
                                             tuple(pose.translation), image.camera_id, name, [])
        cm.write_cameras_bin(sparse / "cameras.bin", cameras)
        cm.write_images_bin(sparse / "images.bin", images)
        cm.write_points3D_bin(sparse / "points3D.bin", primary_model.points3D)
        # A classic COLMAP three-file model is complete without frame/rig metadata.
        # Reusing either input's frames would attach stale phone poses to this model.
        missing_masks = []
        for image in images.values():
            relative = _safe_relative_path(image.name)
            if relative is None:
                raise ValueError(f"unsafe image name: {image.name}")
            target = dataset / "images" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(image_root / relative, target)
            mask = args.project / masks[image.name]["path"] if image.name in masks else None
            if mask is None or not mask.is_file():
                missing_masks.append(image.name)
            else:
                destination = dataset / "masks" / (image.name + ".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(mask, destination)
        if missing_masks:
            raise ValueError(f"missing training masks: {missing_masks[:5]} ({len(missing_masks)} total)")
        loaded = pycolmap.Reconstruction(str(sparse))
        if len(loaded.reg_image_ids()) != len(images) or len(loaded.points3D) != len(primary_model.points3D):
            raise RuntimeError("comparison COLMAP round trip changed image/point counts")
        manifest = {"format": "sphere-reconstruct-export", "version": 3,
                    "images": len(images), "points3D": len(primary_model.points3D), "masks": len(images),
                    "mask_source": "training", "model_source": report["strategy"],
                    "camera_models": sorted({camera.model for camera in cameras.values()}),
                    "training_crop": {"enabled": False}, "condition": label,
                    "seed_points_sha256": file_hash(sparse / "points3D.bin")}
        write_json(dataset / "export_manifest.json", manifest)
        report["datasets"][label] = manifest
    if report["datasets"]["baseline"]["seed_points_sha256"] != report["datasets"]["candidate"]["seed_points_sha256"]:
        raise RuntimeError("paired datasets do not share identical seed points")
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "primary", "baseline", "candidate", "cohort", "catalog", "backend-src", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    execute(args)


if __name__ == "__main__":
    main()
