"""Canonical image catalog から mixed-camera COLMAP workspace を構築する。"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image as PilImage

from ..domain.mask_artifact import MaskPurpose, load_mask_manifest, mask_manifest_path, records_by_name
from ..imaging import catalog_validity
from ..pipeline import prepared_images


@dataclass(frozen=True)
class FeatureBatch:
    id: str
    source_id: str
    camera_model: str
    camera_params: list[float]
    single_camera: bool
    single_camera_per_folder: bool
    image_list_path: str
    image_count: int


@dataclass(frozen=True)
class InputSpec:
    version: int
    reconstruction_mode: str
    image_count: int
    source_count: int
    primary_source_id: str
    primary_image_names: list[str]
    sources: list[dict]
    images: list[dict]
    feature_batches: list[FeatureBatch]
    image_path: str
    mask_path: str | None
    feature_masks_enabled: bool
    rig_config_path: str | None
    refine_intrinsics: bool
    refine_rig: bool
    multiple_models: bool

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> InputSpec:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 3:
            raise ValueError(f"unsupported input spec version: {data.get('version')}")
        data["feature_batches"] = [FeatureBatch(**batch) for batch in data["feature_batches"]]
        return cls(**data)


def build(
    project_dir: Path,
    output_dir: Path,
    use_feature_masks: bool,
    progress: Callable[[int, int], None] | None = None,
) -> InputSpec:
    catalog = prepared_images.load_catalog(project_dir)
    generated_masks: dict[str, Path] = {}
    if use_feature_masks:
        path = mask_manifest_path(project_dir, MaskPurpose.FEATURE)
        if not path.is_file():
            raise RuntimeError("generate_feature_masks must run before masked feature extraction")
        manifest = load_mask_manifest(project_dir, MaskPurpose.FEATURE)
        generated_masks = {
            name: project_dir / record["path"] for name, record in records_by_name(manifest).items()
        }

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    physical_mask_templates: dict[str, Path] = {}
    materialize_masks = use_feature_masks or any(
        image.get("valid_mask_path") is not None
        or image["valid_region"]["kind"] != "full"
        for image in catalog["images"]
    )
    for image_number, image in enumerate(catalog["images"], 1):
        source = project_dir / image["path"]
        expected_size = (int(image["width"]), int(image["height"]))
        try:
            with PilImage.open(source) as decoded:
                actual_size = decoded.size
        except (OSError, ValueError) as error:
            raise RuntimeError(f"cannot read prepared image: {source}") from error
        if actual_size != expected_size:
            raise RuntimeError(
                f"prepared image dimensions changed for {image['name']}: "
                f"{actual_size[0]}x{actual_size[1]} != {expected_size[0]}x{expected_size[1]}"
            )
        _link_or_copy(source, images_dir / image["name"])
        if materialize_masks:
            destination = masks_dir / f"{image['name']}.png"
            generated = generated_masks.get(image["name"])
            if generated is not None:
                try:
                    with PilImage.open(generated) as decoded:
                        mask_size = decoded.size
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"cannot read feature mask: {generated}") from error
                if mask_size != expected_size:
                    raise RuntimeError(
                        f"feature mask dimensions changed for {image['name']}: "
                        f"{mask_size[0]}x{mask_size[1]} != {expected_size[0]}x{expected_size[1]}"
                    )
                _link_or_copy(generated, destination)
            elif use_feature_masks:
                raise RuntimeError(f"feature mask missing for prepared image: {image['name']}")
            else:
                key = catalog_validity.cache_key(image, *expected_size)
                template = physical_mask_templates.get(key)
                if template is None:
                    _write_valid_region_mask(project_dir, image, destination)
                    physical_mask_templates[key] = destination
                else:
                    _link_or_copy(template, destination)
        if progress is not None:
            progress(image_number, len(catalog["images"]))

    list_dir = output_dir / "image_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for index, group in enumerate(catalog["camera_groups"]):
        names = sorted(group["image_names"])
        list_path = list_dir / f"batch_{index:03d}.txt"
        list_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        batches.append(
            FeatureBatch(
                id=group["id"],
                source_id=group["source_id"],
                camera_model=group["camera_model"],
                camera_params=[float(value) for value in group["camera_params"]],
                single_camera=bool(group["single_camera"]),
                single_camera_per_folder=bool(group["single_camera_per_folder"]),
                image_list_path=str(list_path.relative_to(output_dir)),
                image_count=len(names),
            )
        )

    rig_config_path = None
    if catalog.get("rig_config_path"):
        source = project_dir / prepared_images.DIRECTORY / catalog["rig_config_path"]
        destination = output_dir / "rig_config.json"
        shutil.copy2(source, destination)
        rig_config_path = "rig_config.json"
    images = [
        {
            "name": image["name"],
            "source_id": image["source_id"],
            "source_role": image["source_role"],
            "capture_index": image["capture_index"],
            "sensor_id": image["sensor_id"],
        }
        for image in catalog["images"]
    ]
    primary_id = catalog["primary_source_id"]
    spec = InputSpec(
        version=3,
        reconstruction_mode=catalog["reconstruction_mode"],
        image_count=len(images),
        source_count=len(catalog["sources"]),
        primary_source_id=primary_id,
        primary_image_names=[image["name"] for image in images if image["source_id"] == primary_id],
        sources=[
            {key: source[key] for key in ("id", "label", "role", "projection")}
            for source in catalog["sources"]
        ],
        images=images,
        feature_batches=batches,
        image_path="images",
        mask_path="masks" if materialize_masks else None,
        feature_masks_enabled=use_feature_masks,
        rig_config_path=rig_config_path,
        refine_intrinsics=any(group["refine_intrinsics"] for group in catalog["camera_groups"]),
        refine_rig=False,
        multiple_models=False,
    )
    spec.write(output_dir / "input_spec.json")
    return spec


def _write_valid_region_mask(project_dir: Path, image: dict, destination: Path) -> None:
    import cv2  # noqa: PLC0415

    width, height = int(image["width"]), int(image["height"])
    mask = catalog_validity.render_mask(project_dir, image, width, height) * 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), mask):
        raise RuntimeError(f"failed to write mask: {destination}")


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
