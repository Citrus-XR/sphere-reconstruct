"""Compare paired LFS metrics and the same preselected image viewpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_metrics(training: Path) -> list[dict]:
    with (training / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no evaluation metrics: {training}")
    return rows


def execute(args):
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    probe_names = cohort["probes"]
    comparison = cohort["comparison"]
    if comparison["datasets"]["baseline"]["seed_points_sha256"] != comparison["datasets"]["candidate"]["seed_points_sha256"]:
        raise ValueError("paired training must use identical initial points")
    training = {label: getattr(args, label) / "training" for label in ["baseline", "candidate"]}
    report = {
        "official_metrics": {label: read_metrics(path) for label, path in training.items()},
        "comparison": comparison,
        "omitted_probes": cohort["omitted_probes"],
        "views": [],
        "limitations": [
            "JPEG view PSNR is a diagnostic, distinct from official heldout evaluation.",
            "Reference RGB and masks are resized to rendered dimensions for this diagnostic.",
            "The cohort may contain both training and evaluation views.",
            "Single stochastic training runs cannot establish statistical significance.",
        ],
    }
    if not probe_names:
        raise ValueError("no registered probe images remain")
    tile_width, tile_height, label_height = 512, 384, 36
    canvas = Image.new("RGB", (tile_width * 3, len(probe_names) * (tile_height + label_height)), "#222222")
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(probe_names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError(f"unsafe probe image name: {name}")
        source = args.datasets / "baseline"
        with Image.open(source / "images" / relative) as image:
            original = image.convert("RGB")
        with Image.open(source / "masks" / (name + ".png")) as image:
            mask = image.convert("L")
        y = row * (tile_height + label_height)
        draw.text((8, y + 8), f"Original: {relative.name} ({relative.parent.name})", fill="white")
        canvas.paste(ImageOps.contain(original, (tile_width, tile_height)), (0, y + label_height))
        measurement = {"name": name, "conditions": {}}
        rendered_size = None
        for column, (label, path) in enumerate(training.items(), 1):
            rendered_path = path / "timelapse" / relative.with_suffix("") / f"{args.iteration:06d}.jpg"
            with Image.open(rendered_path) as image:
                rendered = image.convert("RGB")
            if rendered_size is not None and rendered.size != rendered_size:
                raise ValueError(f"paired renders have different dimensions: {name}")
            rendered_size = rendered.size
            resized = original.resize(rendered.size, Image.Resampling.BILINEAR)
            valid = np.asarray(mask.resize(rendered.size, Image.Resampling.NEAREST)) >= 128
            if not valid.any():
                raise ValueError(f"probe image has no valid masked pixels: {name}")
            residual = (np.asarray(resized, dtype=float) - np.asarray(rendered, dtype=float)) / 255
            mse = float(np.mean(residual[valid] ** 2))
            psnr = float(-10 * np.log10(mse)) if mse > 0 else None
            measurement["conditions"][label] = {
                "diagnostic_jpeg_rgb_psnr_db": psnr, "mse": mse,
                "valid_pixels": int(valid.sum()), "dimensions": rendered.size,
            }
            score = f"{psnr:.2f} dB" if psnr is not None else "exact match"
            draw.text((column * tile_width + 8, y + 8), f"{label}: {score}", fill="white")
            canvas.paste(ImageOps.contain(rendered, (tile_width, tile_height)),
                         (column * tile_width, y + label_height))
        report["views"].append(measurement)
    args.output.mkdir(parents=True, exist_ok=False)
    canvas.save(args.output / "contact.jpg", quality=94)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"official_metrics": report["official_metrics"], "views": report["views"],
                      "omitted_probes": report["omitted_probes"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["datasets", "baseline", "candidate", "cohort", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    if args.iteration < 1:
        parser.error("iteration must be positive")
    execute(args)


if __name__ == "__main__":
    main()
