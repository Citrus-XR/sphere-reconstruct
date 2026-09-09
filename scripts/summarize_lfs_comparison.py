"""同じ native training view の見た目と throughput を比較する。精度の ground truth ではない。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from benchmark_fixed_pose import fingerprint, write_json
from PIL import Image, ImageDraw


def read_progress(path):
    pattern = re.compile(r"\[(?:(\d+)h:)?(\d+)m:(\d+)s<[^\]]+\] (\d+)/(\d+) \| Loss: ([\d.eE+-]+) \| Splats: (\d+)")
    rows = []
    for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
        hours, minutes, seconds, iteration, total, loss, splats = match.groups()
        rows.append({"iteration": int(iteration), "total": int(total), "loss": float(loss), "splats": int(splats),
                     "elapsed_seconds": (int(hours) if hours else 0) * 3600 + int(minutes) * 60 + int(seconds)})
    return rows


def execute(dataset, runs, output, iteration):
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run in runs:
        gpu = [json.loads(line) for line in (run / "gpu.jsonl").read_text(encoding="utf-8").splitlines()]
        memories = [int(row["gpu"].split(",")[0]) for row in gpu if "gpu" in row]
        perf_path = run / "training" / "perf_bench.json"
        summaries.append({"run": run.name, "status": json.loads((run / "status.json").read_text(encoding="utf-8")),
                          "sampled_device_peak_mib": max(memories) if memories else None,
                          "telemetry_errors": [row for row in gpu if "telemetry_error" in row],
                          "progress": read_progress(run / "stdout.log"),
                          "perf": json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.exists() else None})
    names = sorted({path.parent.relative_to(run / "training" / "timelapse").as_posix()
                    for run in runs for path in (run / "training" / "timelapse").rglob(f"{iteration:06d}.jpg")})
    if not names:
        raise ValueError(f"no timelapse images at iteration {iteration}")
    rows = []
    for name in names:
        image_name = name + ".png"
        with Image.open(dataset / "images" / image_name) as image:
            ground_truth = image.convert("RGB")
        mask_path = dataset / "masks" / (image_name + ".png")
        if not mask_path.is_file():
            mask_path = dataset / "masks" / image_name
        with Image.open(mask_path) as image:
            mask_image = image.convert("L")
        valid = np.asarray(mask_image) >= 128
        if not valid.any() or mask_image.size != ground_truth.size:
            raise ValueError(f"mask has no valid pixels or mismatched dimensions: {mask_path}")
        truth = np.asarray(ground_truth, dtype=float) / 255
        panels = [("original RGB", ground_truth), ("training mask", mask_image.convert("RGB"))]
        scores = []
        for run in runs:
            invocation = json.loads((run / "invocation.json").read_text(encoding="utf-8"))["command"]
            run_dataset = Path(invocation[invocation.index("-d") + 1])
            for reference in (dataset / "images" / image_name, mask_path):
                counterpart = run_dataset / reference.relative_to(dataset)
                if not os.path.samefile(reference, counterpart) and fingerprint(reference) != fingerprint(counterpart):
                    raise ValueError(f"comparison inputs differ: {reference} != {counterpart}")
            path = run / "training" / "timelapse" / name / f"{iteration:06d}.jpg"
            if not path.is_file():
                raise ValueError(f"missing matching render: {path}")
            with Image.open(path) as image:
                render = image.convert("RGB")
            if render.size != ground_truth.size:
                raise ValueError(f"render was resized: {render.size} != {ground_truth.size}")
            error = np.mean(np.square(np.asarray(render, dtype=float) / 255 - truth), axis=2)
            mse = float(error[valid].mean())
            scores.append({"run": run.name, "masked_mse": mse, "masked_psnr": float(-10 * np.log10(mse)),
                           "valid_pixels": int(valid.sum())})
            panels.append((run.name.replace("testo2-lfs-", "").replace("-20260909", ""), render))
        width, height = 640, 676
        sheet = Image.new("RGB", (width * len(panels), height), "#1b1e23")
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(panels):
            draw.text((i * width + 8, 8), label, fill="white")
            sheet.paste(image.resize((640, 640), Image.Resampling.LANCZOS), (i * width, 36))
        filename = name.replace("/", "_") + ".jpg"
        sheet.save(output / filename, quality=94)
        # 元 pixel の crop も保存し、縮小 contact sheet だけで細部を評価しない。
        crop_sheet = Image.new("RGB", (640 * len(panels), 3 * 676), "#1b1e23")
        draw = ImageDraw.Draw(crop_sheet)
        w, h = ground_truth.size
        for j, (cx, cy) in enumerate([(w // 2, h // 2), (w // 2, h * 3 // 4), (w * 3 // 4, h // 2)]):
            for i, (label, image) in enumerate(panels):
                draw.text((i * 640 + 8, j * 676 + 8), f"{label} crop {cx},{cy}", fill="white")
                crop_sheet.paste(image.crop((cx - 320, cy - 320, cx + 320, cy + 320)),
                                 (i * 640, j * 676 + 36))
        crop_sheet.save(output / ("crops_" + filename), quality=96)
        rows.append({"image": image_name, "contact_sheet": filename, "scores": scores})
    result = {"iteration": iteration, "evaluation_type": "training-view fidelity from JPEG timelapse; not held-out geometry validation",
              "runs": summaries, "images": rows}
    write_json(output / "comparison.json", result)
    print(json.dumps({"images": len(rows), "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.dataset, args.runs, args.output, args.iteration)


if __name__ == "__main__":
    main()
