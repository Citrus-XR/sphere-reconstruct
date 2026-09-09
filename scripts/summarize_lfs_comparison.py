"""同じ native training view の見た目と throughput を比較する。精度の ground truth ではない。"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import numpy as np
from analyze_lfstudio_ply import analyze
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


def write_review_page(result, output):
    headers = "".join(f"<th>{html.escape(run['run'])}</th>" for run in result["runs"])
    summaries = []
    for i, run in enumerate(result["runs"]):
        perf = run["perf"]
        score = np.mean([row["scores"][i]["masked_psnr"] for row in result["images"]])
        numbers = ([f"{perf['wall_seconds'] / 60:.2f}", f"{perf['steady_ms_per_iter']:.2f}",
                    f"{perf['peak_cuda_used_bytes'] / 2**20:.1f}", f"{perf['last_live_splats']:,}"]
                   if perf is not None else ["running"] * 4)
        summaries.append([*numbers, f"{score:.2f}"])
    metrics = ["完走時間 (min)", "Steady ms / iteration", "Peak CUDA (MiB)", "最終 Gaussian 数",
               f"{len(result['images'])} 視点平均 PSNR (dB)"]
    table = "".join(f"<tr><th>{label}</th>" + "".join(f"<td>{row[i]}</td>" for row in summaries) + "</tr>"
                    for i, label in enumerate(metrics))
    images = []
    for row in result["images"]:
        filename = html.escape(row["contact_sheet"], quote=True)
        scores = ", ".join(f"{score['masked_psnr']:.2f}" for score in row["scores"])
        images.append(f"<article><h2>{html.escape(row['image'])}</h2><p>PSNR（表の列順）: {scores}</p>"
                      f"<a href='{filename}'><img loading='lazy' src='{filename}'></a>"
                      f"<p><a href='crops_{filename}'>元 pixel の crop を開く</a></p></article>")
    page = ("<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>LFStudio cleanup 比較</title><style>body{background:#181b20;color:#e4e8ee;font:16px system-ui;"
            "margin:24px}a{color:#a7d3ff}img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:10px;"
            "border:1px solid #49515c;text-align:left}th{overflow-wrap:anywhere}article{margin:32px 0}h2{font-size:18px}"
            "</style><h1>LFStudio cleanup 比較</h1>"
            f"<p>Iteration {result['iteration']:,}。元解像度の native fisheye、GUT、既存 Training mask。"
            "元 RGB・mask・各 run の順に並ぶ。画像をクリックすると全サイズで開く。</p>"
            "<p>PSNR は JPEG timelapse の Training mask 内で算出した training-view fidelity。"
            "Held-out accuracy や浮遊点の正解判定ではない。視点ごとの mask 内 pixel 数は同一で、"
            "RGB / mask の file identity または hash を照合済み。MRNF には乱数差がある。</p>"
            f"<table><tr><th>指標</th>{headers}</tr>{table}</table>"
            + "".join(images) + "<p><a href='comparison.json'>詳細な数値と検証結果</a></p></html>")
    (output / "index.html").write_text(page, encoding="utf-8")


def execute(dataset, runs, output, iteration):
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run in runs:
        gpu = [json.loads(line) for line in (run / "gpu.jsonl").read_text(encoding="utf-8").splitlines()]
        memories = [int(row["gpu"].split(",")[0]) for row in gpu if "gpu" in row]
        perf_path = run / "training" / "perf_bench.json"
        perf = json.loads(perf_path.read_text(encoding="utf-8")) if perf_path.exists() else None
        final_ply = run / "training" / f"splat_{perf['total_iters']}.ply" if perf is not None else None
        summaries.append({"run": run.name, "status": json.loads((run / "status.json").read_text(encoding="utf-8")),
                          "sampled_device_peak_mib": max(memories) if memories else None,
                          "telemetry_errors": [row for row in gpu if "telemetry_error" in row],
                          "progress": read_progress(run / "stdout.log"),
                          "perf": perf,
                          "gaussian_geometry": analyze(final_ply) if final_ply is not None and final_ply.exists() else None})
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
            w, h = ground_truth.size
            crop_metrics = []
            for cx, cy in [(w // 2, h // 2), (w // 2, h * 3 // 4), (w * 3 // 4, h // 2)]:
                bounds = (slice(cy - 320, cy + 320), slice(cx - 320, cx + 320))
                crop_valid = valid[bounds]
                crop_mse = float(error[bounds][crop_valid].mean()) if crop_valid.any() else None
                crop_metrics.append({"center": [cx, cy], "valid_pixels": int(crop_valid.sum()),
                                     "masked_mse": crop_mse,
                                     "masked_psnr": float(-10 * np.log10(crop_mse)) if crop_mse is not None else None})
            scores.append({"run": run.name, "masked_mse": mse, "masked_psnr": float(-10 * np.log10(mse)),
                           "valid_pixels": int(valid.sum()), "crops": crop_metrics})
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
    write_review_page(result, output)
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
