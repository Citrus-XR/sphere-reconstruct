"""LFStudio CLI を隔離 output で実行し、終了状態と GPU 使用量を記録する。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark_fixed_pose import fingerprint, write_json


def execute(args):
    command = [str(args.executable), "-d", str(args.dataset), "-o", str(args.output / "training"),
               "--headless", "--train", "--safe-mode", "--gut", "--resize_factor", "1", "--max-width", "0",
               "--mask-mode", "segment", "--min-track-length", "0", "--strategy", "mrnf",
               "--iter", str(args.iterations), "--max-cap", str(args.max_cap),
               "--export", "ply", "--perf-bench", "--log-file", str(args.output / "training.log")]
    command.extend(args.lfs_args)
    write_json(args.output / "invocation.json", {"command": command,
               "executable_sha256": fingerprint(args.executable),
               "dataset_manifest": json.loads((args.dataset / "export_manifest.json").read_text(encoding="utf-8"))})
    started = time.time()
    with (args.output / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (args.output / "stderr.log").open("w", encoding="utf-8") as stderr, \
            (args.output / "gpu.jsonl").open("w", encoding="utf-8") as samples:
        process = subprocess.Popen(command, cwd=args.executable.parent, stdout=stdout, stderr=stderr)
        try:
            while process.poll() is None:
                state = {"status": "running", "pid": process.pid, "elapsed_seconds": time.time() - started}
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                             "--format=csv,noheader,nounits"], capture_output=True,
                                            text=True, timeout=30, check=True)
                    state["gpu"] = result.stdout.strip()
                except (OSError, subprocess.SubprocessError) as error:
                    state["telemetry_error"] = repr(error)
                    print(f"GPU telemetry failed; training supervision continues: {error}", file=sys.stderr, flush=True)
                samples.write(json.dumps(state) + "\n")
                samples.flush()
                write_json(args.output / "status.json", state)
                if args.timeout_seconds and state["elapsed_seconds"] > args.timeout_seconds:
                    raise TimeoutError(f"training exceeded explicit timeout {args.timeout_seconds}s")
                time.sleep(10)
            code = process.wait()
            if code == 0:
                expected = [args.output / "training" / name for name in
                            ["project.licht", f"splat_{args.iterations}.ply", "perf_bench.json"]]
                missing = [path.name for path in expected if not path.is_file() or path.stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"LFStudio exited without required training outputs: {missing}")
            state = {"status": "succeeded" if code == 0 else "failed", "exit_code": code,
                     "elapsed_seconds": time.time() - started}
            write_json(args.output / "status.json", state)
            if code:
                raise RuntimeError(f"LFStudio exited with code {code}; inspect training/stdout/stderr logs")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-cap", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--wait-for", type=Path)
    args, args.lfs_args = parser.parse_known_args()
    args.executable, args.dataset, args.output = args.executable.resolve(), args.dataset.resolve(), args.output.resolve()
    if args.wait_for is not None:
        args.wait_for = args.wait_for.resolve()
        if not args.wait_for.is_file() or args.wait_for.is_relative_to(args.output):
            raise ValueError("predecessor status file must exist outside this output")
    if args.iterations <= 0 or args.max_cap <= 0 or args.timeout_seconds < 0:
        raise ValueError("iterations/cap must be positive and timeout nonnegative")
    if not args.executable.is_file() or not (args.dataset / "export_manifest.json").is_file():
        raise ValueError("executable and exported dataset manifest must exist")
    if args.output.is_relative_to(args.dataset):
        raise ValueError("training output must be outside the dataset")
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[argument for argument in sys.argv[1:] if argument != "--detach"]]
        if args.output.exists():
            raise FileExistsError(args.output)
        if os.name == "nt":
            # SSH session の job object 終了から training supervisor を分離する。
            literal = subprocess.list2cmdline(command).replace("'", "''")
            result = subprocess.run(["powershell", "-NoProfile", "-Command",
                                     ("Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
                                      f"-Arguments @{{CommandLine='{literal}'}} | ConvertTo-Json")],
                                    check=True, capture_output=True, text=True)
            launch = json.loads(result.stdout)
            if launch["ReturnValue"] != 0:
                raise RuntimeError(f"Win32_Process.Create failed: {launch}")
            pid = launch["ProcessId"]
        else:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            pid = process.pid
        print(json.dumps({"launcher_pid": pid, "output": str(args.output)}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.wait_for is not None:
            while True:
                predecessor = json.loads(args.wait_for.read_text(encoding="utf-8"))
                if predecessor["status"] == "succeeded":
                    break
                if predecessor["status"] == "failed":
                    raise RuntimeError(f"predecessor failed: {predecessor}")
                write_json(args.output / "status.json", {"status": "waiting", "predecessor": str(args.wait_for)})
                time.sleep(15)
        execute(args)
    except Exception as error:
        status_path = args.output / "status.json"
        state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        write_json(status_path, {**state, "status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
