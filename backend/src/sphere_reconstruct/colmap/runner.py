"""COLMAP CLI 子プロセス駆動.

colmap の各サブコマンド (feature_extractor / sequential_matcher / mapper など) を
子プロセスとして実行し, 標準出力/エラーを行単位で callback へ流す.

キャンセルは呼び出し側がプロセスを terminate すれば良い. 本 runner は 1 コマンド
= 1 プロセスの同期実行を提供し, ログ行を逐次コールバックする.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ColmapError(RuntimeError):
    pass


def resolve_colmap_bin(explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"colmap binary not found: {explicit}")
        return str(p)
    found = shutil.which("colmap")
    if found:
        return found
    record = _auto_installed_colmap_record()
    if record.is_file():
        installed = Path(record.read_text(encoding="utf-8").strip())
        if installed.is_file():
            return str(installed)
    raise FileNotFoundError(
        "colmap not found in PATH or .runtime installer record; "
        "set binaries.colmap in config.toml or use the start script"
    )


def _auto_installed_colmap_record() -> Path:
    return Path(__file__).resolve().parents[4] / ".runtime" / "colmap-path.txt"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    log_tail: list[str]


def run_command(
    colmap_bin: str,
    args: list[str],
    *,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    tail_lines: int = 40,
) -> CommandResult:
    """`colmap <args>` を実行し, 出力を逐次 on_line へ渡す.

    stderr は stdout にマージして 1 本のストリームで扱う. COLMAP は進捗を stderr に
    出すことが多いため.
    """
    full = [colmap_bin, *args]
    log_f = log_path.open("w", encoding="utf-8") if log_path else None
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            full,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_runtime_environment(),
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if log_f:
                log_f.write(raw)
                log_f.flush()
            tail.append(line)
            if len(tail) > tail_lines:
                tail.pop(0)
            if on_line is not None:
                on_line(line)
        proc.wait()
        rc = proc.returncode
    finally:
        if log_f:
            log_f.close()

    if rc != 0:
        raise ColmapError(f"colmap {args[0] if args else '?'} failed (rc={rc}):\n" + "\n".join(tail[-15:]))
    return CommandResult(command=full, returncode=rc, log_tail=tail)


def _runtime_environment() -> dict[str, str]:
    """COLMAP の ONNX CUDA provider が同じ venv の CUDA runtime を見つけられる環境を返す."""
    env = os.environ.copy()
    if os.name == "nt":
        torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        variable = "PATH"
    else:
        candidates = list((Path(sys.prefix) / "lib").glob("python*/site-packages/torch/lib"))
        torch_lib = candidates[0] if candidates else Path()
        variable = "LD_LIBRARY_PATH"
    if torch_lib.is_dir():
        current = env.get(variable, "")
        values = [str(torch_lib), *(value for value in current.split(os.pathsep) if value)]
        env[variable] = os.pathsep.join(dict.fromkeys(values))
    return env


# -- 各サブコマンドの薄いヘルパ ---------------------------------------------------


def feature_extractor(
    colmap_bin: str,
    *,
    database_path: Path,
    image_path: Path,
    camera_model: str = "PINHOLE",
    single_camera: bool = True,
    single_camera_per_folder: bool = False,
    camera_params: str | None = None,
    use_gpu: bool = True,
    feature_type: str = "SIFT",
    mask_path: Path | None = None,
    camera_mask_path: Path | None = None,
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    args = [
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        camera_model,
        "--FeatureExtraction.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureExtraction.type",
        feature_type,
    ]
    if single_camera_per_folder:
        # rig 使用時: 各サブフォルダを独立カメラにする (pinhole rig は 12, native は front/back の 2).
        args += ["--ImageReader.single_camera_per_folder", "1"]
    else:
        args += ["--ImageReader.single_camera", "1" if single_camera else "0"]
    if camera_params is not None:
        # 既知の内部パラメータを与える (PINHOLE なら "fx,fy,cx,cy"). 仮想 pinhole は
        # レンダリング時に厳密な intrinsics で作っているので, 推定させず固定する.
        args += ["--ImageReader.camera_params", camera_params]
    if mask_path is not None:
        # COLMAP mask 規則: mask_path/<image_name>.png. 黒 (0) 画素を無視する.
        args += ["--ImageReader.mask_path", str(mask_path)]
    if camera_mask_path is not None:
        # 全画像に共通の 1 枚マスク (native fisheye の円形有効領域など). 黒画素を無視する.
        args += ["--ImageReader.camera_mask_path", str(camera_mask_path)]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def rig_configurator(
    colmap_bin: str,
    *,
    database_path: Path,
    rig_config_path: Path,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    """既知の rig 相対姿勢を DB に設定する. feature_extractor の後, mapper の前に呼ぶ."""
    args = [
        "rig_configurator",
        "--database_path",
        str(database_path),
        "--rig_config_path",
        str(rig_config_path),
    ]
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def database_creator(colmap_bin: str, *, database_path: Path) -> CommandResult:
    """空の COLMAP DB を現行スキーマで作る (ALIKED 経路で自前書き込みする前段)."""
    return run_command(colmap_bin, ["database_creator", "--database_path", str(database_path)])


def matches_importer(
    colmap_bin: str,
    *,
    database_path: Path,
    match_list_path: Path,
    match_type: str = "raw",
    min_num_inliers: int = 15,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    """テキスト match list を取り込み, 幾何検証して two_view_geometries を埋める.

    match_type="raw": 生の対応 (keypoint index 対) を検証する.
    """
    args = [
        "matches_importer",
        "--database_path",
        str(database_path),
        "--match_list_path",
        str(match_list_path),
        "--match_type",
        match_type,
        "--TwoViewGeometry.min_num_inliers",
        str(min_num_inliers),
    ]
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def sequential_matcher(
    colmap_bin: str,
    *,
    database_path: Path,
    overlap: int = 10,
    loop_detection: bool = False,
    vocab_tree_path: Path | None = None,
    use_gpu: bool = True,
    matching_type: str = "SIFT_BRUTEFORCE",
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    args = [
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--SequentialMatching.overlap",
        str(overlap),
        "--FeatureMatching.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureMatching.type",
        matching_type,
    ]
    if loop_detection and vocab_tree_path is not None:
        args += [
            "--SequentialMatching.loop_detection",
            "1",
            "--SequentialMatching.vocab_tree_path",
            str(vocab_tree_path),
        ]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def vocab_tree_matcher(
    colmap_bin: str,
    *,
    database_path: Path,
    vocab_tree_path: Path,
    use_gpu: bool = True,
    matching_type: str = "SIFT_BRUTEFORCE",
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    """vocab tree による全体マッチ (順序非依存). 大量/ループ撮影向け."""
    args = [
        "vocab_tree_matcher",
        "--database_path",
        str(database_path),
        "--VocabTreeMatching.vocab_tree_path",
        str(vocab_tree_path),
        "--FeatureMatching.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureMatching.type",
        matching_type,
    ]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def exhaustive_matcher(
    colmap_bin: str,
    *,
    database_path: Path,
    use_gpu: bool = True,
    matching_type: str = "SIFT_BRUTEFORCE",
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    args = [
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureMatching.type",
        matching_type,
    ]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def mapper(
    colmap_bin: str,
    *,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    refine_intrinsics: bool = True,
    refine_rig: bool = True,
    multiple_models: bool = True,
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    output_path.mkdir(parents=True, exist_ok=True)
    args = [
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
    ]
    if not refine_intrinsics:
        # 既知の厳密 intrinsics を固定する (仮想 pinhole rig).
        args += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]
    if not refine_rig:
        # rig 外参 (sensor_from_rig) を固定する. offset_v3 の校正を厳密に信頼する場合や,
        # 前後半球で共有点が無く相対姿勢を実測値で強制したい native fisheye で使う.
        args += ["--Mapper.ba_refine_sensor_from_rig", "0"]
    if not multiple_models:
        # 1 つの再構成のみ作る. COLMAP は既定で次善のシード群から複数サブモデルを吐くが,
        # rig 拘束下では最大モデル 1 つで十分なので分裂を抑止する.
        args += ["--Mapper.multiple_models", "0"]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def view_graph_calibrator(
    colmap_bin: str,
    *,
    database_path: Path,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    return run_command(
        colmap_bin,
        ["view_graph_calibrator", "--database_path", str(database_path)],
        log_path=log_path,
        on_line=on_line,
    )


def global_mapper(
    colmap_bin: str,
    *,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    refine_intrinsics: bool = True,
    refine_rig: bool = True,
    use_gpu: bool = True,
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    output_path.mkdir(parents=True, exist_ok=True)
    args = [
        "global_mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--GlobalMapper.ba_ceres_use_gpu",
        "1" if use_gpu else "0",
        "--GlobalMapper.gp_use_gpu",
        "1" if use_gpu else "0",
    ]
    if not refine_intrinsics:
        args += [
            "--GlobalMapper.ba_refine_focal_length",
            "0",
            "--GlobalMapper.ba_refine_principal_point",
            "0",
            "--GlobalMapper.ba_refine_extra_params",
            "0",
        ]
    if not refine_rig:
        args += ["--GlobalMapper.refine_sensor_from_rig", "0"]
    if extra_args:
        args += extra_args
    return run_command(colmap_bin, args, log_path=log_path, on_line=on_line)


def model_transformer(
    colmap_bin: str,
    *,
    input_path: Path,
    output_path: Path,
    transform_path: Path,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CommandResult:
    output_path.mkdir(parents=True, exist_ok=True)
    return run_command(
        colmap_bin,
        [
            "model_transformer",
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--transform_path",
            str(transform_path),
        ],
        log_path=log_path,
        on_line=on_line,
    )
