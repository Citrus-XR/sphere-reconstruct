"""外部 `.insv.pb` の読み取り骨組.

X5 では INSV と同名で拡張子だけ違う `.insv.pb` が MISC ディレクトリに置かれる
ことがあり, ここには (INSV フッタの offset_v3 より詳細な) MEI 拡張畜れみモデル
一式が入っていると観測されている.

`.pb` の名前から Protocol Buffers を連想するが, 実際の serialization がそう
なのかは要確認. 独立実装するため, 当面は「バイナリを読み込んで, 既知の
マジックがあるか確認, 具体的な decode は実サンプル入手後」に留める.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .calibration import DualLensCalibration


@dataclass
class PbFile:
    path: Path
    size: int
    raw_head: bytes  # 先頭 64 bytes だけ. デバッグ / マジック確認用.


def probe(path: Path, head_bytes: int = 64) -> PbFile:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        head = f.read(head_bytes)
    return PbFile(path=path, size=path.stat().st_size, raw_head=head)


def find_pb_for(insv_path: Path) -> Path | None:
    """INSV に紐付く .insv.pb を通常配置から探す.

    - 同ディレクトリの `<basename>.insv.pb`
    - 兄弟 `MISC/Camera01/<basename>.insv.pb`

    見つからなければ None. PB は必須ではない (offset_v3 / 内蔵 profile へ降級可能).
    """
    same_dir = insv_path.with_suffix(insv_path.suffix + ".pb")
    if same_dir.exists():
        return same_dir

    # ".insv" -> "" して .insv.pb に置換.
    # X5 の DCIM/CameraNN/xxx.insv に対する MISC/CameraNN/xxx.insv.pb 想定.
    parts = insv_path.parts
    if "DCIM" in parts:
        idx = parts.index("DCIM")
        base = Path(*parts[:idx])
        after = Path(*parts[idx + 1 :])
        candidate = base / "MISC" / after.with_suffix(after.suffix + ".pb")
        if candidate.exists():
            return candidate

    return None


def parse(path: Path) -> DualLensCalibration | None:
    """PB を読んで DualLensCalibration に変換する.

    現状は未実装. 手元の X5 サンプル (および D: 上の全 INSV) には外部 `.insv.pb` が
    存在せず, 標定は INSV footer 内の offset_v3 (ASCII) に埋まっている. そちらは
    calibration.parse_offset_v3_ascii で完全に解釈でき, 実機で pinhole rig の 100%
    登録・0.64px 誤差を確認済み.

    PB の binary フォーマットは実サンプルが無いと確定できず, 盲目的に推測実装しても
    検証できないため, ここは意図的に未実装のまま残す. PB を伴う個体の INSV が入手
    でき次第, probe() の raw_head からマジックを確認して本実装を足す. それまでは
    None を返し, 呼び出し側は offset_v3 へ降級する.
    """
    _ = probe(path)
    return None
