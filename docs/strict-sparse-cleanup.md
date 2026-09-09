# Native fisheye の厳格な sparse cleanup

この実験は既存 COLMAP 点群のうち、観測支持と位置の安定性を検証できた点だけを残す。既存点の座標・色・camera pose・intrinsics は変更せず、不合格点とその observation association を除去する。新しい点の補完、ERP 化、単眼 depth、道路の平面拘束は使わない。

[固定 pose 補完実験](fixed-pose-triangulation.md) では予測 coverage が増えても目視で大きな改善がなかった。このため、今回は基底点にも同じ除去条件を適用し、証拠不足の点も残さない。見た目の穴や遠景の欠損を許容する比較であり、production の cleanup default を変更するものではない。

## 判定条件

TestO2 の native `OPENCV_FISHEYE` を対象とする。Script は `OPENCV_FISHEYE` / `THIN_PRISM_FISHEYE` を明示的に受け付け、pinhole / ERP を含む入力は計算前に拒否する。混合 pinhole 入力に対する実証は今回の範囲外である。

1. Point が少なくとも四つの capture に観測されること。同一 source・同時刻の二眼は一つの capture と数える。
2. Capture を前後半と交互の二通りで分け、各 half の観測だけから ray triangulation を行うこと。退化した half や camera 後方へ出る結果は除去する。複数の非同期 fisheye source にまたがる track は、時刻を比較せず capture center の主軸順で分ける。
3. 各 split の二つの推定 XYZ 間、および各推定 XYZ と元 point 間の差が、元 point までの median camera range の 2% 以下であること。
4. Half で得た XYZ を他方の観測へ投影し、cross reprojection error の P95 が 2 px 以下、最大が 4 px 以下であること。Export で保持する元 XYZ 自体も全 observation に対して同じ投影条件を満たすこと。
5. 各 half の pixel-to-position の条件付き不確実性が range の 2% 以下であること。

Pixel の Jacobian は native fisheye projection の中央差分で計算する。固定 camera pose / intrinsics、独立等方 Gaussian noise `sigma=1 px` の仮定で、`sigma² (JᵀJ)⁻¹` を位置 covariance とする。3D の 95% ellipsoid の最大半径は `sqrt(chi²(3, 0.95)) × sigma / smallest_singular_value(J)`、係数は約 2.79548。この半径を元 point の median range で正規化し、四つの half 推定すべてが budget 以下であることを求める。

2% は比較用の許容値、1 px は測定ノイズの仮定であり、機種固有の calibration 値や実世界の精度保証ではない。Pose / calibration の系統誤差、相関する測定誤差、繰り返し模様の一貫した誤対応は、この条件付き評価を通過する可能性がある。

## TestO2 の実行結果

2026-09-09、元の 110,919 points を全数検査し、9,700 points（8.75%）を保持、101,219 points（91.25%）を除去した。理由は判定順の最初の不合格条件で集計する。

| 結果 | 点数 |
|---|---:|
| 保持 | 9,700 |
| 四 capture 未満 | 47,924 |
| 条件付き不確実性が budget 超過 | 30,545 |
| Split 間または元 XYZ との位置差が budget 超過 | 12,602 |
| Cross reprojection が上限超過 | 10,140 |
| Invalid projection | 5 |
| Split の三角化が退化 | 3 |

Sparse geometry の大部分を除去する設定であり、残った点の密度を目的に threshold を緩めていない。全 camera を残し、点を移動・追加しない。削除数は精度指標ではなく、実際の浮遊点や道路形状の見た目が改善したかは、この export の比較で判断する。

Stock COLMAP `model_analyzer` は 1 rig、2 cameras、276 registered frames、552 registered images、9,700 points、321,686 observations を読み込んだ。全 image / training mask 552 組の decode・寸法検査は成功し、両 sensor とも 3840×3840 を保持する。Surviving point の XYZ / RGB / track と全 camera は元 model と完全一致し、全保持点が要求した metric 上限を満たした。保持 observation がゼロになった画像は lens0 に 5 枚、lens1 に 1 枚あるが、画像と pose は削除していない。初期点の欠損が training に与える影響は目視比較の対象となる。

## 実行と成果物

Repository root から Backend environment の Python を使用する。各引数は実行環境の値へ置き換え、出力には source project 外の新規 directory を指定する。

今回の出力 directory 名は experiment root 相対で `testo2-strict-cleanup-final-20260909`。

```text
python scripts/export_strict_sparse.py <project> <experiment-output> --relative-error 0.02 --pixel-sigma 1 --max-cross-error 2
```

出力は以下の構成になる。

```text
<experiment-output>/
  status.json
  report.json
  point_assessment.npz
  removed_points.ply
  export_dataset/
    images/
    masks/
    sparse/0/
    preview/
    points.ply
    export_manifest.json
```

`point_assessment.npz` は全 input point の ID、最初に不合格になった理由、計算できた指標と model fingerprint を保存する。未計算の指標は NaN とし、良好な値で埋めない。`report.json` の理由別件数は排他的な最初の判定理由であり、他の条件には合格したという意味ではない。

LFStudio では `export_dataset` 自体を開く。`points.ply` は残した点、`removed_points.ply` は除去した点の比較用である。RGB と training mask は既存 exporter と同じ materialization / decode validation を使用し、画像は元の full resolution、mask は Feature ではなく Training の成果物を使う。同じ filesystem では hardlink を使うため、export 内の image / mask を直接上書き編集しない。LFStudio の通常の training output は別 file に保存される。

Viewer / training では native fisheye 用に GUT を有効、undistort を無効、mask mode を `segment` にする。この export に training hyperparameter の変更は含めない。点の除去で training 後の浮遊 Gaussian が必ず消えるとは限らない。

出力検査では binary model を再読込し、camera 不変性と point / image 双方向 track を確認する。全 registered image と training mask の decode、寸法と camera model の対応、元 model と関連 manifest の fingerprint も検査する。
