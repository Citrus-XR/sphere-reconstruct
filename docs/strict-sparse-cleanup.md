# Full-track sparse cleanup と native fisheye 比較

正式な `cleanup_sparse` Step は既存 COLMAP 点群のうち、全 track の観測支持と位置の安定性を検証できた点だけを残す。既存点の座標・色・camera pose・intrinsics は変更せず、不合格点とその observation association を除去する。新しい点の補完、ERP 化、単眼 depth、道路の平面拘束は使わない。

[固定 pose 補完実験](fixed-pose-triangulation.md) では予測 coverage が増えても目視で大きな改善がなかった。全既存点を対象とした split cleanup は浮遊点を減らした一方、大きな欠損が生じ、LFStudio が遅いという報告があった。元 model・split・full-track の LFStudio 比較と利用者の目視評価を踏まえ、full-track を正式 Step に採用した。削除数や coverage 単独で最適解とは判定しない。

## 正式 Step の設定と対象

`scene_alignment/sparse/0` と `extract_features/input_spec.json` を入力にし、後者の `(source_id, capture_index)` で独立 capture を識別する。異なる source の同じ frame 番号は別 capture。同一 capture の二眼や virtual views はまとめて留保し、source 間の撮影開始時刻や固定相対 pose を仮定しない。異なる camera は自身の intrinsics、distortion、pose で直接投影する。

既定は有効、相対誤差 `relative_error=0.02`、pixel noise の仮定 `pixel_sigma=1`、reprojection P95 上限 `max_cross_error=2 px`（最大値は 4 px）。Inspector では相対誤差を百分率で編集する。最低 3 capture は固定条件。全 parameter は正の有限値とし、全点が不合格なら空 model を publish せず失敗を報告する。無効時は元 model を通す。

対応 camera は `OPENCV_FISHEYE`、`THIN_PRISM_FISHEYE`、`PINHOLE`、`SIMPLE_PINHOLE`、`SIMPLE_RADIAL`、`RADIAL`。通常の写真・phone video を含む mixed input にも同じ計算を適用できる。Mixed の合成幾何テストはあるが、以下の実素材比較は TestO2 の二眼 fisheye のみであり、mixed の画質改善を実証したものではない。ERP など未対応 model は明示的に失敗するため、当該 workflow では cleanup を無効にする。

`cleanup_sparse/point_assessment.npz` に point ID、metric column、判定理由を記録し、`cleanup_sparse.json` と manifest に理由別件数と observation がゼロの画像数を保存する。Camera、rig、frame、保持 point の XYZ / RGB / track を保持し、scene-aligned model から直接 export できる。判定実装は Backend の `colmap/point_stability.py` に集約し、比較 CLI も共有する。旧 distance / angle filter は重ねて適用しない。

旧 cleanup の UI 編集値は起動時に server DB で新設定へ移行する。保存済み enable 状態や他 Step の数値は保持する。Triangulation preset 名は七つの実値から導出し、別保存されていた名称は DB から除去する。以前の実行 manifest と成果物は当時の記録として保持し、再実行時は実装 version `2.0` と新 parameter / input hash により旧 cache を採用しない。再実行するまで既存 export / training の内容は変わらない。Sparse reconstruction の未指定値は [厳格 preset](setup-gpu.md#incremental-mapper-の三角測量設定) になり、明示的に保存した各値は上書きしない。

## Split policy の判定条件

以下の比較は TestO2 の native `OPENCV_FISHEYE` を対象とする。比較 CLI `export_strict_sparse.py` は `OPENCV_FISHEYE` / `THIN_PRISM_FISHEYE` に対象を限定する。正式 Step の mixed 対応範囲とは区別する。

1. Point が少なくとも四つの capture に観測されること。同一 source・同時刻の二眼は一つの capture と数える。
2. Capture を前後半と交互の二通りで分け、各 half の観測だけから ray triangulation を行うこと。退化した half や camera 後方へ出る結果は除去する。複数の非同期 fisheye source にまたがる track は、時刻を比較せず capture center の主軸順で分ける。
3. 各 split の二つの推定 XYZ 間、および各推定 XYZ と元 point 間の差が、元 point までの median camera range の 2% 以下であること。
4. Half で得た XYZ を他方の観測へ投影し、cross reprojection error の P95 が 2 px 以下、最大が 4 px 以下であること。Export で保持する元 XYZ 自体も全 observation に対して同じ投影条件を満たすこと。
5. 各 half の pixel-to-position の条件付き不確実性が range の 2% 以下であること。

Pixel の Jacobian は native fisheye projection の中央差分で計算する。固定 camera pose / intrinsics、独立等方 Gaussian noise `sigma=1 px` の仮定で、`sigma² (JᵀJ)⁻¹` を位置 covariance とする。3D の 95% ellipsoid の最大半径は `sqrt(chi²(3, 0.95)) × sigma / smallest_singular_value(J)`、係数は約 2.79548。この半径を元 point の median range で正規化し、四つの half 推定すべてが budget 以下であることを求める。

2% は採用した許容値、1 px は測定ノイズの仮定であり、機種固有の calibration 値や実世界の精度保証ではない。Pose / calibration の系統誤差、相関する測定誤差、繰り返し模様の一貫した誤対応は、この条件付き評価を通過する可能性がある。

## Full-track policy

`--policy full_track` は 3 capture 以上を要求し、全観測の Jacobian で条件付き不確実性を評価する。全 track の再三角化結果と元 XYZ の差も 2% 以下とする。その後、各 capture を一つずつ留保し、残る capture で再三角化して留保画像へ投影する。同時刻の二眼は一緒に除外する。退化、camera 後方、非有限値、全留保予測の P95 > 2 px または最大 > 4 px を不合格とする。元 XYZ 自体も同じ pixel 上限を満たす必要がある。

留保ごとの XYZ 差は diagnostic として記録するが、独立した 2% gate は課さない。「各 subset が単独でも同じ深度精度を持つ」という split policy の条件を維持する方式ではない。条件付き不確実性の仮定と実世界精度の限界は両 policy で共通する。

`--policy original` は無加工の baseline export であり、error budget を受け付けず geometry assessment を行わない。

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

Split は sparse geometry の大部分を除去する設定であり、目視では浮遊点の減少と大きな欠損が報告された。全 camera を残し、点を移動・追加しない。削除数は精度指標ではない。

Stock COLMAP `model_analyzer` は 1 rig、2 cameras、276 registered frames、552 registered images、9,700 points、321,686 observations を読み込んだ。全 image / training mask 552 組の decode・寸法検査は成功し、両 sensor とも 3840×3840 を保持する。Surviving point の XYZ / RGB / track と全 camera は元 model と完全一致し、全保持点が要求した metric 上限を満たした。保持 observation がゼロになった画像は lens0 に 5 枚、lens1 に 1 枚あるが、画像と pose は削除していない。初期点の欠損が training に与える影響は目視比較の対象となる。

Full-track は同じ 110,919 points から 55,034 points（49.62%）を保持した。最初の除去理由は capture 数不足 29,553、条件付き不確実性 20,343、留保 reprojection 5,988、invalid projection 1。保持点の最大留保 XYZ 差が 2% を超えた点は 0、point ごとの最大留保差の中央値は 0.125%、P95 は 0.540%、最大は 1.849%。これは今回の観測結果であり、任意の入力への保証ではない。

| 指標 | Original | Split | Full track |
|---|---:|---:|---:|
| 保持点数 | 110,919 | 9,700 | 55,034 |
| Observation がゼロの画像 | 0 | 6 | 0 |
| 画像ごとの occupied-cell 保持率 P05 | 100% | 5.71% | 58.19% |
| 同中央値 | 100% | 46.67% | 80.33% |
| 同 P95 | 100% | 69.65% | 93.12% |

Coverage は各画像を 32×32 cell に分け、元 sparse observation が存在する cell のうち保持された割合である。初期点分布を表し、道路面の完全性や geometry accuracy の ground truth ではない。両 cleanup の binary 再読込、point / image 双方向 association、surviving XYZ / RGB / track の一致を確認した。Camera center・rotation・intrinsics の差は 0。元 model と metadata の fingerprint は不変で、全 552 組の RGB / Training mask の decode・寸法検査に合格した。

## LFStudio 比較条件

Installed build は `v0.5.3-386-g395c7f31-dirty`。旧 preset JSON を読み込まず、CLI と実行ログを確認する。`scripts/benchmark_lfs_training.py` が隔離 output、invocation、binary fingerprint、stdout / stderr、10 秒ごとの GPU 使用量、終了状態、`perf_bench.json` を保存する。GPU job は順番に実行し、正常終了だけでなく `.licht`・PLY・perf report の存在も確認する。

共通条件は MRNF、30,000 iterations、1,000,000 Gaussian cap、`resize_factor=1`、`max_width=0`、GUT、`mask_mode=segment`、元画像と既存 Training mask。Undistort、PPISP / controller、depth / normal loss、sparsity は有効にしない。Cap は比較用の共通予算であり、画質上の最適値や任意場面での 12 GB 保証値ではない。

標準三群と成長期間延長の追加群は 30,000 steps を正常完走した。時間は supervisor の開始・保存を含む値、steady は LFStudio の perf collector、peak は CUDA 使用量の記録である。比較ページの時間は perf collector の wall time のため、起動・終了待ちの分だけ短い。

| 指標 | Original | Split | Full track | Split + growth 延長 |
|---|---:|---:|---:|---:|
| 完走時間 (min) | 36.67 | 21.93 | 32.14 | 28.63 |
| Steady ms / step | 72.77 | 42.92 | 63.64 | 56.69 |
| Peak CUDA (MiB) | 7,131.5 | 7,305.5 | 7,587.5 | 8,523.5 |
| 最終 Gaussian 数 | 1,000,000 | 348,475 | 1,000,000 | 1,000,000 |
| 8 視点平均 masked PSNR (dB) | 20.752 | 20.236 | 20.669 | 20.545 |

四群とも非有限の Gaussian 座標・scale は検出されず、OOM は発生しなかった。今回の条件では Split が 1 step あたり遅くなる現象は再現しなかった。ただし以前の GUI training の設定が不明なため、その遅さの原因まで確定した結果ではない。

Full track は Split より欄干や植生の輪郭を保持し、8 視点平均 PSNR は 0.433 dB 高い。Original との差は −0.082 dB と小さく、全体として Original の描写へ近づいた結果であって、明確な画質向上の実証ではない。元 pixel crop では三群とも近距離の柏油路がぼやけ、天空の大きな色むらも残る。RNG 差を含む単回比較であり、小さな score 差を最適化の根拠にしない。Training 後の浮遊 geometry を ground truth で分類したわけでもない。

標準 MRNF は `grow_until_iter=15000` で新規増加を止める。ログでも Split は iteration 14,800 に 348,475 点へ増え、その後は soft prune と同数の replacement のみで、100 万 cap には達しなかった。Cap は目標点数ではない。追加比較では `scripts/configs/lfs_mrnf_full_refinement_growth.json` を使い、同じ厳格 dataset の成長期間だけを既存 `stop_refine=28500` まで延長した。Config の他の明示 field は当該 build の MRNF defaults と JSON 必須 field を保持し、残りは `strategy=mrnf` に対応する defaults から読む。LFS source は変更していない。

延長群は iteration 16,000 に 19,072 点、16,200 に 20,030 点を追加し、15,000 以降にも増加することをログで確認した。最終的に 100 万点へ到達し、Split より平均 PSNR が 0.309 dB 改善した。ただし所要時間は約 6.70 分、peak CUDA は 1,218 MiB 増え、描写は Full track に達しない。近景路面の大きなぼけや斑状の描写も残った。厳格 cleanup 後の成長制限は損失の一因だが、成長期間だけの延長を一般的な修復設定として採用する根拠はない。

初期点を約半分に減らしつつ Original に近い描写を維持し、利用者の目視で最も良好と評価された Full track を正式 cleanup に採用する。これは初期点削減と coverage の比較上の妥協案であり、最終浮遊点の完全除去や低 texture 面の正しい深度を実証したものではない。LFS source と training defaults は変更しない。

成長期間の根拠: [default](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/core/include/core/parameters.hpp#L225)、[新規増加の条件](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/strategies/mrnf.cpp#L1972)、[strategy defaults の JSON 読込](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/core/parameters.cpp#L703)。この追加比較は標準三群とは異なる schedule であり、別群として扱う。

開始・中盤二箇所・終盤の両眼、計 8 視点を 5,000 iteration ごとに JPEG timelapse へ出力する。全画像を training に使うため、今回は training-view fidelity の比較であり、held-out accuracy ではない。`scripts/summarize_lfs_comparison.py` は元 RGB、Training mask、render の contact sheet と元 pixel の crop、mask 内 MSE / PSNR、Gaussian 数の推移を保存する。JPEG と training view の限界があり、PSNR だけで浮遊点の解消とは判定しない。

`--test-every` は camera 配列 index による分割で、capture 単位ではない。交互二眼順では片眼だけを留保する場合がある。Camera sampler は固定 seed だが MRNF growth / noise は時刻 seed を含み、完全再現性はない。`segment` は黒い mask 領域の RGB loss を除外し、alpha を下げる penalty も適用する。`ignore` は RGB loss を除外するが alpha penalty を加えない。「Mask を使わない」という意味ではない。

実装根拠: [camera split](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/training_setup.cpp#L654)、[native GUT distortion](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/rasterization/gsplat_rasterizer.cpp#L304)、[mask loss](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/trainer.cpp#L1793)、[timelapse](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/trainer.cpp#L7639)、[MRNF RNG](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31/src/training/strategies/mrnf.cpp#L1900)。`dirty` suffix により upstream source と binary の完全一致は保証しない。

## 実行と成果物

Repository root から Backend environment の Python を使用する。各引数は実行環境の値へ置き換え、出力には source project 外の新規 directory を指定する。

出力は experiment root 相対で次の directory に分離する。

| 内容 | Directory |
|---|---|
| 元 dataset | `testo2-original-baseline-20260909/export_dataset` |
| Split dataset | `testo2-strict-cleanup-final-20260909/export_dataset` |
| Full-track dataset | `testo2-full-track-cleanup-20260909/export_dataset` |
| Split training | `testo2-lfs-strict-30k-valid-20260909/training` |
| Full-track training | `testo2-lfs-full-track-30k-20260909/training` |
| Original training | `testo2-lfs-original-30k-20260909/training` |
| Split の成長期間延長 | `testo2-lfs-strict-extended-30k-20260909/training` |
| 最終 Render comparison | `testo2-lfs-comparison-20260909/final/index.html` |

```text
python scripts/export_strict_sparse.py <project> <original-output> --policy original
python scripts/export_strict_sparse.py <project> <split-output> --policy split --relative-error 0.02 --pixel-sigma 1 --max-cross-error 2
python scripts/export_strict_sparse.py <project> <full-track-output> --policy full_track --relative-error 0.02 --pixel-sigma 1 --max-cross-error 2
python scripts/compare_sparse_cleanup.py <project>/reconstruct/sparse/0 <split-output>/export_dataset/sparse/0 <full-track-output>/export_dataset/sparse/0 --output <comparison>/coverage.json
python scripts/benchmark_lfs_training.py <lfs-executable> <dataset> <training-output> --iterations 30000 --max-cap 1000000 --detach --timelapse-every 5000 --timelapse-images <image-name-1> --timelapse-images <image-name-2>
```

`--timelapse-images` は画像ごとに繰り返す。後続 job の `--wait-for <previous-output>/status.json` は前 job が成功した場合だけ開始し、失敗は後続 job も失敗として記録する。

各 training directory の `project.licht` は完走した工程、`splat_30000.ply` は最終 Gaussian。最終比較の `index.html` は元 RGB・Training mask・四群の画面と元 pixel crop へのリンクをまとめる。`comparison.json` は全画像と crop の score、性能記録、Gaussian geometry 要約を持つ。元 RGB / mask の file identity または hash、render の元解像度を照合してから生成する。

Cleanup output は以下の構成になる。

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
