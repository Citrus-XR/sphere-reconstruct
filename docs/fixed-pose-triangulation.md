# 固定 pose 再三角化と独立観測の検証

2026-09-09 の TestO2 では、近い capture の対応を一律に削除しても改善しなかった。一方、厳格な点群を保ち、低視差候補のうち訓練側の前後時刻で安定した点だけを補完すると、留保画像で 2 px 以内に投影できる予測が 25–42% 増えた。元の浮遊点を除去した結果ではなく、既存 camera poses と verified matches を条件とする coverage 改善である。Production dataset と既定設定は変更していない。

## 対象と比較の境界

COLMAP 4.1.1、native `OPENCV_FISHEYE` 二眼、276 captures / 552 images、SIFT の同一 matched database を使用する。元 model は 110,919 points。Image、keypoint、intrinsics、sensor-from-rig、frame pose は固定し、ERP 合成、再投影画像、学習型 depth、道路全体の平面拘束を使わない。COLMAP / LFStudio source も変更しない。

以前の motion3 / motion4 抽帧比較には交絡があった。元の `tri_create/continue_max_angle_error=0.75`、`tri_merge/complete_max_reproj_error=1`、`tri_min_angle=5` に対し、それらの試行では `2 / 4 / 1.5` に戻っていた。その結果から motion3 が最良だとは結論できない。現行 spatial flow は縮小画像の Farneback flow で、回転補償した並進視差ではない。今回の近距離 pair 除外も、動画を再抽出して選択を最適化する実験とは異なる。

`point_triangulator` には `--clear_points 1`、`--refine_intrinsics 0`、`--Mapper.fix_existing_frames 1` と sensor / intrinsic refinement の無効化を明示する。CPU CERES、元 manifest の BA iteration 設定、七つの三角化閾値を再使用する。基準閾値は filter reprojection 1 px、filter angle 5°、create / continue angle 0.75°、merge / complete reprojection 1 px、triangulation angle 5°。

CLI の成功だけでは固定性を保証しない。各出力を元 model と直接比較し、Sim3 alignment を挟まず、552 images、camera center 差の最大 `1.30e-14`、rotation matrix 差 `4.44e-16`、intrinsic 差 0 を確認した。Database keypoint row と model の XY も一致する。元 database / binary model / manifest / input spec の fingerprint は実験前後で不変だった。

根拠: [point_triangulator の固定条件](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/exe/sfm.cc#L628-L657)、[retriangulation pipeline](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/controllers/incremental_pipeline.cc#L752-L783)。

## 検証方法

Capture index が `index % 5 == 2` の capture を留保する。同時刻の二眼をまとめて分離し、留保画像に関係する全 verified pairs を三角化用 database copy から削除する。候補 point tracks に留保 observation が一つもないことを出力から確認する。

留保 keypoint に対して、元の verified matches を介し、異なる訓練 capture 二つ以上が同じ candidate point を支持するときだけ投影を評価する。複数 point が競合する keypoint は ambiguous として除外し、数を別記する。全 case 共通の分母は、二つ以上の訓練 capture との対応を持つ 573,983 keypoints。候補の削除で誤差だけが小さくなることを避けるため、coverage と 2 px 以内の予測数を同時に評価する。

時間分割では、track が四つ以上の異なる capture にまたがる場合に限り、前半と後半の native fisheye rays から XYZ を別々に三角化する。その距離差を元 point までの median camera range で割り、互いの観測への cross reprojection も測る。反対向きに近い ray も退化として扱う。四 capture 未満は検証不能であり、誤点と断定しない。複数の非同期 source をまたぐ track の時間分割は未定義として除外するため、今回の結果だけで混合入力への有効性は保証しない。

これは geometry の ground truth 検証ではない。留保画像の poses と verified matches は元の SfM に由来する。軌跡全体の歪みや、繰り返し模様に対して共通する誤対応は通過し得る。2 px は評価用の許容誤差で、正しい実世界位置の証明ではない。

## 三角化と補完の結果

`heldout_wide2` は同一 source の rig-center 距離が隣接 capture 移動中央値の二倍未満の pair を除外する。この dataset では境界は 0.652232 model units。基準 3,876 pairs から 3,203 pairs に減った。

`heldout_far` は filter / triangulation の最小角だけを 5° から 1.5° に変更する。全 244,351 points の時間分割検査では 126,336 points を解け、10,753 points（8.51%）が 10% 超、2,270 points が 25% 超の XYZ 差を示した。117,954 points は capture 数不足、61 points は退化である。追加点をそのまま採用する根拠はない。

補完版は厳格な `heldout_all` をそのまま保持する。`heldout_far` の訓練側時間分割について、相対 XYZ 差が各 budget 以下、cross reprojection 中央値が 2 px 以下の点を候補とする。既存 point に割り当て済みの observation を一つでも含む候補は追加しない。選択に留保画像の誤差は使わない。Budget 2 / 5 / 10% は比較する許容誤差であり、機種共通の新しい既定値ではない。新しく生成する stability archive は model binary の SHA-256 を持ち、補完時に同じ model であることを検査する。Fingerprint のない初回実験の archive を再利用する場合は、同じ candidate を再 audit して作り直す。

| 条件 | Points | 留保予測 coverage | 誤差中央値 | 誤差 P95 | 2 px 以内の予測数 |
|---|---:|---:|---:|---:|---:|
| 厳格 `heldout_all` | 103,761 | 25.68% | 0.763 px | 2.966 px | 129,513 |
| 近距離 pair 除外 | 87,834 | 23.79% | 0.753 px | 2.934 px | 120,429 |
| 最小角 1.5°、全候補 | 244,351 | 36.35% | 0.797 px | 3.197 px | 180,356 |
| 厳格 + 安定候補 2% | 121,102 | 32.14% | 0.758 px | 2.965 px | 162,358 |
| 厳格 + 安定候補 5% | 133,463 | 35.04% | 0.762 px | 2.963 px | 176,967 |
| 厳格 + 安定候補 10% | 140,787 | 36.39% | 0.767 px | 2.969 px | 183,596 |

近距離 pair 除外は coverage を相対 7.37% 失い、中央値改善は約 0.01 px に留まる。採用しない。安定候補 2% は 17,341 points を追加し、2 px 以内の留保予測は純増 32,845（25.36%）。5% は 29,702 points を追加し、純増 47,454（36.64%）。10% はより多くの depth 不確実性を許容するため、preview 優先候補は保守的な 2% と coverage を重視した 5% とする。

三つの補完版すべてで、基準と共通の留保 keypoint の投影誤差は完全に同じだった。2% では 38,487 predictions が追加され、そのうち 33,736 が 2 px 以内。一方、候補競合により以前は評価可能だった 1,418 predictions が ambiguous になり、そのうち 891 が 2 px 以内だった。純増はこの損失を差し引く。既存点の XYZ を変えないことと、競合が増えないことは同義ではない。

両 sensor・軌跡四分割の全区間で 2 px 以内の予測が純増した。2% の前半から後半への純増は lens0 で `6335 / 5843 / 4583 / 3153`、lens1 で `2853 / 4235 / 3826 / 2017`。開始部分だけの改善ではない。ただし、道路・空・遠景などの semantic 領域別の正解率や training 画質は未検証である。

元 model と全観測での固定再三角化は、それぞれ 110,919 / 133,196 points だった。留保対象の画像も三角化へ使うため、これらの投影結果を上表の未使用 observation と直接比較しない。元 model の時間分割 sample 9,244 points では、4,009 が capture 数不足、5,235 が解け、そのうち 58 が XYZ 差 10% 超だった。長い track の多くは条件付きで安定する一方、短い track には信頼できる深度判定の証拠が不足する。

## 成果物と再実行

Backend environment の Python から実行する。Output は source project 外の新規 directory とし、実在 path を引数へ渡す。

```text
python scripts/benchmark_fixed_pose.py <project> <strict-experiment> --colmap <colmap> --cases fixed_all heldout_all heldout_wide2 --detach
python scripts/audit_fixed_pose.py <strict-experiment> --database <project>/reconstruct/database.db
python scripts/benchmark_fixed_pose.py <project> <far-experiment> --colmap <colmap> --cases heldout_far --detach
python scripts/audit_fixed_pose.py <far-experiment> --database <project>/reconstruct/database.db --cases heldout_far --skip-reference-audit --sample-points 1000000
python scripts/supplement_stable_points.py <strict-experiment>/heldout_all/sparse/0 <far-experiment>/heldout_far/sparse/0 <far-experiment>/stability_heldout_far.npz <supplement-experiment> --experiment-reference <strict-experiment> --relative-error-budgets 0.02 0.05 0.1 --max-cross-error-px 2
python scripts/audit_fixed_pose.py <supplement-experiment> --database <project>/reconstruct/database.db --cases heldout_supplement_02pct heldout_supplement_05pct heldout_supplement_10pct --skip-reference-audit
python scripts/compare_fixed_pose_validation.py <strict-experiment>/validation_heldout_all.npz <supplement-experiment>/validation_heldout_supplement_02pct.npz <supplement-experiment>/validation_heldout_supplement_05pct.npz <supplement-experiment>/validation_heldout_supplement_10pct.npz --database <project>/reconstruct/database.db --spec <strict-experiment>/input_spec.json --output <supplement-experiment>/cohort_comparison.json
```

今回の保存先は workspace sibling の `experiments/testo2-fixed-pose-20260909`、`testo2-fixed-pose-far-20260909`、`testo2-fixed-pose-supplement-20260909`。各補完 case に native camera の `sparse/0` と colored `points.ply` を保存した。`audit.json`、`validation_*.npz`、`stability_*.npz`、`cohort_comparison.json` は評価根拠である。5% 版は stock COLMAP の `model_analyzer` でも読め、1 rig、2 cameras、276 registered frames、552 registered images、133,463 points を確認した。これらは検証用 sparse model / point cloud で、画像と training mask を含む LFStudio dataset export ではない。

この段階で示せたのは、カメラを変えずに検証可能な補完点を増やせることまでである。元の浮遊点は保持しており、弱 texture 全体の clean geometry や最終学習画質の解決は未確認。Production cleanup に採用するには、既存の弱支持点を分類する独立の証拠と視覚検証が必要になる。
