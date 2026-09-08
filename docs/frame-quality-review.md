# 抽帧品質レビュー

対象は `Reflct/sharp-frames-python` の commit [`95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1`](https://github.com/Reflct/sharp-frames-python/tree/95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1)。本 project の Spatial sampler と source role を維持し、同 library 全体は依存関係へ追加しない。Sharp Frames は鮮鋭度と時間分布を選択する library であり、optical flow、parallax、camera rotation、SfM connectivity は推定しない。

## 再現できた問題

- 現在の鮮鋭度は画像を四分の一に縮小した Laplacian variance。解析解像度が source resolution に依存し、noise によって score が増える。
- Fisheye の黒い矩形 corner を exposure 判定に含めている。`r=0.5` の円外は約 21.45%。有効円内の暗い画素が 5% あるだけで全画像の暗部率は約 25.38% となり、`max_clip=0.25` を超える。
- Spatial 選択で候補が全件棄却された場合、現実装は uniform fallback で未 filter の候補を戻す。Threshold を満たした選択と区別して扱う必要がある。
- Spatial branch は dense local sharpness refinement を通らない。`candidate_fps=1.5` では約 0.667 秒間隔の候補同士しか比較できない。
- Flow median は画像上の移動 proxy であり、回転と translation を分離した幾何学的 parallax ではない。

## 比較実験

Project-local Python environment で upstream の scoring code を実行した synthetic probe。値は別の尺度であり、そのまま threshold を置き換えない。

| 入力 | 現在の Laplacian | Upstream focus |
| --- | ---: | ---: |
| 鮮明な edge | 254.0 | 188.7 |
| Blur + noise の edge | 2998.0 | 118.3 |
| 同じ edge、256 / 1024 / 4096 px | 2032 / 508 / 127 | 135.7 / 188.7 / 188.7 |

Upstream の [focus scoring](https://github.com/Reflct/sharp-frames-python/blob/95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1/sharp_frames/focus_scoring.py#L15) は long edge 512、軽い Gaussian smoothing、Laplacian と Tenengrad の組合せを使う。Synthetic 1920-square 画像の single-thread scoring は現方式約 1.49 ms/image、upstream 約 5.17 ms/image。Decode、SIFT、flow は計測に含めず、GPU は使用しない。これは scoring の比較であり、実素材の再構成品質向上を実証した結果ではない。

## 改善の優先順位

1. 物理 sensor support だけで exposure / feature / flow を評価し、全件棄却は診断付き失敗として扱う。保存済み brush を抽帧へ適用する場合は、現行の `PREPARE_IMAGES` から始まる invalidation 依存も変更する必要がある。
2. Canonical resolution と noise-resistant score を A/B 比較する。Source 内の relative score と [local median / MAD による blur dip](https://github.com/Reflct/sharp-frames-python/blob/95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1/sharp_frames/selection_methods.py#L192) は、scene 固有の絶対閾値を増やさず候補を順位付けできる。現在保存されている `min_sharpness` の数値を新尺度へ流用しない。
3. Coarse motion anchor の周囲で元動画を細かく再評価し、鮮明な近傍 frame へ移動する。移動後は motion / overlap / rolling shutter を再検査し、dual-lens の同期と primary / supplemental role を保つ。
4. Fisheye-only と mixed-source の実素材で registered fraction、graph connectivity、triangulation angle、reprojection error、near-duplicate rate を比較する。鮮鋭度 score だけで最良と判定しない。

この UI / state 改修では抽帧 algorithm、保存済み threshold、既存 dataset を変更していない。実素材での A/B は別の計算実験として扱う。Upstream 自身も [audit](https://github.com/Reflct/sharp-frames-python/blob/95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1/docs/APPLICATION_AUDIT.md#L218) で代表的な実素材・cross-camera validation を残課題にしている。

License は [MIT](https://github.com/Reflct/sharp-frames-python/blob/95fca3ce2e86d3f6004bf8dfa65c073ac8ced7c1/LICENSE#L1)。将来 code を取り込む場合は attribution を残す。Package 全体は Textual と `opencv-python` などを導入するため、現行 headless OpenCV 環境には scoring の必要部分だけを統合する方が適切。
