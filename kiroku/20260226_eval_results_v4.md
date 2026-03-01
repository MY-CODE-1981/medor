# fan_pipe_v4 評価結果

## 概要
WNF（Winding Number Field）のメッシュ厚み付けにより、布形状復元の精度を大幅に改善。

## 変更点
1. **メッシュ厚み付け**: 布の薄い2D表面を法線方向に±0.02オフセットして閉じた体積を作成
   - 変更ファイル: `scripts/convert_fan_dataset.py:compute_wnf()`
   - WNF > 0.5 のボクセル数: 224 (0.01%) → 72,972 (3.48%)
2. **surface_sample_ratio**: 0 → 0.5（学習時のサンプリングを表面近傍に偏向）
   - 変更ファイル: `garmentnets/config/train_pipeline_default.yaml`
3. **chamfer_distance_my() タプル修正**: eval時のクラッシュを修正
   - 変更ファイル: `garmentnets/networks/conv_implicit_wnf.py` (line 859)

## 学習設定
- モデル: fan_pipe_v4
- Canonicalization: fan_canon_v2 (500 epochs, 共有)
- Pipeline: 300 epochs, val_loss=0.0007
- データセット: fan_cloth_v2 (v7_back, 133 traj)
- チェックポイント: `data/release/fan_release/fan_pipe_v4/epoch=298-val_loss=0.0007.ckpt`

## 評価結果

### バージョン比較

| 指標 | v3 (薄いWNF) | v4 (厚み付きWNF) | 改善倍率 |
|------|-------------|-----------------|---------|
| chamfer_mesh | 0.0086 | 0.0002 | 43x |
| chamfer_pc | 0.0092 | 0.0011 | 8x |
| Marching cubes警告 | 14/14件 | 0/14件 | 全メッシュ正常 |

### v4 詳細結果（finetune なし）

| Sample | chamfer_mesh | chamfer_pc |
|--------|-------------|-----------|
| 0 | 0.000203 | 0.001107 |
| 1 | 0.000201 | 0.001149 |
| 2 | 0.000202 | 0.001074 |
| 3 | 0.000204 | 0.001152 |
| 4 | 0.000202 | 0.001070 |
| 5 | 0.000205 | 0.001093 |
| 6 | 0.000206 | 0.001072 |
| 7 | 0.000206 | 0.001099 |
| 8 | 0.000204 | 0.001072 |
| 9 | 0.000207 | 0.001072 |
| 10 | 0.000204 | 0.001081 |
| 11 | 0.000204 | 0.001070 |
| 12 | 0.000199 | 0.000963 |
| 13 | 0.000216 | 0.001662 |
| **mean** | **0.0002** | **0.0011** |
| **std** | **0.0000** | **0.0002** |

### v4 詳細結果（finetune あり）

| 指標 | 値 |
|------|-----|
| chamfer_mesh | 0.0002 (std=0.0000) |
| chamfer_pc | 0.0011 (std=0.0002) |
| chamfer_mesh_opt | 0.0008 (std=0.0001) |
| chamfer_pc_opt | 0.0009 (std=0.0004) |

## 可視化
- finetune なし: `data/release/fan_release/fan_eval_v4/{0-13}_vis.html`
- finetune あり: `data/release/fan_release/fan_eval_v4_ft/{0-13}_vis.html`

## 日付
2026-02-26
