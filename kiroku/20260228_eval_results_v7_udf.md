# fan_pipe_v7 (UDF) 評価結果

## モデル変更点 (vs fan_pipe_v6)
- Volume表現: WNF (Winding Number Field) → UDF (Unsigned Distance Field)
- tsdf_clip_value: null → 0.1 (距離を[0,1]に正規化)
- volume_absolute_value: True
- iso_surface_level: 0.5 → 0.1 (UDF用)
- gradient_direction: ascent → descent
- use_gradient_filter: True → False
- val_loss: 0.0000 (UDF値が小さいため数値上は低い)

## 評価結果 (15 test samples)

| 設定 | chamfer_mesh | chamfer_pc | chamfer_mesh_opt | chamfer_pc_opt |
|------|-------------|-----------|-----------------|---------------|
| fan_pipe_v7 (no ft) | 0.0001 | 0.0018 | 0.0001 | 0.0018 |
| fan_pipe_v7 + ft C | 0.0001 | 0.0018 | 0.0001 | 0.0008 |

## v6 との比較
| 設定 | chamfer_mesh | chamfer_pc | chamfer_mesh_opt | chamfer_pc_opt |
|------|-------------|-----------|-----------------|---------------|
| fan_pipe_v6 (no ft) | 0.0001 | 0.0010 | 0.0001 | 0.0010 |
| fan_pipe_v6 + ft C | 0.0001 | 0.0010 | 0.0001 | **0.0006** |
| fan_pipe_v7 (no ft) | 0.0001 | 0.0018 | 0.0001 | 0.0018 |
| fan_pipe_v7 + ft C | 0.0001 | 0.0018 | 0.0001 | 0.0008 |

## 分析
- **chamfer_mesh**: v6と同等 (0.0001) — volume表現の違いはメッシュ形状精度に大きく影響しない
- **chamfer_pc**: v6より悪化 (0.0010 → 0.0018) — UDFのsurface decoder学習が不十分
- **finetune C効果**: chamfer_pc_opt 0.0018 → 0.0008 (大幅改善) — finetuneが有効に機能

## 結論
- UDFはWNFと比べてchamfer_meshは同等だが、chamfer_pcが悪化
- UDFの利点(厚み付け不要)はあるが、精度面ではWNF+thickeningの方が良い
- **fan_pipe_v6 + finetune C が現時点での最良設定**
