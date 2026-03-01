# fan_pipe_v6 評価結果

## モデル変更点 (vs fan_pipe_v5)
- Surface decoder MLP: [128,256,256,3] → [128,512,512,256,3] (464K params, ~2.3x)
- num_surface_sample: 6000 → 12000
- max_epochs: 300 → 500
- val_loss: 0.0003 → **0.0002**

## 評価結果 (15 test samples)

| 設定 | chamfer_mesh | chamfer_pc | chamfer_mesh_opt | chamfer_pc_opt |
|------|-------------|-----------|-----------------|---------------|
| fan_pipe_v5 (baseline) | 0.0002 | 0.0008 | 0.0009 | 0.0008 |
| fan_pipe_v6 (no finetune) | **0.0001** | 0.0010 | 0.0001 | 0.0010 |
| fan_pipe_v6 + finetune (default) | 0.0001 | 0.0010 | 0.0010 | 0.0009 |

## RMSE (mm) 換算 (√chamfer × 1000)
| 設定 | mesh RMSE | pc RMSE | mesh_opt RMSE | pc_opt RMSE |
|------|----------|---------|--------------|-------------|
| fan_pipe_v5 | 14.1 | 28.3 | 30.0 | 28.3 |
| fan_pipe_v6 (no ft) | **10.0** | 31.6 | 10.0 | 31.6 |
| fan_pipe_v6 + ft (default) | 10.0 | 31.6 | 31.6 | 30.0 |

## 結論
- chamfer_mesh: 0.0002 → 0.0001 (RMSE 14.1mm → 10.0mm, **29%改善**)
- finetune (default設定) ではまだchamfer_mesh_optが悪化 (0.0001 → 0.0010)
- chamfer_pcはやや悪化 (0.0008 → 0.0010) — surface decoderの表面精度向上が点群精度に直結していない
