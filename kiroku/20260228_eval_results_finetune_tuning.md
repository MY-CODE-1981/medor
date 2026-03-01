# Finetune正則化チューニング結果

## ベースライン問題
fan_pipe_v6のfinetune (default設定) でchamfer_mesh_optが10倍悪化:
- before finetune: chamfer_mesh=0.0001
- after finetune: chamfer_mesh_opt=0.0010

Default finetune設定:
- chamfer3d_w=1.0, laplacian_w=0.01, edge_w=0.02
- obs_consist_w=10.0 (50iter後無効化)
- opt_lr=0.001, opt_iter_total=100

## 実験結果

| 実験 | 設定 | chamfer_mesh_opt | chamfer_pc_opt |
|------|------|-----------------|---------------|
| Baseline (no ft) | - | 0.0001 | 0.0010 |
| Default ft | lr=0.001, iter=100, lap=0.01, edge=0.02 | 0.0010 | 0.0009 |
| **A**: 正則化強化 | lap=0.5, edge=0.5, obs=20 | 0.0010 | 0.0013 |
| **B**: 低LR+少iter | lr=0.0001, iter=30 | **0.0001** | 0.0008 |
| **C**: A+B組合せ | lap=0.5, edge=0.5, lr=0.0001, iter=50 | **0.0001** | **0.0006** |

## RMSE (mm) 換算
| 実験 | mesh_opt RMSE | pc_opt RMSE |
|------|--------------|-------------|
| No finetune | 10.0 | 31.6 |
| Default ft | 31.6 | 30.0 |
| A: 正則化強化 | 31.6 | 36.1 |
| B: 低LR+少iter | **10.0** | 28.3 |
| **C: 組合せ** | **10.0** | **24.5** |

## 分析
- **Experiment A** (正則化のみ強化): 効果なし。chamfer lossがまだ支配的でメッシュを歪める
- **Experiment B** (低LR+少iter): 大幅改善! finetune悪化が解消。保守的な最適化が有効
- **Experiment C** (A+B組合せ): **最良**。正則化+低LRの組合せでchamfer_pc_optも0.0006に

## 結論
- 問題の根本原因: 学習率が高すぎ (0.001) + iteration数が多すぎ (100)
- **推奨設定**: `--opt_lr 0.0001 --opt_iter_total 50 --laplacian_w 0.5 --edge_w 0.5`
- この設定で chamfer_pc_opt が 0.0010 → 0.0006 (RMSE 31.6mm → 24.5mm, **22%改善**)
