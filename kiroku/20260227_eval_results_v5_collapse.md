# fan_pipe_v5 + collapse_shell 評価結果

## 概要
MC殻メッシュを薄膜に変換する後処理（collapse_shell）を追加して評価。
結果: **chamfer_meshは微増、finetuneの悪化は未解決**。

## 実装内容
- `garmentnets/common/marching_cubes_util.py` に `collapse_shell_to_surface()` 追加
- MC出力の上面のみ保持し、z座標を中央値に射影
- 頂点数を約半減: 23,111 → 12,562 (54%)
- `--collapse_shell` フラグで有効化

## 評価結果

### 全バージョン比較（物理スケール: 布幅231mm）

| 指標 | v5 (shell) | v5+collapse | v5+collapse+ft |
|------|-----------|-------------|----------------|
| chamfer_mesh | 0.0002 | 0.0002 | 0.0002 |
| → RMSE (mm) | **12.8** | 13.8 | 13.8 |
| chamfer_mesh_opt | 0.0009 | 0.0002 | 0.0011 |
| → RMSE (mm) | 30.0 | 13.8 | 33.2 |
| chamfer_pc | 0.0008 | 0.0010 | 0.0010 |
| → RMSE (mm) | 28.3 | 31.6 | 31.6 |

### v5+collapse 各サンプル詳細（finetune なし）

| Sample | chamfer_mesh | RMSE(mm) | chamfer_pc | RMSE(mm) |
|--------|-------------|----------|-----------|----------|
| 0 | 0.000186 | 13.6 | 0.001045 | 32.3 |
| 1 | 0.000187 | 13.7 | 0.001043 | 32.3 |
| 2 | 0.000187 | 13.7 | 0.001047 | 32.4 |
| 3 | 0.000187 | 13.7 | 0.001031 | 32.1 |
| 4 | 0.000188 | 13.7 | 0.001061 | 32.6 |
| 5 | 0.000188 | 13.7 | 0.001044 | 32.3 |
| 6 | 0.000188 | 13.7 | 0.001046 | 32.3 |
| 7 | 0.000189 | 13.7 | 0.001040 | 32.3 |
| 8 | 0.000189 | 13.8 | 0.001061 | 32.6 |
| 9 | 0.000189 | 13.8 | 0.001018 | 31.9 |
| 10 | 0.000189 | 13.8 | 0.001079 | 32.9 |
| 11 | 0.000188 | 13.7 | 0.001035 | 32.2 |
| 12 | 0.000192 | 13.9 | 0.000666 | 25.8 |
| 13 | 0.000201 | 14.2 | 0.001199 | 34.6 |
| **mean** | **0.0002** | **13.8** | **0.0010** | **32.0** |

## 分析: なぜcollapseが効かなかったか

### 1. ボトルネックはMC殻ではなくsurface decoder
- MC→GT距離 (NOCS): 3.4mm（殻の厚み由来）
- chamfer_mesh RMSE: 12.8mm（全体誤差）
- **差分 9.4mm** はsurface decoderのwarp field予測誤差
- collapseはMCレベルの3.4mmしか改善できない

### 2. 被覆率低下による悪化
- collapse後の頂点数: 23k → 12.5k（MC段階）
- 下流のdownsampling後: さらに少ない頂点
- GT→pred方向のchamfer距離が増加（被覆率低下）

### 3. finetuneの悪化は殻構造が原因ではない
- collapseあり+ft: 0.0002 → 0.0011（5.5倍悪化）
- collapseなし+ft: 0.0002 → 0.0009（4.5倍悪化）
- **finetuneの悪化は部分観測 (partial observation) が原因**
  - 点群は布の片面のみ → メッシュ全体を点群に寄せると歪む

## 結論
MC後処理単体では精度改善が限定的。改善には以下が必要:
1. **Surface decoderの精度向上** (主因: warp field予測の13mm誤差)
2. **Finetuneの改善**: 部分観測に対応した正則化

## 可視化
- collapse (finetune なし): `data/release/fan_release/fan_eval_v5_collapse/{0-15}_vis.html`
- collapse (finetune あり): `data/release/fan_release/fan_eval_v5_collapse_ft/{0-15}_vis.html`

## 日付
2026-02-27
