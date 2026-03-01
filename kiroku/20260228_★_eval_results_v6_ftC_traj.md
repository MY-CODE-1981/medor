# fan_eval_v6_ftC_traj 評価結果

## 概要
各テスト軌道の**最初のフレーム（frame 0 = 最も折り畳まれた状態）**を1つずつ評価。
16軌道 × 1フレーム = 16サンプル。

## 保存先
```
data/release/fan_release/fan_eval_v6_ftC_traj/
├── 0_vis.html   (trajectory_angle48_dist38)
├── 1_vis.html   (trajectory_angle48_dist40)
├── ...
└── 15_vis.html  (trajectory_angle48_dist75)
```

## 学習データセット: fan_cloth_v2

### データソース
- **センサデータ**: `sensor_output_fan_2cam_v7_back/` (v7_back カメラ)
- **頂点軌道**: `cloth_trajectory_X24Y23_fan_v7_back/unfold/`
- unfold軌道を使用: frame 0 = 折り畳み状態, frame 84 = 展開状態

### データ分割
| split | 軌道数 | フレーム数 | サンプル数 |
|-------|--------|-----------|-----------|
| train | 120 | 85/軌道 | 10,200 |
| val | 15 | 85/軌道 | 1,275 |
| test | 16 | 85/軌道 | 1,360 |
| **合計** | **133** (v7_back全軌道) | | **12,835** |

### 布メッシュ
- グリッド: 47列 × 49行 = 2,303頂点
- 布幅: 約231mm (NOCS空間 [0,1]^3 に正規化)

### WNF (Winding Number Field)
- 解像度: 128^3
- 厚み付け (thickness): 0.005 (NOCS単位)
- 法線方向に±0.005オフセットして閉じたボリュームを生成

### カメラパラメータ (v7_back)
- 位置: (-0.0025, -0.0523, 0.317)
- 焦点距離: FX=288.29, FY=271.33
- 画像中心: CX=312.0, CY=267.0
- 深度: uint16 PNG (mm単位, DEPTH_SCALE=1000)

### データ変換コマンド
```bash
python scripts/convert_fan_dataset.py --output_dir dataset/fan_cloth_v2
```

## 使用モデル
- **Pipeline**: `data/release/fan_release/fan_pipe_v6` (epoch=497, val_loss=0.0002)
- **Canonicalization**: `data/release/fan_release/fan_canon_v2`
- Surface decoder: MLP [128,512,512,256,3], num_surface_sample=12000, 500 epochs

## 実行コマンド
```bash
source activate_env.sh && WANDB_MODE=offline python garmentnets/eval_pipeline.py \
  --model_path data/release/fan_release/fan_pipe_v6 \
  --cloth_type fan_cloth --exp_name fan_eval_v6_ftC_traj \
  --log_dir data/release/fan_release --max_test_num 15 --tt_finetune \
  --laplacian_w 0.5 --edge_w 0.5 --opt_lr 0.0001 --opt_iter_total 50 \
  --traj_first
```

### コード変更点
`garmentnets/eval_pipeline.py` に以下を追加:
- `--traj_first`: 各軌道の先頭フレームのみ評価するフラグ
- `--frames_per_traj 85`: 1軌道あたりのフレーム数（デフォルト85）
- 内部で sample index = [0, 85, 170, ..., 1275] を生成し、それ以外をスキップ

## 各サンプルの結果

| # | sample | 軌道名 | chamfer_mesh | chamfer_pc |
|---|--------|--------|-------------|-----------|
| 0 | 0 | angle48_dist38 | 0.000111 | 0.001002 |
| 1 | 85 | angle48_dist40 | 0.000112 | 0.000956 |
| 2 | 170 | angle48_dist43 | 0.000120 | 0.000966 |
| 3 | 255 | angle48_dist45 | 0.000134 | 0.000963 |
| 4 | 340 | angle48_dist48 | 0.000153 | 0.001003 |
| 5 | 425 | angle48_dist50 | 0.000156 | 0.000810 |
| 6 | 510 | angle48_dist53 | 0.000169 | 0.000831 |
| 7 | 595 | angle48_dist55 | 0.000166 | 0.000847 |
| 8 | 680 | angle48_dist58 | 0.000177 | 0.000965 |
| 9 | 765 | angle48_dist60 | 0.000186 | 0.000922 |
| 10 | 850 | angle48_dist63 | 0.000200 | 0.000782 |
| 11 | 935 | angle48_dist65 | 0.000207 | 0.000777 |
| 12 | 1020 | angle48_dist68 | 0.000224 | 0.000830 |
| 13 | 1105 | angle48_dist70 | 0.000219 | 0.000926 |
| 14 | 1190 | angle48_dist73 | 0.000228 | 0.000866 |
| 15 | 1275 | angle48_dist75 | 0.000211 | 0.000830 |

## 集計

| 指標 | mean | std |
|------|------|-----|
| chamfer_mesh | 0.0002 | 0.0000 |
| chamfer_pc | 0.0009 | 0.0001 |
| **chamfer_mesh_opt** | **0.0001** | **0.0000** |
| **chamfer_pc_opt** | **0.0005** | **0.0000** |

## 旧評価との比較

旧評価 (`fan_eval_v6_ftC`) は `--max_test_num 14` で連続16サンプル（= 1軌道のframe 0〜15）を評価。
全サンプルがほぼ同じ折れ量だった。

| 評価方法 | chamfer_mesh | chamfer_pc | chamfer_mesh_opt | chamfer_pc_opt |
|---------|-------------|-----------|-----------------|---------------|
| 旧: 連続16フレーム (1軌道) | 0.0001 | 0.0010 | 0.0001 | 0.0006 |
| **新: 各軌道frame 0 (16軌道)** | **0.0002** | **0.0009** | **0.0001** | **0.0005** |

### 差分の分析
- **chamfer_mesh 0.0001→0.0002**: 旧評価は1軌道の連続フレーム（ほぼ同じ形状）で評価していたため見かけ上良かった。多様な軌道で評価するとやや悪化。
- **chamfer_pc_opt 0.0006→0.0005**: finetuneが多様な入力に対してもロバストに機能している。
- 多様な折れ量での評価がより信頼性の高い指標。

## RMSE (mm) 換算

| 指標 | 旧 (1軌道) | 新 (16軌道) |
|------|----------|----------|
| chamfer_mesh_opt | 10.0 mm | **10.0 mm** |
| chamfer_pc_opt | 24.5 mm | **22.4 mm** |

## v5ベースラインからの改善 (新評価基準)

| 指標 | v5 (旧評価) | v6+ftC (新評価) | 改善率 |
|------|-----------|---------------|-------|
| chamfer_mesh_opt | 0.0009 (30.0mm) | **0.0001 (10.0mm)** | **67%** |
| chamfer_pc_opt | 0.0008 (28.3mm) | **0.0005 (22.4mm)** | **21%** |
