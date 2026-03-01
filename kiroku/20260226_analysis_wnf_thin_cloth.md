# 分析: GarmentNets (medor) は薄い布に不向きか？

## 結論
**部分的にYES**: WNF + marching cubes のパイプラインは薄い一枚布には本質的な制約がある。ただし、GarmentNets全体としては改善の余地がある。

## 問題の本質

### WNF (Winding Number Field) とは
- 3D空間の各点が「メッシュの内側にあるか」を示すスカラー場
- 閉じたメッシュの内側 → WNF ≈ 1、外側 → WNF ≈ 0
- iso=0.5でmarching cubesすることで表面メッシュを抽出

### 元のGarmentNets (Tshirt, Trousers) の場合
- Tshirtの正準形: **前身頃+後身頃の二重構造** → 薄いが閉じた体積がある
- WNFに有意な内部空間が存在（二枚の布の間の空間）
- Marching cubesが正常に表面を抽出可能

### fan_cloth（扇形一枚布）の場合
- 正準形: **単一面の開いたサーフェス**（boundary edges = 188本）
- 数学的に「内側」が定義不可能 → WNF ≈ 0 everywhere
- 厚み付けなし: WNF > 0.5 = 224 voxels / 2,097,152 (0.01%) → MC失敗
- 厚み付けあり(t=0.005): WNF > 0.5 = 15,278 voxels (0.73%) → MC成功するが「厚い殻」が生成

### 定量的なギャップ

| 項目 | GT (薄い布) | MC出力 (t=0.005) |
|------|-----------|-----------------|
| 頂点数 | 2,303 | 31,246 (13.6x) |
| z方向厚み | 6.7mm | ~13mm |
| メッシュ構造 | open surface | closed shell |
| MC→GT距離 | - | 3.4mm (NOCS) |

実際の評価結果 (fan_pipe_v5):
- **chamfer_mesh RMSE = 12.8mm** (布幅231mmの5.5%)
- 全サンプルでほぼ同一誤差 → **系統的バイアス**（形状再構成の失敗ではなく、表現の限界）

## Gradient Filterの分析
`wnf_to_mesh()` にはgradient filteringがあり、WNFの勾配が低い頂点を除去する。
- 理論上は殻の側面を除去できる
- **実際には効果が限定的**: フィルタ前3.4mm → フィルタ後3.5mm（改善なし）
- 薄いWNFでは勾配の高低差が不十分

## Finetuneが悪化する原因
- chamfer_mesh: 0.0002 → chamfer_mesh_opt: 0.0009 (4.5倍悪化)
- finetuneはWNF予測を入力点群に合わせて微調整するが
- MC出力が殻構造なので、点群（片面のみ）に合わせるとメッシュが歪む

## 改善提案

### A. 短期的改善（現アーキテクチャ内）

1. **MCメッシュの後処理: 殻→薄膜への変換**
   - MC出力の閉じた殻メッシュに対して、対応する上面・下面の頂点を平均
   - 法線方向にクラスタリングして一層に潰す
   - 最も簡単で効果的（推定: RMSE 12.8mm → 3-5mm）

2. **WNFグリッド解像度を上げる (128→256)**
   - 薄い構造をより正確に表現
   - 計算コスト4倍、メモリ8倍

3. **異なるiso値の使用**
   - iso=1.0〜1.5で中央面に近い表面を抽出可能
   - ただしネットワーク予測値との整合性に注意

### B. 中期的改善（アーキテクチャ修正）

4. **WNFの代わりにUDF (Unsigned Distance Field) を使用**
   - 開いた表面に対しても距離が定義できる
   - iso=0 で薄膜表面を直接抽出
   - ネットワークのdecoder部分のみ変更

5. **Point-to-surface mapping**
   - WNF/MC のパイプラインを廃止
   - canonicalization後のNOCS点群から直接メッシュを構築
   - Poisson surface reconstructionなど

### C. 長期的改善

6. **Occupancy Fieldを厚み付きで学習、推論時に中央面抽出**
   - 学習: 厚み付きWNF (現在の手法)
   - 推論: MCで殻を抽出 → 自動的に中央面に射影

## 推奨: 提案A-1（MC後処理）が最もコストパフォーマンスが高い

実装案:
```python
def collapse_shell_to_surface(mc_verts, mc_faces, gt_nocs_verts=None):
    """MC殻メッシュの上面・下面を平均して薄膜に変換"""
    # 1. Z方向の中央値を計算
    z_mid = np.median(mc_verts[:, 2])

    # 2. 上面 (z > z_mid) と下面 (z < z_mid) に分割
    upper = mc_verts[mc_verts[:, 2] >= z_mid]
    lower = mc_verts[mc_verts[:, 2] < z_mid]

    # 3. 各上面頂点に対して最近傍の下面頂点を見つけて平均
    from scipy.spatial import cKDTree
    lower_tree = cKDTree(lower[:, :2])  # xy平面で検索
    _, idx = lower_tree.query(upper[:, :2])

    # 4. 上面と対応する下面の中点を取る
    collapsed = (upper + lower[idx]) / 2
    return collapsed
```

## 日付
2026-02-26
