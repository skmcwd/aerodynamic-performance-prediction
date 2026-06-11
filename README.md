# Aerodynamic Performance Prediction

本项目面向二维翼型的速度适用性分类与气动性能预测。当前数据集包含翼型名称、多个 Mach 工况下的阻力/升力系数，以及 19 维几何 Mode 特征。现有聚类流程将翼型划分为 `Low-subsonic`、`Transonic`、`Supersonic` 三类，分别对应 Ma=0.25、Ma=0.734、Ma=1.5 的主要适用速度工况。

## 1. 原始聚类流程

`main.py` 使用 `data/CST_19ML_src.csv` 中的 Ma=0.25、Ma=0.734、Ma=1.5 三个工况进行速度适用性聚类。Ma=2.0 工况中大量样本存在 `Cd=0.1, Cl=0.12` 的占位错误，因此不参与聚类。Ma=1.5 中的少量异常值会通过占位值检测和 MAD 鲁棒规则剔除。

运行：

```powershell
& 'D:\Programs\Anaconda\envs\torch290\python.exe' main.py
```

主要输出：

- `data/CST_19ML_clustered.csv`：填充聚类类别、质量状态、三工况性能分数。
- `data/CST_19ML_ignored_ma15.csv`：记录 Ma=1.5 异常样本。
- `figures/airfoil_speed_clusters_pca.png`：聚类 PCA 可视化。
- `figures/airfoil_speed_cluster_summary.png`：类别数量与性能分数摘要。

## 2. 真实锚点与分类有效性验证

仅靠无监督聚类不能证明三类标签就是“真实设计类别”。因此项目引入 `data/known_airfoil_references.csv`，记录低速、跨音速、超音速翼型的外部权威锚点来源。已有验证报告位于：

- [cluster_validation_report.md](reports/cluster_validation_report.md)
- [anchor_perturbation_dataset_report.md](reports/anchor_perturbation_dataset_report.md)

## 3. 超音速强锚点扩展

此前数据集中没有可用的强超音速锚点。当前新增流程引入 8 个 `Supersonic` 强锚点：

- `double_wedge_tc04/tc06/tc08`
- `biconvex_tc04/tc06/tc08`
- `naca_rm_a53b02_304b`
- `naca_rm_a53b02_504b`

其中 double-wedge 与 biconvex 由 NACA / NASA 权威解析定义生成；NACA RM A53B02 样本由报告中的超音速实验翼型坐标表数字化得到。所有坐标统一保存为项目通用的 `x,y` CSV 格式：

- `data/supersonic/*.csv`
- `data/supersonic/supersonic_anchor_manifest.csv`

运行：

```powershell
& 'D:\Programs\Anaconda\envs\torch290\python.exe' scripts\prepare_supersonic_anchors.py
```

## 4. 可复现 19-mode 几何基

新增超音速翼型没有历史 `Mode 1..19`，且原始降维基未保存在仓库中。因此项目新增一套可复现的 PCA/SVD 几何基，输出为 `recomputed_mode_1..19`。这不会覆盖历史 Mode，而是作为外部锚点验证的并行表征。

运行：

```powershell
& 'D:\Programs\Anaconda\envs\torch290\python.exe' scripts\build_recomputed_mode_basis.py
```

主要输出：

- `data/mode_basis/recomputed_19mode_basis.npz`
- `data/mode_basis/recomputed_mode_diagnostics.csv`
- `data/recomputed_modes_all_airfoils.csv`
- `figures/recomputed_mode_explained_variance.png`
- `figures/recomputed_mode_explained_variance.pdf`

## 5. 锚点扰动数据集

`scripts/generate_anchor_perturbation_dataset.py` 基于强锚点在 `recomputed_mode_*` 空间内生成小半径扰动样本。扰动半径同时受最近异类锚点距离和真实几何流形尺度约束，避免生成明显越界样本。

运行：

```powershell
& 'D:\Programs\Anaconda\envs\torch290\python.exe' scripts\generate_anchor_perturbation_dataset.py
```

当前验证结果：

- 强锚点数：`Low-subsonic=13`、`Transonic=10`、`Supersonic=8`
- 合成样本数：`Low-subsonic=520`、`Transonic=400`、`Supersonic=320`
- 总合成样本数：1240
- 同类邻域保持率：100.0%
- 真实几何流形内比例：100.0%

主要输出：

- `data/supersonic_anchor_perturbation_synthetic_labels.csv`
- `data/supersonic_anchor_perturbation_validation_summary.csv`
- `figures/supersonic_anchor_geometries.png`
- `figures/supersonic_anchor_perturbation_pca.png`
- `figures/supersonic_anchor_validation_metrics.png`

所有新增 PNG 图像均以 Times New Roman 英文字体绘制，DPI 约为 420，并同步输出 PDF 版本。

## 6. 方法边界

- `category` 聚类标签更准确地解释为“三工况相对性能偏好”，不必然等同于真实设计类别。
- `true_category` 来自外部权威锚点证据，适合用于分类合理性验证。
- `recomputed_mode_*` 是新增可复现几何表征，不应直接覆盖历史 `Mode 1..19`。
- 扰动样本不是 CFD 真值，若用于最终气动性能建模，仍需对关键样本做 CFD 或实验复核。
