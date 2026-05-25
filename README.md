# aerodynamic-performance-prediction
Aerodynamic performance prediction

引入transformer注意力机制进行特征提取，然后使用深度学习模型预测机翼（先是二维翼型，之后再考虑延申到三维）的气动力性能（升力与阻力系数）

data/CST_19ML_src.csv数据表中的数据，工况为：
Ma=0.25，α=2.832，Re=6.5E+6
Ma=0.734，α=2.832，Re=6.5E+6
Ma=1.5，α=2.832，Re=6.5E+6
Ma=2.0，α=2.832，Re=6.5E+6

## 翼型适用速度工况聚类方案

本项目当前使用 `main.py` 对 `data/CST_19ML_src.csv` 中的翼型进行无监督聚类，将可用样本划分为三类：`Low-subsonic`、`Transonic`、`Supersonic`，分别对应 Ma=0.25、Ma=0.734、Ma=1.5 的主要适用速度工况。

### 数据使用原则

1. 不使用 Ma=2.0 工况参与聚类。该工况中大量样本的仿真结果为 `Cd=0.1, Cl=0.12`，属于明显占位或错误结果。
2. 使用 Ma=0.25、Ma=0.734、Ma=1.5 三个工况的阻力系数和升力系数构造聚类特征。
3. Ma=0.734 数据中同时存在“阻力系数”和“新阻力系数”，代码默认使用“新阻力系数”作为跨音速阻力输入；如需改用原始阻力列，可在 `Config.mach_columns` 中将 `transonic` 的 `cd_col` 从 `7` 改为 `6`。
4. Ma=1.5 中若出现 `Cd=0.1, Cl=0.12` 的占位值，或经 MAD 鲁棒规则识别为异常点，则该翼型不参与三工况聚类，并写入 `data/CST_19ML_ignored_ma15.csv` 供复核。

### 特征与聚类方法

每个翼型先在三个马赫数下计算综合性能分数。分数由升力系数、阻力系数和升阻比共同构成：较高的 `Cl`、较低的 `Cd`、较高的 `L/D` 会得到更高分。为避免异常值主导结果，所有数值特征均使用中位数和四分位距进行鲁棒标准化。

随后将三个工况的相对性能分数和分数差值作为输入特征，采用速度工况原型聚类：Ma=0.25、Ma=0.734、Ma=1.5 分别作为 `Low-subsonic`、`Transonic`、`Supersonic` 的物理原型，翼型被分配到相对性能分数最高的工况类别。这里使用“相对分数”而不是原始绝对分数，是为了让聚类重点关注翼型更适合哪个速度工况，而不是只按总体气动性能强弱分组。

### 运行方式与输出

运行环境建议使用 Python 3.12，并安装：

```bash
pip install numpy matplotlib tqdm
```

执行：

```bash
python main.py
```

脚本会生成以下结果：

- `data/CST_19ML_clustered.csv`：保留原始数据，并填充有效样本的 `category`，同时新增簇编号、质量状态、忽略原因和三工况性能分数。
- `data/CST_19ML_ignored_ma15.csv`：记录因 Ma=1.5 异常而未参与聚类的翼型。
- `figures/airfoil_speed_clusters_pca.png`：三类翼型在 PCA 二维空间中的聚类分布图。
- `figures/airfoil_speed_cluster_summary.png`：各类别数量及三工况平均性能分数诊断图。
- `log/当前日期-时间-main.log`：记录数据读取、异常剔除、聚类过程和输出路径等信息。
