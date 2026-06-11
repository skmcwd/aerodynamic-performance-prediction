# Development Log

## 2026-05-25

- 新增 `main.py` 中的翼型速度工况聚类流程：读取 CSV、剔除 Ma=1.5 异常样本、构造三工况气动性能特征、输出聚类结果。
- 聚类仅使用 Ma=0.25、Ma=0.734、Ma=1.5；Ma=2.0 仅用于统计占位异常，不参与模型输入。
- 新增根目录 `log/` 日志输出逻辑，日志文件按“当前日期-时间-代码文件名”命名。
- 新增 `figures/` 可视化输出逻辑，生成 PCA 聚类散点图和类别诊断图。
- 更新 `README.md`，说明聚类思路、异常处理策略、运行方式和输出文件。

## 2026-06-03

- 新增 `data/known_airfoil_references.csv`，整理低速、跨音速和超音速典型翼型锚点及权威来源链接。
- 新增 `scripts/validate_cluster_results.py`，对外部锚点与当前聚类结果进行比较，输出锚点比较表、混淆矩阵图和分类置信度分布图。
- 新增 `data/known_airfoil_reference_comparison.csv`、`figures/anchor_reference_confusion.png` 和 `figures/cluster_score_margin_distribution.png`。
- 新增 `reports/cluster_validation_report.md`，说明当前聚类标签更适合解释为“三工况相对性能偏好”，而非严格真实设计类别。
- 新增锚点扰动验证思路，初步生成低速和跨音速真标签扰动样本，并指出当时缺少强超音速锚点。

## 2026-06-09

- 新增 `scripts/prepare_supersonic_anchors.py`，生成并登记 8 个强超音速锚点，包括 double-wedge、biconvex 解析翼型和 NACA RM A53B02 坐标表数字化翼型。
- 增量更新 `data/known_airfoil_references.csv`，将新增超音速锚点记录为 `true_category=Supersonic`、`anchor_strength=strong`，并保留 NACA / NASA / OpenVSP 来源链接。
- 新增 `data/supersonic/*.csv` 和 `data/supersonic/supersonic_anchor_manifest.csv`，统一保存超音速锚点坐标及几何诊断。
- 新增 `scripts/build_recomputed_mode_basis.py`，从 `data/d_PV_20_coord/` 的 2152 个原始翼型坐标训练可复现 19 维 PCA/SVD 几何基。
- 新增 `data/mode_basis/recomputed_19mode_basis.npz`、`data/mode_basis/recomputed_mode_diagnostics.csv` 和 `data/recomputed_modes_all_airfoils.csv`。
- 重写 `scripts/generate_anchor_perturbation_dataset.py`，默认使用 `recomputed_mode_1..19` 进行强锚点局部扰动、验证和可视化。
- 新增 `data/supersonic_anchor_perturbation_synthetic_labels.csv` 和 `data/supersonic_anchor_perturbation_validation_summary.csv`；当前生成 1240 个合成样本，其中 `Supersonic=320`。
- 新增 PNG/PDF 双格式图像：`supersonic_anchor_geometries`、`recomputed_mode_explained_variance`、`supersonic_anchor_perturbation_pca`、`supersonic_anchor_validation_metrics`；PNG DPI 约为 420。
- 同步重跑 `scripts/validate_cluster_results.py`，更新 `data/known_airfoil_reference_comparison.csv` 并将锚点对比图 DPI 提升至约 420。
- 更新 `reports/anchor_perturbation_dataset_report.md`，形成可直接展示的超音速锚点扩展、双轨 19-mode 表征和扰动验证报告。
- 更新 `README.md`，补充三步复现命令、输出文件和方法边界。
- 进一步强化 `reports/anchor_perturbation_dataset_report.md` 的超音速数据来源说明，逐一解释 `data/supersonic` 中 8 个翼型的来源位置、解析公式、坐标表数字化流程、图题和表题。
