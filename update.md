# Development Log

## 2026-05-25

- 在 `main.py` 中新增翼型速度工况聚类完整流程：读取 CSV、剔除 Ma=1.5 异常样本、构造三工况气动性能特征、使用速度工况原型聚类并输出结果。
- 聚类仅使用 Ma=0.25、Ma=0.734、Ma=1.5；Ma=2.0 仅统计异常占位数量，不参与模型输入。
- 新增根目录 `log` 日志输出逻辑，日志文件按“当前日期-时间-代码文件名称”命名。
- 新增 `figures` 可视化输出逻辑，生成 PCA 聚类散点图和类别诊断图，图中文字使用英文并配置 Times New Roman。
- 在 `README.md` 中补充聚类思路、异常处理策略、运行方式和输出文件说明。
- 根据诊断图修正聚类特征，将聚类输入改为三工况相对性能分数和分数差值，使类别更聚焦“适用速度工况”而非整体性能强弱。
- 运行最终脚本生成结果：有效聚类样本 2024 个，其中 `Low-subsonic` 884 个、`Transonic` 527 个、`Supersonic` 613 个；Ma=1.5 占位异常样本 124 个被忽略。

## 2026-06-03

- 新增 `data/known_airfoil_references.csv`，整理低速、跨音速和超音速典型翼型锚点及权威来源链接。
- 新增 `scripts/validate_cluster_results.py`，自动对齐外部锚点与当前聚类结果，输出锚点比较表、混淆矩阵图和分类裕度分布图。
- 新增 `data/known_airfoil_reference_comparison.csv`、`figures/anchor_reference_confusion.png` 和 `figures/cluster_score_margin_distribution.png`。
- 新增 `reports/cluster_validation_report.md`，形成用于汇报的分类有效性验证报告，并指出当前标签更适合解释为“三工况相对性能偏好”而非真实设计类别。
- 新增 `scripts/generate_anchor_perturbation_dataset.py`，基于强真实锚点在 19-mode 空间内生成小扰动真标签样本，并验证同类锚点邻域与数据流形约束。
- 新增 `data/anchor_perturbation_synthetic_labels.csv`、`data/anchor_perturbation_validation_summary.csv` 和 `figures/anchor_perturbation_pca.png`。
- 新增 `reports/anchor_perturbation_dataset_report.md`，说明锚点微扰真标签数据集的可行性、结果和边界：当前可生成低速/跨音速合成真标签样本，但缺少强超音速锚点。
