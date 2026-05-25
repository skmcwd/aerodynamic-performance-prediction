from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - 仅在缺少 tqdm 时兜底
    class tqdm:  # type: ignore[no-redef]
        """简易进度条兜底，保证没有 tqdm 时程序仍可运行。"""

        def __init__(self, iterable: Iterable | None = None, total: int | None = None, **_: object) -> None:
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable or [])

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def update(self, _: int = 1) -> None:
            return None

        def set_postfix(self, **_: object) -> None:
            return None

        def close(self) -> None:
            return None


@dataclass(frozen=True)
class CoefficientColumns:
    """记录一个马赫数工况下阻力系数、升力系数所在列。"""

    mach: float
    cd_col: int
    cl_col: int
    display_name: str


@dataclass(frozen=True)
class Config:
    """集中管理聚类流程中的路径、列号、异常剔除规则和绘图参数。"""

    input_csv: Path = Path("data/CST_19ML_src.csv")
    clustered_csv: Path = Path("data/CST_19ML_clustered.csv")
    ignored_csv: Path = Path("data/CST_19ML_ignored_ma15.csv")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    csv_encoding: str = "gb18030"
    output_encoding: str = "utf-8-sig"
    category_col: int = 1
    name_col: int = 0
    m20_cd_col: int = 2
    m20_cl_col: int = 3
    mach_columns: dict[str, CoefficientColumns] = field(
        default_factory=lambda: {
            "low": CoefficientColumns(0.25, 9, 10, "Ma 0.25"),
            # Ma=0.734 同时有“阻力系数”和“新阻力系数”，默认使用修正后的“新阻力系数”。
            "transonic": CoefficientColumns(0.734, 7, 8, "Ma 0.734"),
            "supersonic": CoefficientColumns(1.5, 4, 5, "Ma 1.5"),
        }
    )
    category_names: tuple[str, str, str] = ("Low-subsonic", "Transonic", "Supersonic")
    regime_keys: tuple[str, str, str] = ("low", "transonic", "supersonic")
    regime_display_names: tuple[str, str, str] = ("Ma 0.25", "Ma 0.734", "Ma 1.5")
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "Low-subsonic": "#2A9D8F",
            "Transonic": "#E9C46A",
            "Supersonic": "#E76F51",
            "Ignored": "#8D99AE",
        }
    )
    n_clusters: int = 3
    min_cd: float = 1e-8
    placeholder_cd: float = 0.1
    placeholder_cl: float = 0.12
    placeholder_atol: float = 1e-12
    supersonic_mad_threshold: float = 8.0
    lift_weight: float = 0.45
    drag_weight: float = 0.35
    lift_drag_weight: float = 0.20
    figure_dpi: int = 320


def setup_logging(config: Config) -> Path:
    """创建根目录 log 文件夹，并按“日期-时间-代码文件名”生成日志文件。"""

    config.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{Path(__file__).stem}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def read_airfoil_csv(config: Config) -> tuple[list[str], list[list[str]]]:
    """读取翼型数据，保留原始列顺序，方便写回新的分类结果。"""

    with config.input_csv.open("r", encoding=config.csv_encoding, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    column_count = len(header)
    fixed_rows: list[list[str]] = []
    for row in rows:
        if len(row) < column_count:
            row = row + [""] * (column_count - len(row))
        fixed_rows.append(row[:column_count])

    logging.info("Loaded %d airfoils and %d columns from %s", len(fixed_rows), column_count, config.input_csv)
    return header, fixed_rows


def parse_float_column(rows: list[list[str]], col: int) -> np.ndarray:
    """将指定列转为 float；解析失败的位置记为 NaN，后续由质量控制剔除。"""

    values = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        try:
            values[index] = float(row[col])
        except (IndexError, TypeError, ValueError):
            values[index] = np.nan
    return values


def load_coefficients(config: Config, rows: list[list[str]]) -> dict[str, dict[str, np.ndarray]]:
    """按 Config 中定义的列号提取 Ma 0.25、0.734、1.5 的 Cd 和 Cl。"""

    coefficients: dict[str, dict[str, np.ndarray]] = {}
    for key, columns in config.mach_columns.items():
        coefficients[key] = {
            "cd": parse_float_column(rows, columns.cd_col),
            "cl": parse_float_column(rows, columns.cl_col),
        }
    return coefficients


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """用中位数和 IQR 得到鲁棒中心与尺度，降低异常点对标准化的影响。"""

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    median = float(np.median(finite_values))
    q25, q75 = np.percentile(finite_values, [25, 75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(finite_values))
    if scale <= 1e-12:
        scale = 1.0
    return median, scale


def robust_standardize(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """仅用有效样本拟合尺度，再把所有样本变换到同一标准化空间。"""

    center, scale = robust_location_scale(values[fit_mask])
    return (values - center) / scale


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """对可能为负的 L/D 做平滑压缩，保留符号并降低极端值影响。"""

    return np.sign(values) * np.log1p(np.abs(values))


def detect_bad_mach20(config: Config, rows: list[list[str]]) -> int:
    """统计 Ma=2.0 中典型占位错误数量；该工况不参与聚类。"""

    cd = parse_float_column(rows, config.m20_cd_col)
    cl = parse_float_column(rows, config.m20_cl_col)
    bad_mask = np.isclose(cd, config.placeholder_cd, atol=config.placeholder_atol) & np.isclose(
        cl,
        config.placeholder_cl,
        atol=config.placeholder_atol,
    )
    return int(np.sum(bad_mask))


def detect_valid_airfoils(
    config: Config,
    coefficients: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """剔除不能用于三工况聚类的样本，重点过滤 Ma=1.5 的占位和鲁棒异常值。"""

    row_count = len(next(iter(coefficients.values()))["cd"])
    reasons: list[list[str]] = [[] for _ in range(row_count)]
    valid_mask = np.ones(row_count, dtype=bool)
    basic_invalid_mask = np.zeros(row_count, dtype=bool)

    for key in config.regime_keys:
        cd = coefficients[key]["cd"]
        cl = coefficients[key]["cl"]
        invalid = (~np.isfinite(cd)) | (~np.isfinite(cl)) | (cd <= config.min_cd)
        for index in np.where(invalid)[0]:
            reasons[index].append(f"invalid_{config.mach_columns[key].display_name}")
        valid_mask &= ~invalid
        basic_invalid_mask |= invalid

    supersonic_cd = coefficients["supersonic"]["cd"]
    supersonic_cl = coefficients["supersonic"]["cl"]
    placeholder_mask = np.isclose(
        supersonic_cd,
        config.placeholder_cd,
        atol=config.placeholder_atol,
    ) & np.isclose(
        supersonic_cl,
        config.placeholder_cl,
        atol=config.placeholder_atol,
    )

    for index in np.where(placeholder_mask)[0]:
        reasons[index].append("placeholder_Ma_1.5_Cd_0.1_Cl_0.12")

    basic_supersonic_valid = (
        np.isfinite(supersonic_cd)
        & np.isfinite(supersonic_cl)
        & (supersonic_cd > config.min_cd)
        & (~placeholder_mask)
    )
    lift_drag = supersonic_cl / np.clip(supersonic_cd, config.min_cd, None)
    outlier_metrics = np.column_stack(
        [
            np.log(np.clip(supersonic_cd, config.min_cd, None)),
            supersonic_cl,
            signed_log1p(lift_drag),
        ]
    )
    robust_outlier_mask = np.zeros(row_count, dtype=bool)

    for metric in outlier_metrics.T:
        fit_values = metric[basic_supersonic_valid]
        if fit_values.size == 0:
            continue
        median = float(np.median(fit_values))
        mad = float(np.median(np.abs(fit_values - median)))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(fit_values))
        if scale <= 1e-12:
            continue
        z_score = np.abs((metric - median) / scale)
        robust_outlier_mask |= basic_supersonic_valid & (z_score > config.supersonic_mad_threshold)

    for index in np.where(robust_outlier_mask)[0]:
        reasons[index].append("robust_outlier_Ma_1.5")

    valid_mask &= ~(placeholder_mask | robust_outlier_mask)
    reason_text = [";".join(item) for item in reasons]
    stats = {
        "invalid_basic": int(np.sum(basic_invalid_mask)),
        "ma15_placeholder": int(np.sum(placeholder_mask)),
        "ma15_robust_outlier": int(np.sum(robust_outlier_mask)),
        "valid": int(np.sum(valid_mask)),
        "ignored": int(row_count - np.sum(valid_mask)),
    }
    return valid_mask, reason_text, stats


def build_feature_matrix(
    config: Config,
    coefficients: dict[str, dict[str, np.ndarray]],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """构造聚类特征：保留每个翼型对三个速度工况的相对偏好。"""

    score_columns: list[np.ndarray] = []
    diagnostics: dict[str, np.ndarray] = {}

    for key in config.regime_keys:
        cd = coefficients[key]["cd"]
        cl = coefficients[key]["cl"]
        lift_drag = cl / np.clip(cd, config.min_cd, None)

        log_cd_z = robust_standardize(np.log(np.clip(cd, config.min_cd, None)), valid_mask)
        cl_z = robust_standardize(cl, valid_mask)
        lift_drag_z = robust_standardize(signed_log1p(lift_drag), valid_mask)

        score = (
            config.lift_weight * cl_z
            - config.drag_weight * log_cd_z
            + config.lift_drag_weight * lift_drag_z
        )
        score_columns.append(score)
        diagnostics[f"{key}_ld"] = lift_drag

    score_matrix = np.column_stack(score_columns)
    relative_scores = score_matrix - np.nanmean(score_matrix, axis=1, keepdims=True)
    score_differences = np.column_stack(
        [
            score_matrix[:, 0] - score_matrix[:, 1],
            score_matrix[:, 1] - score_matrix[:, 2],
            score_matrix[:, 2] - score_matrix[:, 0],
        ]
    )
    raw_features = np.column_stack([relative_scores, score_differences])
    feature_matrix = robust_scale_matrix(raw_features[valid_mask])
    return feature_matrix, score_matrix, diagnostics


def robust_scale_matrix(matrix: np.ndarray) -> np.ndarray:
    """对聚类特征矩阵逐列做鲁棒标准化。"""

    scaled = np.empty_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        center, scale = robust_location_scale(matrix[:, col])
        scaled[:, col] = (matrix[:, col] - center) / scale
    return scaled


def assign_regime_prototype_clusters(
    config: Config,
    valid_scores: np.ndarray,
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """按三种速度工况原型归簇，保证类别含义与 Ma 0.25/0.734/1.5 一一对应。"""

    relative_scores = valid_scores - valid_scores.mean(axis=1, keepdims=True)
    labels = np.empty(valid_scores.shape[0], dtype=int)
    for index in tqdm(range(valid_scores.shape[0]), desc="Regime clustering", unit="airfoil"):
        labels[index] = int(np.argmax(relative_scores[index]))

    cluster_to_category = dict(enumerate(config.category_names))
    cluster_score_means = np.zeros((config.n_clusters, len(config.regime_keys)), dtype=float)
    for cluster_id in range(config.n_clusters):
        cluster_mask = labels == cluster_id
        if not np.any(cluster_mask):
            raise ValueError(f"No airfoils were assigned to {config.category_names[cluster_id]}.")
        cluster_score_means[cluster_id] = valid_scores[cluster_mask].mean(axis=0)

    for cluster_id, category in cluster_to_category.items():
        logging.info(
            "Cluster %d (%s), mean scores=%s",
            cluster_id,
            category,
            np.array2string(cluster_score_means[cluster_id], precision=4),
        )
    return labels, cluster_to_category, cluster_score_means


def write_clustered_csv(
    config: Config,
    header: list[str],
    rows: list[list[str]],
    valid_mask: np.ndarray,
    reason_text: list[str],
    labels: np.ndarray,
    cluster_to_category: dict[int, str],
    score_matrix: np.ndarray,
) -> None:
    """输出带 category、质量状态、簇编号和三工况性能分数的新 CSV。"""

    config.clustered_csv.parent.mkdir(parents=True, exist_ok=True)
    output_header = header + [
        "cluster_id",
        "quality_status",
        "ignored_reason",
        "score_Ma_0.25",
        "score_Ma_0.734",
        "score_Ma_1.5",
        "best_score_regime",
    ]
    valid_indices = np.where(valid_mask)[0]
    row_to_cluster = dict(zip(valid_indices, labels, strict=True))

    with config.clustered_csv.open("w", encoding=config.output_encoding, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(output_header)

        for row_index, row in enumerate(rows):
            output_row = list(row)
            scores = score_matrix[row_index]
            if valid_mask[row_index]:
                cluster_id = int(row_to_cluster[row_index])
                category = cluster_to_category[cluster_id]
                output_row[config.category_col] = category
                quality_status = "valid"
                reason = ""
                best_regime = config.regime_display_names[int(np.argmax(scores))]
            else:
                cluster_id = -1
                quality_status = "ignored"
                reason = reason_text[row_index] or "invalid_for_three_regime_clustering"
                best_regime = ""

            writer.writerow(
                output_row
                + [
                    cluster_id,
                    quality_status,
                    reason,
                    f"{scores[0]:.8f}" if np.isfinite(scores[0]) else "",
                    f"{scores[1]:.8f}" if np.isfinite(scores[1]) else "",
                    f"{scores[2]:.8f}" if np.isfinite(scores[2]) else "",
                    best_regime,
                ]
            )

    logging.info("Clustered CSV saved to %s", config.clustered_csv)


def write_ignored_csv(
    config: Config,
    rows: list[list[str]],
    valid_mask: np.ndarray,
    reason_text: list[str],
    coefficients: dict[str, dict[str, np.ndarray]],
) -> None:
    """单独保存被忽略的 Ma=1.5 异常翼型，便于后续复核 CFD 结果。"""

    config.ignored_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.ignored_csv.open("w", encoding=config.output_encoding, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "ignored_reason", "Ma_1.5_Cd", "Ma_1.5_Cl"])
        for row_index in np.where(~valid_mask)[0]:
            writer.writerow(
                [
                    rows[row_index][config.name_col],
                    reason_text[row_index],
                    f"{coefficients['supersonic']['cd'][row_index]:.12g}",
                    f"{coefficients['supersonic']['cl'][row_index]:.12g}",
                ]
            )
    logging.info("Ignored airfoil list saved to %s", config.ignored_csv)


def configure_plot_style() -> None:
    """设置论文风格绘图参数，图中文字统一使用英文和 Times New Roman。"""

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def compute_pca_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 SVD 计算二维 PCA 投影，不依赖 sklearn。"""

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return projection, explained[:2]


def plot_pca_clusters(
    config: Config,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    cluster_to_category: dict[int, str],
) -> Path:
    """绘制二维 PCA 聚类散点图，观察三类翼型在特征空间中的分离情况。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figures_dir / "airfoil_speed_clusters_pca.png"
    projection, explained = compute_pca_projection(feature_matrix)

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=config.figure_dpi)
    for category in config.category_names:
        cluster_ids = [cluster_id for cluster_id, name in cluster_to_category.items() if name == category]
        if not cluster_ids:
            continue
        mask = labels == cluster_ids[0]
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=28,
            c=config.palette[category],
            label=f"{category} (n={int(np.sum(mask))})",
            alpha=0.80,
            edgecolors="white",
            linewidths=0.35,
        )
        center = projection[mask].mean(axis=0)
        ax.scatter(
            center[0],
            center[1],
            s=220,
            c=config.palette[category],
            marker="*",
            edgecolors="#1F2937",
            linewidths=0.8,
            zorder=5,
        )

    ax.set_title("Airfoil Speed-Regime Clusters", pad=16, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(True, color="#E5E7EB", linewidth=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("PCA cluster figure saved to %s", output_path)
    return output_path


def plot_cluster_summary(
    config: Config,
    labels: np.ndarray,
    cluster_to_category: dict[int, str],
    valid_scores: np.ndarray,
) -> Path:
    """绘制类别数量和三工况平均性能分数，检查簇到速度工况的映射是否合理。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figures_dir / "airfoil_speed_cluster_summary.png"

    ordered_cluster_ids = [
        next(cluster_id for cluster_id, category in cluster_to_category.items() if category == target_category)
        for target_category in config.category_names
    ]
    counts = np.array([np.sum(labels == cluster_id) for cluster_id in ordered_cluster_ids], dtype=int)
    mean_scores = np.vstack([valid_scores[labels == cluster_id].mean(axis=0) for cluster_id in ordered_cluster_ids])
    colors = [config.palette[category] for category in config.category_names]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=config.figure_dpi, gridspec_kw={"width_ratios": [0.9, 1.25]})

    axes[0].bar(config.category_names, counts, color=colors, edgecolor="#1F2937", linewidth=0.8)
    axes[0].set_title("Cluster Size", pad=12, fontweight="bold")
    axes[0].set_ylabel("Number of airfoils")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.9)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)
    axes[0].tick_params(axis="x", rotation=18)
    for index, count in enumerate(counts):
        axes[0].text(index, count + max(counts) * 0.015, str(count), ha="center", va="bottom", fontsize=12)

    image = axes[1].imshow(mean_scores, cmap="RdYlBu_r", aspect="auto")
    axes[1].set_title("Mean Regime Score", pad=12, fontweight="bold")
    axes[1].set_xticks(range(len(config.regime_display_names)))
    axes[1].set_xticklabels(config.regime_display_names)
    axes[1].set_yticks(range(len(config.category_names)))
    axes[1].set_yticklabels(config.category_names)
    for row_index in range(mean_scores.shape[0]):
        for col_index in range(mean_scores.shape[1]):
            axes[1].text(
                col_index,
                row_index,
                f"{mean_scores[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                color="#111827",
            )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Score")

    fig.suptitle("Aerodynamic Clustering Diagnostics", fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Cluster summary figure saved to %s", output_path)
    return output_path


def summarize_results(
    config: Config,
    labels: np.ndarray,
    cluster_to_category: dict[int, str],
    valid_count: int,
    ignored_count: int,
) -> None:
    """将最终类别数量写入日志，便于复现实验时快速检查结果。"""

    logging.info("Valid airfoils clustered: %d", valid_count)
    logging.info("Ignored airfoils: %d", ignored_count)
    for cluster_id, category in sorted(cluster_to_category.items()):
        logging.info("%s: %d", category, int(np.sum(labels == cluster_id)))


def main() -> None:
    """主流程：读取数据、质量控制、特征构造、速度原型聚类、保存结果和图像。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)

    header, rows = read_airfoil_csv(config)
    coefficients = load_coefficients(config, rows)
    bad_mach20_count = detect_bad_mach20(config, rows)
    logging.info("Ma=2.0 placeholder rows ignored by design: %d", bad_mach20_count)

    valid_mask, reason_text, quality_stats = detect_valid_airfoils(config, coefficients)
    logging.info("Quality statistics: %s", quality_stats)
    if quality_stats["valid"] < config.n_clusters:
        raise ValueError(
            f"Only {quality_stats['valid']} valid airfoils remain; "
            f"at least {config.n_clusters} are required for clustering."
        )

    feature_matrix, score_matrix, _ = build_feature_matrix(config, coefficients, valid_mask)
    valid_scores = score_matrix[valid_mask]
    labels, cluster_to_category, _ = assign_regime_prototype_clusters(config, valid_scores)

    write_clustered_csv(config, header, rows, valid_mask, reason_text, labels, cluster_to_category, score_matrix)
    write_ignored_csv(config, rows, valid_mask, reason_text, coefficients)
    plot_pca_clusters(config, feature_matrix, labels, cluster_to_category)
    plot_cluster_summary(config, labels, cluster_to_category, valid_scores)
    summarize_results(config, labels, cluster_to_category, quality_stats["valid"], quality_stats["ignored"])


if __name__ == "__main__":
    main()
