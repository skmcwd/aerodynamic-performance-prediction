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
except ModuleNotFoundError:  # pragma: no cover - 缺少 tqdm 时仍允许脚本运行
    class tqdm:  # type: ignore[no-redef]
        """简易进度条兜底类，用于无 tqdm 环境。"""

        def __init__(self, iterable: Iterable | None = None, **_: object) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])



@dataclass(frozen=True)
class Config:
    """集中管理锚点扰动数据集生成和验证参数。"""

    clustered_csv: Path = Path("data/CST_19ML_clustered.csv")
    reference_csv: Path = Path("data/known_airfoil_references.csv")
    synthetic_csv: Path = Path("data/anchor_perturbation_synthetic_labels.csv")
    summary_csv: Path = Path("data/anchor_perturbation_validation_summary.csv")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    anchor_strength: str = "strong"
    category_order: tuple[str, str, str] = ("Low-subsonic", "Transonic", "Supersonic")
    samples_per_anchor: int = 40
    random_seed: int = 20260603
    # 扰动半径按“到最近异类真实锚点距离”的比例设置，避免跨越外部锚点边界。
    cross_class_radius_fraction: float = 0.18
    # 同时用真实数据最近邻距离限制扰动大小，避免生成过远离数据流形的样本。
    manifold_radius_fraction: float = 0.65
    min_radius: float = 1e-4
    figure_dpi: int = 320
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "Low-subsonic": "#2A9D8F",
            "Transonic": "#E9C46A",
            "Supersonic": "#E76F51",
            "Synthetic": "#4B5563",
        }
    )


def setup_logging(config: Config) -> Path:
    """创建日志，文件名符合项目的日期-时间-脚本名规范。"""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 BOM 兼容 CSV 文件。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key).strip(): str(value).strip() for key, value in row.items()})
        return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出字典行 CSV。"""

    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mode_columns(rows: list[dict[str, str]]) -> list[str]:
    """自动识别 Mode 1 到 Mode 19 列。"""

    columns = [f"Mode {index}" for index in range(1, 20)]
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise KeyError(f"Missing mode columns: {missing}")
    return columns


def parse_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    """将指定列解析为浮点矩阵。"""

    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, column in enumerate(columns):
            try:
                matrix[row_index, col_index] = float(row[column])
            except (TypeError, ValueError):
                matrix[row_index, col_index] = np.nan
    return matrix


def robust_scale_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用中位数和 IQR 拟合 19-mode 标准化尺度。"""

    center = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def pairwise_distances(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """计算两个矩阵之间的欧氏距离。"""

    diff = matrix_a[:, None, :] - matrix_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def nearest_neighbor_distance(matrix: np.ndarray) -> np.ndarray:
    """计算每个真实样本到最近其他真实样本的距离，用于估计数据流形局部尺度。"""

    distances = pairwise_distances(matrix, matrix)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def build_anchor_rows(
    config: Config,
    clustered_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """筛选出在当前数据集中存在、质量有效、来源强度满足要求的真实锚点。"""

    clustered_by_name = {row["name"].strip().lower(): row for row in clustered_rows}
    anchors: list[dict[str, str]] = []
    for reference in reference_rows:
        if reference["anchor_strength"].strip() != config.anchor_strength:
            continue
        if reference["true_category"].strip() not in config.category_order:
            continue
        source_row = clustered_by_name.get(reference["airfoil_name"].strip().lower())
        if not source_row or source_row.get("quality_status") != "valid":
            continue

        anchor = dict(source_row)
        anchor["true_category"] = reference["true_category"].strip()
        anchor["anchor_strength"] = reference["anchor_strength"].strip()
        anchor["source_title"] = reference["source_title"].strip()
        anchors.append(anchor)
    return anchors


def compute_anchor_radii(
    config: Config,
    anchor_matrix: np.ndarray,
    anchor_labels: list[str],
    real_nn_median: float,
) -> np.ndarray:
    """为每个锚点计算安全扰动半径。"""

    distances = pairwise_distances(anchor_matrix, anchor_matrix)
    radii = np.zeros(len(anchor_labels), dtype=float)
    for index, label in enumerate(anchor_labels):
        other_class = np.array([item != label for item in anchor_labels], dtype=bool)
        if np.any(other_class):
            cross_distance = float(np.min(distances[index, other_class]))
        else:
            cross_distance = real_nn_median
        radii[index] = min(
            config.cross_class_radius_fraction * cross_distance,
            config.manifold_radius_fraction * real_nn_median,
        )
    return np.maximum(radii, config.min_radius)


def generate_perturbations(
    config: Config,
    anchor_rows: list[dict[str, str]],
    anchor_scaled: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """围绕真实锚点生成小扰动样本。"""

    rng = np.random.default_rng(config.random_seed)
    synthetic_scaled: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []

    for anchor_index, anchor in enumerate(tqdm(anchor_rows, desc="Generating perturbations", unit="anchor")):
        radius = radii[anchor_index]
        for sample_index in range(config.samples_per_anchor):
            direction = rng.normal(size=anchor_scaled.shape[1])
            direction_norm = np.linalg.norm(direction)
            if direction_norm <= 1e-12:
                direction[0] = 1.0
                direction_norm = 1.0
            direction = direction / direction_norm
            # 半径按 19 维球体体积均匀采样，避免样本过度集中在锚点中心。
            sample_radius = radius * (rng.random() ** (1.0 / anchor_scaled.shape[1]))
            sample = anchor_scaled[anchor_index] + sample_radius * direction
            synthetic_scaled.append(sample)
            metadata.append(
                {
                    "synthetic_name": f"{anchor['name']}_perturb_{sample_index + 1:03d}",
                    "source_anchor": anchor["name"],
                    "true_category": anchor["true_category"],
                    "anchor_predicted_category": anchor["category"],
                    "anchor_strength": anchor["anchor_strength"],
                    "perturb_radius_scaled": f"{sample_radius:.8f}",
                    "trust_radius_scaled": f"{radius:.8f}",
                    "source_title": anchor["source_title"],
                }
            )

    return np.vstack(synthetic_scaled), metadata


def validate_synthetic_samples(
    synthetic_scaled: np.ndarray,
    synthetic_metadata: list[dict[str, str]],
    anchor_scaled: np.ndarray,
    anchor_labels: list[str],
    anchor_names: list[str],
    real_scaled: np.ndarray,
    real_nn_threshold: float,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    """验证扰动样本是否仍处于同类锚点邻域，并且不明显离开真实数据流形。"""

    anchor_distances = pairwise_distances(synthetic_scaled, anchor_scaled)
    real_distances = pairwise_distances(synthetic_scaled, real_scaled)
    nearest_anchor_index = np.argmin(anchor_distances, axis=1)
    nearest_real_distance = np.min(real_distances, axis=1)

    validated_rows: list[dict[str, str]] = []
    same_category_count = 0
    manifold_count = 0
    source_nearest_count = 0

    for index, metadata in enumerate(synthetic_metadata):
        true_category = metadata["true_category"]
        nearest_label = anchor_labels[int(nearest_anchor_index[index])]
        nearest_name = anchor_names[int(nearest_anchor_index[index])]
        same_category = nearest_label == true_category
        source_nearest = nearest_name == metadata["source_anchor"]
        on_manifold = nearest_real_distance[index] <= real_nn_threshold
        same_category_count += int(same_category)
        source_nearest_count += int(source_nearest)
        manifold_count += int(on_manifold)

        row = dict(metadata)
        row.update(
            {
                "nearest_anchor": nearest_name,
                "nearest_anchor_true_category": nearest_label,
                "nearest_anchor_distance_scaled": f"{anchor_distances[index, nearest_anchor_index[index]]:.8f}",
                "nearest_real_distance_scaled": f"{nearest_real_distance[index]:.8f}",
                "same_true_category_neighborhood": str(same_category),
                "source_anchor_is_nearest": str(source_nearest),
                "within_real_manifold_threshold": str(on_manifold),
            }
        )
        validated_rows.append(row)

    total = float(len(validated_rows))
    summary = {
        "synthetic_count": total,
        "same_true_category_rate": same_category_count / total if total else float("nan"),
        "source_anchor_nearest_rate": source_nearest_count / total if total else float("nan"),
        "within_real_manifold_rate": manifold_count / total if total else float("nan"),
        "real_manifold_threshold_scaled": real_nn_threshold,
        "median_nearest_real_distance_scaled": float(np.median(nearest_real_distance)),
        "p95_nearest_real_distance_scaled": float(np.percentile(nearest_real_distance, 95)),
    }
    return validated_rows, summary


def inverse_scale(matrix: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """将标准化 19-mode 数据还原到原始 Mode 坐标。"""

    return matrix * scale + center


def append_mode_values(
    rows: list[dict[str, str]],
    mode_values: np.ndarray,
    columns: list[str],
) -> list[dict[str, str]]:
    """将生成的 19-mode 值追加到输出行。"""

    for row, values in zip(rows, mode_values, strict=True):
        for column, value in zip(columns, values, strict=True):
            row[column] = f"{value:.10g}"
    return rows


def pca_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 SVD 计算二维 PCA 投影。"""

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return projection, explained[:2]


def configure_plot_style() -> None:
    """配置英文 Times New Roman 绘图风格。"""

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_perturbation_pca(
    config: Config,
    anchor_scaled: np.ndarray,
    anchor_labels: list[str],
    synthetic_scaled: np.ndarray,
    synthetic_labels: list[str],
) -> Path:
    """绘制真实锚点与扰动样本在 19-mode PCA 空间中的位置。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figures_dir / "anchor_perturbation_pca.png"
    combined = np.vstack([anchor_scaled, synthetic_scaled])
    projection, explained = pca_projection(combined)
    anchor_projection = projection[: len(anchor_scaled)]
    synthetic_projection = projection[len(anchor_scaled) :]

    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=config.figure_dpi)
    for category in config.category_order:
        anchor_mask = np.array([label == category for label in anchor_labels])
        synth_mask = np.array([label == category for label in synthetic_labels])
        if np.any(synth_mask):
            ax.scatter(
                synthetic_projection[synth_mask, 0],
                synthetic_projection[synth_mask, 1],
                s=14,
                c=config.palette[category],
                alpha=0.26,
                edgecolors="none",
                label=f"{category} synthetic",
            )
        if np.any(anchor_mask):
            ax.scatter(
                anchor_projection[anchor_mask, 0],
                anchor_projection[anchor_mask, 1],
                s=95,
                c=config.palette[category],
                marker="*",
                edgecolors="#111827",
                linewidths=0.7,
                label=f"{category} anchors",
                zorder=5,
            )

    ax.set_title("Local Perturbations Around True Anchors", pad=12, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(True, color="#E5E7EB", linewidth=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Perturbation PCA figure saved to %s", output_path)
    return output_path


def write_summary(config: Config, summary: dict[str, float], category_counts: dict[str, int]) -> None:
    """写出验证摘要，便于报告引用。"""

    rows = [
        {"metric": key, "value": f"{value:.8f}" if isinstance(value, float) else str(value)}
        for key, value in summary.items()
    ]
    for category, count in category_counts.items():
        rows.append({"metric": f"synthetic_count_{category}", "value": str(count)})
    write_csv(config.summary_csv, rows)
    logging.info("Validation summary saved to %s", config.summary_csv)


def main() -> None:
    """主流程：读取锚点、局部扰动、几何邻域验证、输出 CSV 和图。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)

    clustered_rows = read_csv(config.clustered_csv)
    reference_rows = read_csv(config.reference_csv)
    valid_rows = [row for row in clustered_rows if row.get("quality_status") == "valid"]
    columns = mode_columns(valid_rows)

    real_matrix = parse_matrix(valid_rows, columns)
    finite_mask = np.all(np.isfinite(real_matrix), axis=1)
    real_matrix = real_matrix[finite_mask]
    center, scale = robust_scale_fit(real_matrix)
    real_scaled = (real_matrix - center) / scale
    real_nn = nearest_neighbor_distance(real_scaled)
    real_nn_median = float(np.median(real_nn))
    real_nn_threshold = float(np.percentile(real_nn, 95))

    anchor_rows = build_anchor_rows(config, clustered_rows, reference_rows)
    if not anchor_rows:
        raise ValueError("No valid strong anchors found in the clustered dataset.")
    anchor_matrix = parse_matrix(anchor_rows, columns)
    anchor_scaled = (anchor_matrix - center) / scale
    anchor_labels = [row["true_category"] for row in anchor_rows]
    anchor_names = [row["name"] for row in anchor_rows]
    anchor_counts = {category: anchor_labels.count(category) for category in config.category_order}
    logging.info("Strong anchor counts: %s", anchor_counts)

    missing_categories = [category for category, count in anchor_counts.items() if count == 0]
    if missing_categories:
        logging.warning(
            "No in-dataset strong anchors found for categories: %s. "
            "Synthetic data cannot provide true labels for these categories.",
            missing_categories,
        )

    radii = compute_anchor_radii(config, anchor_scaled, anchor_labels, real_nn_median)
    synthetic_scaled, metadata = generate_perturbations(config, anchor_rows, anchor_scaled, radii)
    validated_rows, summary = validate_synthetic_samples(
        synthetic_scaled,
        metadata,
        anchor_scaled,
        anchor_labels,
        anchor_names,
        real_scaled,
        real_nn_threshold,
    )
    synthetic_modes = inverse_scale(synthetic_scaled, center, scale)
    append_mode_values(validated_rows, synthetic_modes, columns)
    write_csv(config.synthetic_csv, validated_rows)
    logging.info("Synthetic labeled dataset saved to %s", config.synthetic_csv)

    synthetic_labels = [row["true_category"] for row in validated_rows]
    synthetic_counts = {category: synthetic_labels.count(category) for category in config.category_order}
    write_summary(config, summary, synthetic_counts)
    plot_perturbation_pca(config, anchor_scaled, anchor_labels, synthetic_scaled, synthetic_labels)

    logging.info("Synthetic category counts: %s", synthetic_counts)
    logging.info("Perturbation validation summary: %s", summary)


if __name__ == "__main__":
    main()
