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
except ModuleNotFoundError:  # pragma: no cover - 仅在缺少 tqdm 时启用兜底
    class tqdm:  # type: ignore[no-redef]
        """简易进度条兜底类，保证脚本在无 tqdm 环境中仍可运行。"""

        def __init__(self, iterable: Iterable | None = None, **_: object) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])


@dataclass(frozen=True)
class Config:
    """集中管理锚点扰动数据集生成、验证和可视化参数。"""

    mode_csv: Path = Path("data/recomputed_modes_all_airfoils.csv")
    synthetic_csv: Path = Path("data/supersonic_anchor_perturbation_synthetic_labels.csv")
    summary_csv: Path = Path("data/supersonic_anchor_perturbation_validation_summary.csv")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    mode_prefix: str = "recomputed_mode_"
    n_modes: int = 19
    anchor_strength: str = "strong"
    category_order: tuple[str, str, str] = ("Low-subsonic", "Transonic", "Supersonic")
    samples_per_anchor: int = 40
    random_seed: int = 20260609
    # 半径同时受异类边界和真实流形尺度约束；这里比旧脚本更保守，避免外部超音速锚点越界。
    cross_class_radius_fraction: float = 0.10
    manifold_radius_fraction: float = 0.45
    min_radius: float = 1e-5
    validation_rate_floor: float = 0.95
    figure_dpi: int = 420
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "Low-subsonic": "#2A9D8F",
            "Transonic": "#E9C46A",
            "Supersonic": "#E76F51",
            "Real": "#CBD5E1",
        }
    )


def setup_logging(config: Config) -> Path:
    """创建日志文件，文件名符合“日期-时间-脚本名”的项目规范。"""

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
        return [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    """写出字典表 CSV。"""

    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mode_columns(config: Config, rows: list[dict[str, str]]) -> list[str]:
    """自动识别 recomputed_mode_1 到 recomputed_mode_19。"""

    columns = [f"{config.mode_prefix}{index}" for index in range(1, config.n_modes + 1)]
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise KeyError(f"Missing recomputed mode columns: {missing}")
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
    """用中位数和 IQR 拟合鲁棒标准化尺度。"""

    center = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def pairwise_distances(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """计算两个矩阵行向量之间的欧氏距离。"""

    diff = matrix_a[:, None, :] - matrix_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def nearest_neighbor_distance(matrix: np.ndarray) -> np.ndarray:
    """计算每个真实样本到最近其他真实样本的距离。"""

    distances = pairwise_distances(matrix, matrix)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def filter_real_rows(rows: list[dict[str, str]], matrix: np.ndarray) -> tuple[list[dict[str, str]], np.ndarray]:
    """保留 recomputed mode 完整有限的真实/外部几何样本。"""

    finite_mask = np.all(np.isfinite(matrix), axis=1)
    return [row for row, keep in zip(rows, finite_mask, strict=True) if keep], matrix[finite_mask]


def build_anchor_rows(config: Config, real_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """筛选有真实类别和强来源证据的锚点。"""

    anchors: list[dict[str, str]] = []
    for row in real_rows:
        if row.get("anchor_strength", "").strip() != config.anchor_strength:
            continue
        if row.get("true_category", "").strip() not in config.category_order:
            continue
        # 原始数据集锚点必须通过 Ma=1.5 质量控制；外部超音速锚点没有 CFD 质量字段。
        if row.get("geometry_group") == "original_dataset" and row.get("quality_status") != "valid":
            continue
        anchors.append(row)
    return anchors


def compute_anchor_radii(
    config: Config,
    anchor_scaled: np.ndarray,
    anchor_labels: list[str],
    real_nn_median: float,
) -> np.ndarray:
    """为每个强锚点计算安全扰动半径。"""

    distances = pairwise_distances(anchor_scaled, anchor_scaled)
    radii = np.zeros(len(anchor_labels), dtype=float)
    for index, label in enumerate(anchor_labels):
        other_class = np.array([item != label for item in anchor_labels], dtype=bool)
        cross_distance = float(np.min(distances[index, other_class])) if np.any(other_class) else real_nn_median
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
    """围绕真实强锚点生成小半径扰动样本。"""

    rng = np.random.default_rng(config.random_seed)
    synthetic_scaled: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []

    for anchor_index, anchor in enumerate(tqdm(anchor_rows, desc="Generating perturbations", unit="anchor")):
        radius = radii[anchor_index]
        for sample_index in range(config.samples_per_anchor):
            direction = rng.normal(size=anchor_scaled.shape[1])
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                direction[0] = 1.0
                norm = 1.0
            direction = direction / norm
            # 在 19 维球体内按体积均匀采样，避免样本过度集中在中心。
            sample_radius = radius * (rng.random() ** (1.0 / anchor_scaled.shape[1]))
            sample = anchor_scaled[anchor_index] + sample_radius * direction
            synthetic_scaled.append(sample)
            metadata.append(
                {
                    "synthetic_name": f"{anchor['name']}_perturb_{sample_index + 1:03d}",
                    "source_anchor": anchor["name"],
                    "source_geometry_group": anchor.get("geometry_group", ""),
                    "true_category": anchor["true_category"],
                    "anchor_strength": anchor["anchor_strength"],
                    "perturb_radius_scaled": f"{sample_radius:.8f}",
                    "trust_radius_scaled": f"{radius:.8f}",
                    "source_title": anchor.get("source_title", ""),
                    "source_url": anchor.get("source_url", ""),
                    "notes": anchor.get("notes", ""),
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
    """验证扰动样本是否仍处于同类强锚点邻域和真实几何流形附近。"""

    anchor_distances = pairwise_distances(synthetic_scaled, anchor_scaled)
    real_distances = pairwise_distances(synthetic_scaled, real_scaled)
    nearest_anchor_index = np.argmin(anchor_distances, axis=1)
    nearest_real_distance = np.min(real_distances, axis=1)

    validated_rows: list[dict[str, str]] = []
    same_category_count = 0
    source_nearest_count = 0
    manifold_count = 0

    for index, metadata in enumerate(synthetic_metadata):
        true_category = metadata["true_category"]
        nearest_idx = int(nearest_anchor_index[index])
        nearest_label = anchor_labels[nearest_idx]
        nearest_name = anchor_names[nearest_idx]
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
                "nearest_anchor_distance_scaled": f"{anchor_distances[index, nearest_idx]:.8f}",
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
    """将标准化 recomputed mode 还原到原始 PCA 坐标尺度。"""

    return matrix * scale + center


def append_mode_values(rows: list[dict[str, str]], values: np.ndarray, columns: list[str]) -> None:
    """把生成样本的 recomputed mode 写入输出行。"""

    for row, mode_values in zip(rows, values, strict=True):
        for column, value in zip(columns, mode_values, strict=True):
            row[column] = f"{float(value):.12g}"


def pca_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 SVD 计算二维 PCA 投影，用于可视化。"""

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return projection, explained[:2]


def configure_plot_style() -> None:
    """配置 Times New Roman 英文绘图风格。"""

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_perturbation_pca(
    config: Config,
    real_scaled: np.ndarray,
    anchor_scaled: np.ndarray,
    anchor_labels: list[str],
    synthetic_scaled: np.ndarray,
    synthetic_labels: list[str],
) -> None:
    """绘制真实样本、强锚点和扰动样本在二维 PCA 空间中的位置。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    combined = np.vstack([real_scaled, anchor_scaled, synthetic_scaled])
    projection, explained = pca_projection(combined)
    real_projection = projection[: len(real_scaled)]
    anchor_projection = projection[len(real_scaled) : len(real_scaled) + len(anchor_scaled)]
    synthetic_projection = projection[len(real_scaled) + len(anchor_scaled) :]

    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=config.figure_dpi)
    ax.scatter(
        real_projection[:, 0],
        real_projection[:, 1],
        s=8,
        c=config.palette["Real"],
        alpha=0.22,
        edgecolors="none",
        label="Real geometry manifold",
    )
    for category in config.category_order:
        anchor_mask = np.array([label == category for label in anchor_labels])
        synth_mask = np.array([label == category for label in synthetic_labels])
        if np.any(synth_mask):
            ax.scatter(
                synthetic_projection[synth_mask, 0],
                synthetic_projection[synth_mask, 1],
                s=14,
                c=config.palette[category],
                alpha=0.30,
                edgecolors="none",
                label=f"{category} synthetic",
            )
        if np.any(anchor_mask):
            ax.scatter(
                anchor_projection[anchor_mask, 0],
                anchor_projection[anchor_mask, 1],
                s=92,
                c=config.palette[category],
                marker="*",
                edgecolors="#111827",
                linewidths=0.7,
                label=f"{category} anchors",
                zorder=5,
            )

    ax.set_title("Anchor-Guided Local Perturbations in Recomputed Mode Space", fontweight="bold", pad=10)
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        output = config.figures_dir / f"supersonic_anchor_perturbation_pca.{suffix}"
        fig.savefig(output, dpi=config.figure_dpi if suffix == "png" else None, bbox_inches="tight")
        logging.info("Saved perturbation PCA figure: %s", output)
    plt.close(fig)


def plot_validation_metrics(config: Config, summary: dict[str, float]) -> None:
    """绘制扰动样本验证指标柱状图。"""

    configure_plot_style()
    metrics = [
        ("Same Category", summary["same_true_category_rate"]),
        ("Source Nearest", summary["source_anchor_nearest_rate"]),
        ("Within Manifold", summary["within_real_manifold_rate"]),
    ]
    labels = [item[0] for item in metrics]
    values = np.array([item[1] for item in metrics], dtype=float) * 100.0
    colors = ["#2A9D8F", "#457B9D", "#E76F51"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=config.figure_dpi)
    bars = ax.bar(labels, values, color=colors, edgecolor="#111827", linewidth=0.4)
    ax.axhline(config.validation_rate_floor * 100.0, color="#6B7280", lw=1.2, ls="--", label="Acceptance floor")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Synthetic Label Validation Metrics", fontweight="bold", pad=10)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1.5, 103),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        output = config.figures_dir / f"supersonic_anchor_validation_metrics.{suffix}"
        fig.savefig(output, dpi=config.figure_dpi if suffix == "png" else None, bbox_inches="tight")
        logging.info("Saved validation metrics figure: %s", output)
    plt.close(fig)


def write_summary(
    config: Config,
    summary: dict[str, float],
    anchor_counts: dict[str, int],
    synthetic_counts: dict[str, int],
) -> None:
    """写出验证摘要，便于报告引用。"""

    rows = [
        {"metric": key, "value": f"{value:.8f}" if isinstance(value, float) else str(value)}
        for key, value in summary.items()
    ]
    for category in config.category_order:
        rows.append({"metric": f"strong_anchor_count_{category}", "value": str(anchor_counts.get(category, 0))})
    for category in config.category_order:
        rows.append({"metric": f"synthetic_count_{category}", "value": str(synthetic_counts.get(category, 0))})
    write_csv(config.summary_csv, rows, ["metric", "value"])
    logging.info("Saved validation summary to %s", config.summary_csv)


def ensure_acceptance(config: Config, summary: dict[str, float], anchor_counts: dict[str, int]) -> None:
    """按研究验证要求检查关键通过标准。"""

    if anchor_counts.get("Supersonic", 0) < 6:
        raise ValueError("Fewer than 6 strong Supersonic anchors are available.")
    for key in ("same_true_category_rate", "within_real_manifold_rate"):
        if summary[key] < config.validation_rate_floor:
            raise ValueError(f"{key}={summary[key]:.4f} is below {config.validation_rate_floor:.2f}.")


def main() -> None:
    """主流程：读取 recomputed mode、生成锚点扰动、验证并输出 CSV 与图。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)
    if not config.mode_csv.exists():
        raise FileNotFoundError(
            f"{config.mode_csv} not found. Run scripts/prepare_supersonic_anchors.py and "
            "scripts/build_recomputed_mode_basis.py first."
        )

    all_rows = read_csv(config.mode_csv)
    columns = mode_columns(config, all_rows)
    all_matrix = parse_matrix(all_rows, columns)
    real_rows, real_matrix = filter_real_rows(all_rows, all_matrix)
    center, scale = robust_scale_fit(real_matrix)
    real_scaled = (real_matrix - center) / scale
    real_nn = nearest_neighbor_distance(real_scaled)
    real_nn_median = float(np.median(real_nn))
    real_nn_threshold = float(np.percentile(real_nn, 95))

    anchor_rows = build_anchor_rows(config, real_rows)
    if not anchor_rows:
        raise ValueError("No valid strong anchors found.")
    anchor_matrix = parse_matrix(anchor_rows, columns)
    anchor_scaled = (anchor_matrix - center) / scale
    anchor_labels = [row["true_category"] for row in anchor_rows]
    anchor_names = [row["name"] for row in anchor_rows]
    anchor_counts = {category: anchor_labels.count(category) for category in config.category_order}
    logging.info("Strong anchor counts: %s", anchor_counts)

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
    output_fields = [
        "synthetic_name",
        "source_anchor",
        "source_geometry_group",
        "true_category",
        "anchor_strength",
        "perturb_radius_scaled",
        "trust_radius_scaled",
        "source_title",
        "source_url",
        "notes",
        "nearest_anchor",
        "nearest_anchor_true_category",
        "nearest_anchor_distance_scaled",
        "nearest_real_distance_scaled",
        "same_true_category_neighborhood",
        "source_anchor_is_nearest",
        "within_real_manifold_threshold",
        *columns,
    ]
    write_csv(config.synthetic_csv, validated_rows, output_fields)
    logging.info("Saved synthetic labeled dataset to %s", config.synthetic_csv)

    synthetic_labels = [row["true_category"] for row in validated_rows]
    synthetic_counts = {category: synthetic_labels.count(category) for category in config.category_order}
    write_summary(config, summary, anchor_counts, synthetic_counts)
    ensure_acceptance(config, summary, anchor_counts)
    plot_perturbation_pca(config, real_scaled, anchor_scaled, anchor_labels, synthetic_scaled, synthetic_labels)
    plot_validation_metrics(config, summary)
    logging.info("Synthetic category counts: %s", synthetic_counts)
    logging.info("Perturbation validation summary: %s", summary)


if __name__ == "__main__":
    main()
