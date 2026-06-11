from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
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
        """简易 tqdm 兜底类，用于无 tqdm 环境。"""

        def __init__(self, iterable: Iterable | None = None, **_: object) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])


@dataclass(frozen=True)
class Config:
    """集中管理验证脚本的输入、输出、类别顺序和绘图参数。"""

    clustered_csv: Path = Path("data/CST_19ML_clustered.csv")
    reference_csv: Path = Path("data/known_airfoil_references.csv")
    comparison_csv: Path = Path("data/known_airfoil_reference_comparison.csv")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    category_order: tuple[str, str, str] = ("Low-subsonic", "Transonic", "Supersonic")
    score_columns: tuple[str, str, str] = ("score_Ma_0.25", "score_Ma_0.734", "score_Ma_1.5")
    palette: dict[str, str] | None = None
    figure_dpi: int = 420

    def colors(self) -> dict[str, str]:
        """返回固定配色，保证所有验证图风格一致。"""

        return self.palette or {
            "Low-subsonic": "#2A9D8F",
            "Transonic": "#E9C46A",
            "Supersonic": "#E76F51",
            "Mismatch": "#8D99AE",
        }


def setup_logging(config: Config) -> Path:
    """创建日志文件，文件名符合“当前日期-时间-代码文件名称”的项目约定。"""

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
    """读取 UTF-8 BOM 兼容 CSV，并返回字典行。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key).strip(): str(value).strip() for key, value in row.items()})
        return rows


def parse_float(value: str) -> float:
    """安全解析浮点数，解析失败返回 NaN。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def score_margin(row: dict[str, str], config: Config) -> float:
    """计算最高工况分数与次高工况分数的差值，差值越大说明类别越稳定。"""

    scores = np.array([parse_float(row[column]) for column in config.score_columns], dtype=float)
    if np.sum(np.isfinite(scores)) < 2:
        return float("nan")
    ordered = np.sort(scores[np.isfinite(scores)])
    return float(ordered[-1] - ordered[-2])


def compare_references(
    config: Config,
    clustered_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """将外部真实锚点与当前聚类结果按翼型名称对齐。"""

    clustered_by_name = {row["name"].strip().lower(): row for row in clustered_rows}
    comparison_rows: list[dict[str, str]] = []

    for reference in tqdm(reference_rows, desc="Comparing anchors", unit="anchor"):
        name = reference["airfoil_name"].strip().lower()
        predicted = clustered_by_name.get(name)
        in_dataset = predicted is not None
        quality_status = predicted.get("quality_status", "") if predicted else ""
        predicted_category = predicted.get("category", "") if predicted else ""
        margin = score_margin(predicted, config) if predicted and quality_status == "valid" else float("nan")
        true_category = reference["true_category"]
        is_comparable = in_dataset and quality_status == "valid" and reference["anchor_strength"] != "archetype"
        is_match = is_comparable and predicted_category == true_category

        comparison_rows.append(
            {
                "airfoil_name": reference["airfoil_name"],
                "true_category": true_category,
                "anchor_strength": reference["anchor_strength"],
                "predicted_category": predicted_category,
                "quality_status": quality_status if in_dataset else "not_in_dataset",
                "is_comparable": str(is_comparable),
                "is_match": str(is_match) if is_comparable else "",
                "score_margin": f"{margin:.8f}" if np.isfinite(margin) else "",
                "best_score_regime": predicted.get("best_score_regime", "") if predicted else "",
                "evidence_basis": reference["evidence_basis"],
                "source_title": reference["source_title"],
                "source_url": reference["source_url"],
                "notes": reference["notes"],
            }
        )
    return comparison_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """写出字典行 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_confusion_matrix(
    config: Config,
    comparison_rows: list[dict[str, str]],
    strength_filter: set[str],
) -> np.ndarray:
    """构造外部锚点真实类别与当前预测类别的混淆矩阵。"""

    category_index = {category: index for index, category in enumerate(config.category_order)}
    matrix = np.zeros((len(config.category_order), len(config.category_order)), dtype=int)
    for row in comparison_rows:
        if row["anchor_strength"] not in strength_filter:
            continue
        if row["is_comparable"] != "True":
            continue
        true_category = row["true_category"]
        predicted_category = row["predicted_category"]
        if true_category in category_index and predicted_category in category_index:
            matrix[category_index[true_category], category_index[predicted_category]] += 1
    return matrix


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


def plot_anchor_confusion(config: Config, comparison_rows: list[dict[str, str]]) -> Path:
    """绘制外部锚点混淆矩阵，展示设计意图标签与当前聚类标签的一致性。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figures_dir / "anchor_reference_confusion.png"
    matrix = build_confusion_matrix(config, comparison_rows, {"strong", "contextual"})

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=config.figure_dpi)
    image = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_title("External Anchor Agreement", pad=14, fontweight="bold")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Reference category")
    ax.set_xticks(range(len(config.category_order)))
    ax.set_xticklabels(config.category_order, rotation=22, ha="right")
    ax.set_yticks(range(len(config.category_order)))
    ax.set_yticklabels(config.category_order)

    row_sums = matrix.sum(axis=1)
    text_threshold = matrix.max() * 0.55 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            count = matrix[row_index, col_index]
            percent = 100.0 * count / row_sums[row_index] if row_sums[row_index] else 0.0
            label = f"{count}\n{percent:.0f}%" if count else "0"
            text_color = "white" if count > text_threshold else "#111827"
            ax.text(col_index, row_index, label, ha="center", va="center", color=text_color, fontsize=12)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Anchor count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Anchor confusion figure saved to %s", output_path)
    return output_path


def plot_margin_distribution(config: Config, clustered_rows: list[dict[str, str]]) -> Path:
    """绘制类别分配裕度分布，用于评估当前标签对小扰动的稳健性。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.figures_dir / "cluster_score_margin_distribution.png"
    colors = config.colors()

    grouped_margins: list[list[float]] = []
    for category in config.category_order:
        values = [
            score_margin(row, config)
            for row in clustered_rows
            if row.get("quality_status") == "valid" and row.get("category") == category
        ]
        grouped_margins.append([value for value in values if np.isfinite(value)])

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=config.figure_dpi)
    box = ax.boxplot(
        grouped_margins,
        tick_labels=config.category_order,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, category in zip(box["boxes"], config.category_order, strict=True):
        patch.set_facecolor(colors[category])
        patch.set_edgecolor("#1F2937")
        patch.set_alpha(0.82)

    ax.set_title("Score Margin by Predicted Category", pad=12, fontweight="bold")
    ax.set_ylabel("Top score minus second-best score")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Margin distribution figure saved to %s", output_path)
    return output_path


def summarize_anchor_agreement(comparison_rows: list[dict[str, str]]) -> dict[str, float]:
    """统计外部锚点一致率，strong 与 contextual 分开报告。"""

    summary: dict[str, float] = {}
    for strength in ("strong", "contextual"):
        rows = [
            row
            for row in comparison_rows
            if row["anchor_strength"] == strength and row["is_comparable"] == "True"
        ]
        matches = [row for row in rows if row["is_match"] == "True"]
        summary[f"{strength}_n"] = float(len(rows))
        summary[f"{strength}_matches"] = float(len(matches))
        summary[f"{strength}_agreement"] = float(len(matches) / len(rows)) if rows else float("nan")

    comparable_rows = [row for row in comparison_rows if row["is_comparable"] == "True"]
    comparable_matches = [row for row in comparable_rows if row["is_match"] == "True"]
    summary["all_comparable_n"] = float(len(comparable_rows))
    summary["all_comparable_matches"] = float(len(comparable_matches))
    summary["all_comparable_agreement"] = (
        float(len(comparable_matches) / len(comparable_rows)) if comparable_rows else float("nan")
    )
    return summary


def summarize_margins(
    config: Config,
    clustered_rows: list[dict[str, str]],
) -> dict[str, float]:
    """统计每一类的中位数裕度，供报告描述使用。"""

    summary: dict[str, float] = {}
    for category in config.category_order:
        values = np.array(
            [
                score_margin(row, config)
                for row in clustered_rows
                if row.get("quality_status") == "valid" and row.get("category") == category
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        summary[f"{category}_median_margin"] = float(np.median(values)) if values.size else float("nan")
        summary[f"{category}_p10_margin"] = float(np.percentile(values, 10)) if values.size else float("nan")
    return summary


def main() -> None:
    """执行外部锚点验证和内部裕度诊断。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)

    clustered_rows = read_csv(config.clustered_csv)
    reference_rows = read_csv(config.reference_csv)
    comparison_rows = compare_references(config, clustered_rows, reference_rows)
    write_csv(config.comparison_csv, comparison_rows)
    logging.info("Anchor comparison saved to %s", config.comparison_csv)

    plot_anchor_confusion(config, comparison_rows)
    plot_margin_distribution(config, clustered_rows)

    anchor_summary = summarize_anchor_agreement(comparison_rows)
    margin_summary = summarize_margins(config, clustered_rows)
    logging.info("Anchor agreement summary: %s", anchor_summary)
    logging.info("Margin summary: %s", margin_summary)


if __name__ == "__main__":
    main()
