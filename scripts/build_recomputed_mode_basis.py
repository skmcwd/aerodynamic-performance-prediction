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
except ModuleNotFoundError:  # pragma: no cover - 仅在缺少 tqdm 时启用兜底
    class tqdm:  # type: ignore[no-redef]
        """简易进度条兜底类，保证脚本在无 tqdm 环境中仍可运行。"""

        def __init__(self, iterable: Iterable | None = None, **_: object) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])


@dataclass(frozen=True)
class Config:
    """集中管理可复现 19-mode 几何基构建与输出路径。"""

    coord_dir: Path = Path("data/d_PV_20_coord")
    supersonic_dir: Path = Path("data/supersonic")
    clustered_csv: Path = Path("data/CST_19ML_clustered.csv")
    reference_csv: Path = Path("data/known_airfoil_references.csv")
    output_csv: Path = Path("data/recomputed_modes_all_airfoils.csv")
    basis_dir: Path = Path("data/mode_basis")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    n_modes: int = 19
    figure_dpi: int = 420
    mode_prefix: str = "recomputed_mode_"


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
    """读取 UTF-8 BOM 兼容 CSV。"""

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


def load_coordinates(path: Path) -> np.ndarray:
    """读取一个 x,y 坐标文件并返回二维数组。"""

    rows = read_csv(path)
    if not rows or "x" not in rows[0] or "y" not in rows[0]:
        raise ValueError(f"{path} must contain x,y columns.")
    coords = np.array([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError(f"Invalid coordinate file: {path}")
    return coords


def split_surface(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按最小 x 点把坐标拆成上表面与下表面。"""

    le_index = int(np.argmin(coords[:, 0]))
    upper = coords[: le_index + 1]
    lower = coords[le_index + 1 :]
    if upper.size == 0 or lower.size == 0:
        raise ValueError("Coordinate loop cannot be split into upper/lower surfaces.")
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]


def resample_to_grid(coords: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """将任意同拓扑翼型重采样到项目统一 x 网格。"""

    if coords.shape[0] == target_x.size and np.allclose(coords[:, 0], target_x, atol=1e-10):
        return coords[:, 1].copy()

    target_le_index = int(np.argmin(target_x))
    upper_x, upper_y, lower_x, lower_y = split_surface(coords)
    # np.interp 需要升序 x，因此上表面需要反转。
    upper_y_grid = np.interp(target_x[: target_le_index + 1], upper_x[::-1], upper_y[::-1])
    lower_y_grid = np.interp(target_x[target_le_index + 1 :], lower_x, lower_y)
    return np.concatenate([upper_y_grid, lower_y_grid])


def load_reference_map(path: Path) -> dict[str, dict[str, str]]:
    """读取真实锚点来源表，按翼型名小写索引。"""

    if not path.exists():
        return {}
    return {row["airfoil_name"].strip().lower(): row for row in read_csv(path)}


def load_clustered_map(path: Path) -> dict[str, dict[str, str]]:
    """读取原聚类结果，按翼型名小写索引。"""

    return {row["name"].strip().lower(): row for row in read_csv(path)}


def fit_pca_basis(matrix: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 SVD 训练可复现几何 PCA 基。"""

    if matrix.shape[0] <= n_modes:
        raise ValueError("Not enough samples to fit the requested number of modes.")
    mean_y = matrix.mean(axis=0)
    centered = matrix - mean_y
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_modes]
    scores = centered @ components.T
    explained_variance = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    explained_ratio = explained_variance / np.sum(explained_variance)
    return mean_y, components, scores, explained_ratio[:n_modes]


def project(matrix: np.ndarray, mean_y: np.ndarray, components: np.ndarray) -> np.ndarray:
    """将翼型 y 向量投影到已训练 PCA 基。"""

    return (matrix - mean_y) @ components.T


def parse_original_modes(clustered_map: dict[str, dict[str, str]], names: list[str]) -> np.ndarray:
    """提取原始 Mode 1..19，用于诊断新旧表征的一致性。"""

    columns = [f"Mode {index}" for index in range(1, 20)]
    matrix = np.full((len(names), len(columns)), np.nan, dtype=float)
    for row_index, name in enumerate(names):
        row = clustered_map.get(name.lower())
        if not row:
            continue
        for col_index, column in enumerate(columns):
            try:
                matrix[row_index, col_index] = float(row[column])
            except (KeyError, TypeError, ValueError):
                matrix[row_index, col_index] = np.nan
    return matrix


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Pearson 相关系数，自动处理常数列。"""

    if a.size < 3 or b.size < 3:
        return float("nan")
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def build_diagnostics(
    config: Config,
    training_y: np.ndarray,
    training_scores: np.ndarray,
    mean_y: np.ndarray,
    components: np.ndarray,
    explained_ratio: np.ndarray,
    names: list[str],
    clustered_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """生成解释方差、重构误差、新旧 Mode 相关性诊断。"""

    rows: list[dict[str, str]] = []
    cumulative = 0.0
    for index, value in enumerate(explained_ratio, start=1):
        cumulative += float(value)
        rows.append(
            {
                "diagnostic_type": "explained_variance",
                "recomputed_mode": f"{config.mode_prefix}{index}",
                "original_mode": "",
                "metric": "explained_variance_ratio",
                "value": f"{float(value):.10f}",
                "abs_value": f"{abs(float(value)):.10f}",
                "notes": f"cumulative={cumulative:.10f}",
            }
        )

    reconstructed = mean_y + training_scores @ components
    rmse = np.sqrt(np.mean((training_y - reconstructed) ** 2, axis=1))
    rows.extend(
        [
            {
                "diagnostic_type": "reconstruction",
                "recomputed_mode": f"first_{config.n_modes}_modes",
                "original_mode": "",
                "metric": "mean_rmse_y",
                "value": f"{float(np.mean(rmse)):.10f}",
                "abs_value": f"{float(abs(np.mean(rmse))):.10f}",
                "notes": "PCA reconstruction error on original coordinate set",
            },
            {
                "diagnostic_type": "reconstruction",
                "recomputed_mode": f"first_{config.n_modes}_modes",
                "original_mode": "",
                "metric": "p95_rmse_y",
                "value": f"{float(np.percentile(rmse, 95)):.10f}",
                "abs_value": f"{float(abs(np.percentile(rmse, 95))):.10f}",
                "notes": "PCA reconstruction error on original coordinate set",
            },
        ]
    )

    original_modes = parse_original_modes(clustered_map, names)
    finite_row_mask = np.all(np.isfinite(original_modes), axis=1)
    if np.any(finite_row_mask):
        original_modes = original_modes[finite_row_mask]
        comparable_scores = training_scores[finite_row_mask]
        for rec_index in range(config.n_modes):
            correlations: list[tuple[int, float]] = []
            for orig_index in range(config.n_modes):
                corr = pearson_corr(comparable_scores[:, rec_index], original_modes[:, orig_index])
                correlations.append((orig_index + 1, corr))
            best_original, best_corr = max(correlations, key=lambda item: abs(item[1]) if np.isfinite(item[1]) else -1.0)
            rows.append(
                {
                    "diagnostic_type": "original_mode_alignment",
                    "recomputed_mode": f"{config.mode_prefix}{rec_index + 1}",
                    "original_mode": f"Mode {best_original}",
                    "metric": "best_abs_pearson_correlation",
                    "value": f"{best_corr:.10f}",
                    "abs_value": f"{abs(best_corr):.10f}",
                    "notes": "High values imply similar span/order; low values mean the historical basis is not recoverable exactly.",
                }
            )
    else:
        rows.append(
            {
                "diagnostic_type": "original_mode_alignment",
                "recomputed_mode": "",
                "original_mode": "",
                "metric": "best_abs_pearson_correlation",
                "value": "nan",
                "abs_value": "nan",
                "notes": "No finite original Mode rows were available for comparison.",
            }
        )
    return rows


def collect_coordinate_files(config: Config) -> tuple[list[Path], list[Path]]:
    """收集原始坐标文件与新增超音速坐标文件。"""

    original_files = sorted(config.coord_dir.glob("*.csv"))
    supersonic_files = []
    if config.supersonic_dir.exists():
        for path in sorted(config.supersonic_dir.glob("*.csv")):
            try:
                rows = read_csv(path)
            except (UnicodeDecodeError, csv.Error):
                continue
            if rows and {"x", "y"}.issubset(rows[0].keys()):
                supersonic_files.append(path)
    if not original_files:
        raise FileNotFoundError(f"No original coordinate files found in {config.coord_dir}")
    return original_files, supersonic_files


def build_output_rows(
    config: Config,
    files: list[Path],
    scores: np.ndarray,
    group_name: str,
    reference_map: dict[str, dict[str, str]],
    clustered_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """构造 recomputed_mode 输出表行。"""

    rows: list[dict[str, str]] = []
    for path, values in zip(files, scores, strict=True):
        name = path.stem
        reference = reference_map.get(name.lower(), {})
        clustered = clustered_map.get(name.lower(), {})
        row = {
            "name": name,
            "geometry_group": group_name,
            "coordinate_file": path.as_posix(),
            "true_category": reference.get("true_category", ""),
            "anchor_strength": reference.get("anchor_strength", ""),
            "source_title": reference.get("source_title", ""),
            "source_url": reference.get("source_url", ""),
            "notes": reference.get("notes", ""),
            "clustered_category": clustered.get("category", ""),
            "quality_status": clustered.get("quality_status", "external_anchor" if group_name == "supersonic_anchor" else ""),
            "original_mode_available": str(bool(clustered)),
        }
        for index, value in enumerate(values, start=1):
            row[f"{config.mode_prefix}{index}"] = f"{float(value):.12g}"
        rows.append(row)
    return rows


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
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_explained_variance(config: Config, explained_ratio: np.ndarray) -> None:
    """绘制 recomputed mode 的解释方差图。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    modes = np.arange(1, explained_ratio.size + 1)
    cumulative = np.cumsum(explained_ratio)
    fig, ax1 = plt.subplots(figsize=(8.6, 5.4), dpi=config.figure_dpi)
    ax1.bar(modes, explained_ratio * 100.0, color="#2A9D8F", edgecolor="#0F766E", linewidth=0.5)
    ax1.set_xlabel("Recomputed Mode Index")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(modes, cumulative * 100.0, color="#E76F51", lw=2.0, marker="o", ms=4)
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax1.set_title("Recomputed 19-Mode Geometry Basis", fontweight="bold", pad=10)
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        output = config.figures_dir / f"recomputed_mode_explained_variance.{suffix}"
        fig.savefig(output, dpi=config.figure_dpi if suffix == "png" else None, bbox_inches="tight")
        logging.info("Saved explained variance figure: %s", output)
    plt.close(fig)


def main() -> None:
    """主流程：训练 PCA/SVD 几何基，投影原始与超音速锚点，并输出诊断。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)
    original_files, supersonic_files = collect_coordinate_files(config)
    reference_map = load_reference_map(config.reference_csv)
    clustered_map = load_clustered_map(config.clustered_csv)

    canonical_x = load_coordinates(original_files[0])[:, 0]
    original_y = []
    original_names = []
    for path in tqdm(original_files, desc="Loading original coordinates", unit="airfoil"):
        coords = load_coordinates(path)
        original_y.append(resample_to_grid(coords, canonical_x))
        original_names.append(path.stem)
    original_matrix = np.vstack(original_y)

    mean_y, components, original_scores, explained_ratio = fit_pca_basis(original_matrix, config.n_modes)
    config.basis_dir.mkdir(parents=True, exist_ok=True)
    basis_path = config.basis_dir / "recomputed_19mode_basis.npz"
    np.savez(
        basis_path,
        x_grid=canonical_x,
        mean_y=mean_y,
        components=components,
        explained_variance_ratio=explained_ratio,
        original_airfoil_names=np.array(original_names, dtype=object),
    )
    logging.info("Saved recomputed basis to %s", basis_path)

    supersonic_scores = np.empty((0, config.n_modes), dtype=float)
    if supersonic_files:
        supersonic_y = []
        for path in tqdm(supersonic_files, desc="Projecting supersonic anchors", unit="airfoil"):
            coords = load_coordinates(path)
            supersonic_y.append(resample_to_grid(coords, canonical_x))
        supersonic_scores = project(np.vstack(supersonic_y), mean_y, components)

    rows = build_output_rows(config, original_files, original_scores, "original_dataset", reference_map, clustered_map)
    rows.extend(build_output_rows(config, supersonic_files, supersonic_scores, "supersonic_anchor", reference_map, clustered_map))
    mode_columns = [f"{config.mode_prefix}{index}" for index in range(1, config.n_modes + 1)]
    output_fields = [
        "name",
        "geometry_group",
        "coordinate_file",
        "true_category",
        "anchor_strength",
        "source_title",
        "source_url",
        "notes",
        "clustered_category",
        "quality_status",
        "original_mode_available",
        *mode_columns,
    ]
    write_csv(config.output_csv, rows, output_fields)
    logging.info("Saved recomputed mode table to %s with %d rows.", config.output_csv, len(rows))

    diagnostics = build_diagnostics(
        config,
        original_matrix,
        original_scores,
        mean_y,
        components,
        explained_ratio,
        original_names,
        clustered_map,
    )
    diagnostics_path = config.basis_dir / "recomputed_mode_diagnostics.csv"
    write_csv(
        diagnostics_path,
        diagnostics,
        ["diagnostic_type", "recomputed_mode", "original_mode", "metric", "value", "abs_value", "notes"],
    )
    plot_explained_variance(config, explained_ratio)
    logging.info("Saved diagnostics to %s", diagnostics_path)


if __name__ == "__main__":
    main()
