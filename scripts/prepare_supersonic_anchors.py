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
class SupersonicAnchor:
    """记录一个超音速锚点的几何定义、来源与标签信息。"""

    name: str
    shape_family: str
    thickness_ratio: float
    evidence_basis: str
    source_title: str
    source_url: str
    notes: str
    source_kind: str = "analytic"


@dataclass(frozen=True)
class Config:
    """集中管理超音速锚点生成、引用追加与绘图参数。"""

    coord_dir: Path = Path("data/d_PV_20_coord")
    supersonic_dir: Path = Path("data/supersonic")
    reference_csv: Path = Path("data/known_airfoil_references.csv")
    manifest_csv: Path = Path("data/supersonic/supersonic_anchor_manifest.csv")
    figures_dir: Path = Path("figures")
    log_dir: Path = Path("log")
    figure_dpi: int = 420
    true_category: str = "Supersonic"
    anchor_strength: str = "strong"
    # NACA TN 2982 给出 biconvex 与 double-wedge 超音速翼型的解析形式；
    # NASA OpenVSP 文档也将 biconvex 与 wedge 定义为可参数化的超音速截面。
    analytic_anchors: tuple[SupersonicAnchor, ...] = field(
        default_factory=lambda: (
            SupersonicAnchor(
                name="double_wedge_tc04",
                shape_family="double_wedge",
                thickness_ratio=0.04,
                evidence_basis="NACA TN 2982 and NASA OpenVSP identify wedge/double-wedge sections as thin supersonic airfoils.",
                source_title="NACA TN 2982; NASA OpenVSP Cross-Sections",
                source_url="https://ntrs.nasa.gov/api/citations/19930083707/downloads/19930083707.pdf?attachment=true; https://www.nasa.gov/reference/openvsp-cross-sections/",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.04",
            ),
            SupersonicAnchor(
                name="double_wedge_tc06",
                shape_family="double_wedge",
                thickness_ratio=0.06,
                evidence_basis="NACA TN 2982 treats double-wedge airfoils as canonical supersonic analytical sections.",
                source_title="NACA TN 2982 Supersonic Flow Past Oscillating Airfoils",
                source_url="https://ntrs.nasa.gov/api/citations/19930083707/downloads/19930083707.pdf?attachment=true",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.06",
            ),
            SupersonicAnchor(
                name="double_wedge_tc08",
                shape_family="double_wedge",
                thickness_ratio=0.08,
                evidence_basis="NACA TN 2982 treats double-wedge airfoils as canonical supersonic analytical sections.",
                source_title="NACA TN 2982 Supersonic Flow Past Oscillating Airfoils",
                source_url="https://ntrs.nasa.gov/api/citations/19930083707/downloads/19930083707.pdf?attachment=true",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.08",
            ),
            SupersonicAnchor(
                name="biconvex_tc04",
                shape_family="biconvex",
                thickness_ratio=0.04,
                evidence_basis="NASA OpenVSP defines biconvex as a sharp-edged supersonic airfoil controlled by thickness-to-chord ratio.",
                source_title="NASA OpenVSP Cross-Sections",
                source_url="https://www.nasa.gov/reference/openvsp-cross-sections/",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.04",
            ),
            SupersonicAnchor(
                name="biconvex_tc06",
                shape_family="biconvex",
                thickness_ratio=0.06,
                evidence_basis="NACA TN 2982 gives a biconvex analytical airfoil form for supersonic airfoil theory.",
                source_title="NACA TN 2982 Supersonic Flow Past Oscillating Airfoils",
                source_url="https://ntrs.nasa.gov/api/citations/19930083707/downloads/19930083707.pdf?attachment=true",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.06",
            ),
            SupersonicAnchor(
                name="biconvex_tc08",
                shape_family="biconvex",
                thickness_ratio=0.08,
                evidence_basis="NASA OpenVSP defines biconvex as a sharp-edged supersonic airfoil controlled by thickness-to-chord ratio.",
                source_title="NASA OpenVSP Cross-Sections",
                source_url="https://www.nasa.gov/reference/openvsp-cross-sections/",
                notes="generated from authoritative analytic definition; maximum thickness-to-chord ratio 0.08",
            ),
        )
    )
    # NACA RM A53B02 Table II gives coordinate ordinates for airfoils tested at Mach 2.7-5.0.
    # Coordinates are normalized from a 2-inch chord to chord=1 and mirrored about the chord line.
    digitized_anchors: tuple[SupersonicAnchor, ...] = field(
        default_factory=lambda: (
            SupersonicAnchor(
                name="naca_rm_a53b02_304b",
                shape_family="naca_rm_a53b02_304b",
                thickness_ratio=0.0376,
                evidence_basis="NACA RM A53B02 reports airfoil coordinate ordinates for zero-lift drag tests at Mach 2.7 to 5.0.",
                source_title="NACA RM A53B02 Zero-Lift-Drag Characteristics of Symmetrical Blunt-Trailing-Edge Airfoils",
                source_url="https://ntrs.nasa.gov/api/citations/19930087797/downloads/19930087797.pdf",
                notes="digitized from NACA RM A53B02 Table II; blunt trailing edge retained with implied vertical closure",
                source_kind="digitized_table",
            ),
            SupersonicAnchor(
                name="naca_rm_a53b02_504b",
                shape_family="naca_rm_a53b02_504b",
                thickness_ratio=0.0374,
                evidence_basis="NACA RM A53B02 reports airfoil coordinate ordinates for zero-lift drag tests at Mach 2.7 to 5.0.",
                source_title="NACA RM A53B02 Zero-Lift-Drag Characteristics of Symmetrical Blunt-Trailing-Edge Airfoils",
                source_url="https://ntrs.nasa.gov/api/citations/19930087797/downloads/19930087797.pdf",
                notes="digitized from NACA RM A53B02 Table II; blunt trailing edge retained with implied vertical closure",
                source_kind="digitized_table",
            ),
        )
    )
    palette: dict[str, str] = field(
        default_factory=lambda: {
            "double_wedge": "#D95F02",
            "biconvex": "#1B9E77",
            "naca_rm_a53b02_304b": "#7570B3",
            "naca_rm_a53b02_504b": "#E7298A",
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
    """读取 UTF-8 BOM 兼容 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    """写出字典表 CSV，并确保父目录存在。"""

    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_canonical_x_grid(config: Config) -> np.ndarray:
    """从现有翼型坐标中读取项目统一的 301 点 x 网格。"""

    first_file = next(iter(sorted(config.coord_dir.glob("*.csv"))), None)
    if first_file is None:
        raise FileNotFoundError(f"No coordinate CSV files found in {config.coord_dir}")
    rows = read_csv(first_file)
    x_grid = np.array([float(row["x"]) for row in rows], dtype=float)
    if x_grid.size < 3:
        raise ValueError(f"Canonical x-grid is too short: {first_file}")
    logging.info("Loaded canonical x-grid with %d points from %s", x_grid.size, first_file)
    return x_grid


def double_wedge_half_thickness(x: np.ndarray, thickness_ratio: float) -> np.ndarray:
    """计算对称 double-wedge 翼型上表面的半厚度分布。"""

    half = 0.5 * thickness_ratio * (1.0 - np.abs(2.0 * x - 1.0))
    return np.maximum(half, 0.0)


def biconvex_half_thickness(x: np.ndarray, thickness_ratio: float) -> np.ndarray:
    """计算对称 biconvex 翼型上表面的半厚度分布。"""

    # y = tau / 2 * (1 - (2x - 1)^2)，其中 tau 是最大总厚度/弦长。
    half = 0.5 * thickness_ratio * (1.0 - (2.0 * x - 1.0) ** 2)
    return np.maximum(half, 0.0)


def naca_rm_a53b02_half_thickness(anchor_name: str, x: np.ndarray) -> np.ndarray:
    """按 NACA RM A53B02 Table II 的坐标表插值得到半厚度分布。"""

    # 表中 x 与 y 单位为英寸，弦长为 2 英寸；这里统一归一化为 chord=1。
    tables: dict[str, tuple[list[float], list[float]]] = {
        "naca_rm_a53b02_304b": (
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.087, 1.477, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            [0.0, 0.0047, 0.0094, 0.0142, 0.0189, 0.0234, 0.0273, 0.0307, 0.0336, 0.0357, 0.0372, 0.0376, 0.0376, 0.0376, 0.0368, 0.0350, 0.0322, 0.0283, 0.0233],
        ),
        "naca_rm_a53b02_504b": (
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.295, 1.689, 1.7, 1.8, 1.9, 2.0],
            [0.0, 0.0039, 0.0078, 0.0116, 0.0155, 0.0194, 0.0230, 0.0264, 0.0294, 0.0321, 0.0343, 0.0361, 0.0371, 0.0374, 0.0374, 0.0374, 0.0368, 0.0356, 0.0336],
        ),
    }
    if anchor_name not in tables:
        raise KeyError(f"Unknown NACA RM A53B02 anchor: {anchor_name}")
    x_inch, y_inch = tables[anchor_name]
    x_norm = np.asarray(x_inch, dtype=float) / 2.0
    y_norm = np.asarray(y_inch, dtype=float) / 2.0
    return np.interp(x, x_norm, y_norm)


def build_closed_loop_coordinates(anchor: SupersonicAnchor, x_grid: np.ndarray) -> np.ndarray:
    """按项目坐标顺序生成上表面到下表面的翼型轮廓。"""

    if anchor.shape_family == "double_wedge":
        half = double_wedge_half_thickness(x_grid, anchor.thickness_ratio)
    elif anchor.shape_family == "biconvex":
        half = biconvex_half_thickness(x_grid, anchor.thickness_ratio)
    elif anchor.shape_family.startswith("naca_rm_a53b02"):
        half = naca_rm_a53b02_half_thickness(anchor.name, x_grid)
    else:
        raise ValueError(f"Unsupported shape family: {anchor.shape_family}")

    leading_edge_index = int(np.argmin(x_grid))
    y = half.copy()
    # 项目坐标前半段为上表面，后半段为下表面。
    y[leading_edge_index + 1 :] *= -1.0
    return np.column_stack([x_grid, y])


def validate_coordinates(name: str, coords: np.ndarray) -> dict[str, str]:
    """检查坐标数值、x 范围和尾缘间隙，返回可写入 manifest 的诊断信息。"""

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{name}: coordinates must have shape (n, 2).")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{name}: coordinates contain non-finite values.")
    x = coords[:, 0]
    y = coords[:, 1]
    if np.min(x) < -1e-10 or np.max(x) > 1.0 + 1e-10:
        raise ValueError(f"{name}: x values are outside [0, 1].")
    te_gap = float(abs(y[0] - y[-1]))
    max_thickness = float(np.max(y) - np.min(y))
    return {
        "point_count": str(coords.shape[0]),
        "x_min": f"{float(np.min(x)):.12g}",
        "x_max": f"{float(np.max(x)):.12g}",
        "max_thickness_ratio": f"{max_thickness:.8f}",
        "trailing_edge_gap": f"{te_gap:.8f}",
    }


def write_coordinate_csv(path: Path, coords: np.ndarray) -> None:
    """按项目通用的 x,y 表头写出翼型坐标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        writer.writerows((f"{x:.16g}", f"{y:.16g}") for x, y in coords)


def update_reference_csv(config: Config, anchors: list[SupersonicAnchor]) -> None:
    """幂等追加超音速强锚点，不改变已有表头和已有人工记录。"""

    rows = read_csv(config.reference_csv)
    if not rows:
        raise ValueError(f"Reference CSV is empty: {config.reference_csv}")
    fieldnames = list(rows[0].keys())
    required = ["airfoil_name", "true_category", "anchor_strength", "evidence_basis", "source_title", "source_url", "notes"]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise KeyError(f"Missing required columns in {config.reference_csv}: {missing}")

    by_name = {row["airfoil_name"].strip().lower(): row for row in rows}
    appended = 0
    for anchor in anchors:
        row = {
            "airfoil_name": anchor.name,
            "true_category": config.true_category,
            "anchor_strength": config.anchor_strength,
            "evidence_basis": anchor.evidence_basis,
            "source_title": anchor.source_title,
            "source_url": anchor.source_url,
            "notes": anchor.notes,
        }
        key = anchor.name.lower()
        if key in by_name:
            by_name[key].update(row)
        else:
            rows.append(row)
            by_name[key] = row
            appended += 1

    write_csv(config.reference_csv, rows, fieldnames=fieldnames)
    logging.info("Reference CSV updated at %s; appended %d new rows.", config.reference_csv, appended)


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


def plot_anchor_geometries(config: Config, coordinate_map: dict[str, np.ndarray], anchors: list[SupersonicAnchor]) -> None:
    """绘制超音速锚点几何对比图，输出 PNG 与 PDF。"""

    configure_plot_style()
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), dpi=config.figure_dpi, sharex=True)
    grouped = {
        "Analytic thin supersonic sections": [a for a in anchors if a.source_kind == "analytic"],
        "Digitized NACA RM A53B02 sections": [a for a in anchors if a.source_kind == "digitized_table"],
    }

    for ax, (title, group) in zip(axes, grouped.items(), strict=True):
        for anchor in group:
            coords = coordinate_map[anchor.name]
            color = config.palette.get(anchor.shape_family, "#4B5563")
            ax.plot(coords[:, 0], coords[:, 1], lw=1.5, color=color, label=anchor.name)
        ax.axhline(0.0, color="#9CA3AF", lw=0.8)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("y / c")
        ax.grid(True, color="#E5E7EB", lw=0.8)
        ax.legend(ncol=2, frameon=True, edgecolor="#D1D5DB", facecolor="white")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("x / c")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        output = config.figures_dir / f"supersonic_anchor_geometries.{suffix}"
        fig.savefig(output, dpi=config.figure_dpi if suffix == "png" else None, bbox_inches="tight")
        logging.info("Saved geometry figure: %s", output)
    plt.close(fig)


def main() -> None:
    """主流程：生成超音速坐标、更新锚点 CSV、写出 manifest 并绘图。"""

    config = Config()
    log_path = setup_logging(config)
    logging.info("Log file: %s", log_path)
    x_grid = load_canonical_x_grid(config)
    anchors = list(config.analytic_anchors) + list(config.digitized_anchors)
    coordinate_map: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, str]] = []

    for anchor in tqdm(anchors, desc="Preparing supersonic anchors", unit="anchor"):
        coords = build_closed_loop_coordinates(anchor, x_grid)
        diagnostics = validate_coordinates(anchor.name, coords)
        output_path = config.supersonic_dir / f"{anchor.name}.csv"
        write_coordinate_csv(output_path, coords)
        coordinate_map[anchor.name] = coords
        manifest_rows.append(
            {
                "airfoil_name": anchor.name,
                "true_category": config.true_category,
                "anchor_strength": config.anchor_strength,
                "shape_family": anchor.shape_family,
                "source_kind": anchor.source_kind,
                "coordinate_file": str(output_path.as_posix()),
                "source_title": anchor.source_title,
                "source_url": anchor.source_url,
                "notes": anchor.notes,
                **diagnostics,
            }
        )

    update_reference_csv(config, anchors)
    write_csv(config.manifest_csv, manifest_rows)
    plot_anchor_geometries(config, coordinate_map, anchors)
    logging.info("Prepared %d supersonic anchors.", len(anchors))


if __name__ == "__main__":
    main()
