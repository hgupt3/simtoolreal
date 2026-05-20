"""Figure 1 plot: contact-tolerance evaluation for peg-in-hole.

The values below are placeholders. Replace ``DUMMY_SIM_RESULTS`` and
``DUMMY_REAL_WORLD_RESULTS`` with measured data when available.

    python plot_figures/fig1_plot_contact_tolerances.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
METHODS = ("Play-only", "Play2Win")
TOLERANCE_ROWS = (
    ("2 mm", "2mm"),
    ("0.5 mm", "0p5mm"),
)
EXPECTED_IMAGES = {
    "hole_2mm": "hole_2mm.png",
    "hole_0p5mm": "hole_0p5mm.png",
    "play_only_2mm_1": "play_only_2mm_1.png",
    "play_only_2mm_2": "play_only_2mm_2.png",
    "play2win_2mm_1": "play2win_2mm_1.png",
    "play2win_2mm_2": "play2win_2mm_2.png",
    "play_only_0p5mm_1": "play_only_0p5mm_1.png",
    "play_only_0p5mm_2": "play_only_0p5mm_2.png",
    "play2win_0p5mm_1": "play2win_0p5mm_1.png",
    "play2win_0p5mm_2": "play2win_0p5mm_2.png",
}

DUMMY_SIM_RESULTS = {
    "success_tolerance_mm": np.geomspace(0.2, 2.0, 10),
    "Play-only": {
        "success_rate": np.array([0.00, 0.00, 0.01, 0.02, 0.08, 0.25, 0.52, 0.73, 0.86, 0.92]),
    },
    "Play2Win": {
        "success_rate": np.array([0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.99, 0.995, 0.995]),
    },
}

DUMMY_REAL_WORLD_RESULTS = {
    "throughput_tolerance_mm": np.array([2.0, 0.5, 0.2]),
    "Play-only": {
        "throughput": np.array([11.3, 0.7, 0.0]),
    },
    "Play2Win": {
        "throughput": np.array([11.9, 11.3, 10.8]),
    },
}

STYLE = {
    "Play-only": {"color": "#D73027", "marker": "o"},
    "Play2Win": {"color": "#1A9850", "marker": "o"},
}


def _plot_success_rate(ax: plt.Axes, tolerances: np.ndarray) -> None:
    for method in METHODS:
        ax.plot(
            tolerances,
            100.0 * DUMMY_SIM_RESULTS[method]["success_rate"],
            label=method,
            linewidth=1.8,
            markersize=4.5,
            markeredgewidth=0.0,
            **STYLE[method],
        )

    ax.set_ylabel("Success rate (%)")
    ax.set_xscale("log")
    ax.set_xticks([2.0, 1.0, 0.5, 0.2])
    ax.set_xticklabels(["2", "1", "0.5", "0.2"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(float(tolerances.max()), float(tolerances.min()))
    ax.set_ylim(0.0, 104.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Success rate (sim)", loc="left", fontsize=8, fontweight="bold", pad=3)
    _style_axis(ax)


def _plot_throughput_bars(ax: plt.Axes, tolerances: np.ndarray) -> None:
    x = np.arange(len(tolerances))
    width = 0.30

    for offset, method in zip((-width / 2, width / 2), METHODS):
        ax.bar(
            x + offset,
            DUMMY_REAL_WORLD_RESULTS[method]["throughput"],
            width=width,
            label=method,
            color=STYLE[method]["color"],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in tolerances])
    ax.set_xlabel("Success tolerance (mm)")
    ax.set_ylabel("Throughput\n(success/min)")
    ax.set_ylim(0.0, 12.6)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_title("Throughput (real)", loc="left", fontsize=8, fontweight="bold", pad=3)
    _style_axis(ax)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)


def _draw_image_cell(
    ax: plt.Axes,
    path: Path,
    border_color: str | None = None,
    label: str | None = None,
) -> None:
    if path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.set_facecolor("#F3F3F3")
        ax.text(
            0.5,
            0.5,
            f"Missing\n{path.name}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color="0.35",
        )
    ax.set_xticks([])
    ax.set_yticks([])

    if border_color is None:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2.0)

    if label is not None:
        ax.text(
            0.03,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="0.15",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.16", "alpha": 0.9},
        )


def _draw_full_figure(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.38, 1.0],
        left=0.02,
        right=0.99,
        bottom=0.08,
        top=0.90,
        wspace=0.18,
    )
    left = outer[0].subgridspec(
        3,
        5,
        height_ratios=[0.20, 1.0, 1.0],
        width_ratios=[0.80, 1.20, 1.20, 1.20, 1.20],
        hspace=0.13,
        wspace=0.06,
    )
    right = outer[1].subgridspec(2, 1, hspace=0.56)

    fig.text(0.02, 0.965, "(a) Real-world tolerance tests", fontsize=9.5, fontweight="bold", ha="left")
    fig.text(0.72, 0.965, "(b) Success and throughput", fontsize=9.5, fontweight="bold", ha="left")

    headers = [
        (0, "Hole"),
        (1, "Play-only policy"),
        (3, "Play2Win"),
    ]
    for col, title in headers:
        ax = fig.add_subplot(left[0, col : col + (2 if col > 0 else 1)])
        ax.axis("off")
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=8, fontweight="bold")

    for row_idx, (label, suffix) in enumerate(TOLERANCE_ROWS, start=1):
        hole_ax = fig.add_subplot(left[row_idx, 0])
        _draw_image_cell(hole_ax, INPUT_DIR / EXPECTED_IMAGES[f"hole_{suffix}"])
        hole_ax.set_title(f"{label}\ntolerance", fontsize=7, pad=2)

        policy_specs = [
            (1, "play_only", "Play-only", "1"),
            (2, "play_only", "Play-only", "2"),
            (3, "play2win", "Play2Win", "1"),
            (4, "play2win", "Play2Win", "2"),
        ]
        for col, file_prefix, method, image_label in policy_specs:
            ax = fig.add_subplot(left[row_idx, col])
            _draw_image_cell(
                ax,
                INPUT_DIR / EXPECTED_IMAGES[f"{file_prefix}_{suffix}_{image_label}"],
                border_color=STYLE[method]["color"],
                label=image_label,
            )

    sim_ax = fig.add_subplot(right[0, 0])
    throughput_ax = fig.add_subplot(right[1, 0])
    return sim_ax, throughput_ax


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    sim_tolerances = DUMMY_SIM_RESULTS["success_tolerance_mm"]
    real_world_tolerances = DUMMY_REAL_WORLD_RESULTS["throughput_tolerance_mm"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.8, 3.15))
    sim_ax, throughput_ax = _draw_full_figure(fig)

    _plot_success_rate(sim_ax, sim_tolerances)
    _plot_throughput_bars(throughput_ax, real_world_tolerances)

    handles, labels = sim_ax.get_legend_handles_labels()
    sim_ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    png_path = OUT_DIR / "fig1_contact_tolerances.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
