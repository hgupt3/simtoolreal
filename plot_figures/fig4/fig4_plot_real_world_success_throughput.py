"""Figure 4 draft: real-world success rate and throughput.

Dummy values encode the intended comparison over three real-world tasks:

- Sim versus real success rate for Play2Win.
- Throughput of Play2Win versus teleoperation with a parallel-jaw gripper and
  direct human assembly.
- Qualitative strategy images showing why the parallel-jaw teleop strategy is
  cumbersome.

    python plot_figures/fig4_plot_real_world_success_throughput.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs"
OUT_DIR = ROOT_DIR / "outputs"

TASKS = ("PegInHole", "FabricaAssembly", "FurnitureBench")
APPROACHES = ("Play2Win", "Teleop + PJaw", "Human")

INIT_IMAGES = (
    ("init_peginhole.png", "PegInHole"),
    ("init_fabrica.png", "Fabrica"),
    ("init_furniturebench.png", "Furniture"),
)

STRATEGY_IMAGES = (
    ("strategy_play2win_1.png", "Play2Win", "1"),
    ("strategy_play2win_2.png", "Play2Win", "2"),
    ("strategy_teleop_pjaw_1.png", "Teleop + PJaw", "1"),
    ("strategy_teleop_pjaw_2.png", "Teleop + PJaw", "2"),
    ("strategy_human_1.png", "Human", "1"),
    ("strategy_human_2.png", "Human", "2"),
)

SUCCESS_RATE = {
    "sim": np.array([0.98, 0.94, 0.90]),
    "real": np.array([0.94, 0.86, 0.82]),
}

THROUGHPUT = {
    "Play2Win": np.array([11.8, 5.6, 2.7]),
    "Teleop + PJaw": np.array([2.1, 1.0, 0.45]),
    "Human": np.array([13.6, 6.4, 3.1]),
}

COLORS = {
    "sim": "#74ADD1",
    "real": "#1A9850",
    "Play2Win": "#1A9850",
    "Teleop + PJaw": "#D73027",
    "Human": "#4D4D4D",
}


def _configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", labelsize=7.0, width=0.75, length=2.6)


def _plot_success_rates(ax: plt.Axes) -> None:
    x = np.arange(len(TASKS))
    width = 0.34
    ax.bar(x - width / 2, 100.0 * SUCCESS_RATE["sim"], width=width, color=COLORS["sim"], label="Sim")
    ax.bar(x + width / 2, 100.0 * SUCCESS_RATE["real"], width=width, color=COLORS["real"], label="Real")
    ax.set_title("Success rate", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(["Peg\nHole", "Fabrica", "Furniture"], fontsize=6.5)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.00, 0.02), handlelength=1.1)
    _style_axis(ax)


def _plot_task_throughput(ax: plt.Axes, task_idx: int) -> None:
    x = np.arange(len(APPROACHES))
    vals = [THROUGHPUT[approach][task_idx] for approach in APPROACHES]
    ax.bar(x, vals, color=[COLORS[approach] for approach in APPROACHES], width=0.58)
    ax.set_title(TASKS[task_idx], fontsize=7.4, fontweight="bold", pad=3)
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(x)
    ax.set_xticklabels(["P2W", "PJaw", "Human"], fontsize=6.5, rotation=0)
    if task_idx == 0:
        ax.set_ylabel("Successes/min")
    else:
        ax.set_yticklabels([])
    _style_axis(ax)


def _draw_image(ax: plt.Axes, path: Path, label: str | None = None) -> None:
    if path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.set_facecolor("#F2F2F2")
        ax.text(
            0.5,
            0.52,
            f"Missing\n{path.name}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            color="0.35",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.72")
        spine.set_linewidth(0.75)
    if label is not None:
        ax.text(
            0.04,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.14", "alpha": 0.9},
        )


def _draw_init_variation_panel(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(3, 3, height_ratios=[0.18, 0.14, 1.0], hspace=0.03, wspace=0.07)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.5, "Initial-state variation (10 runs)", ha="left", va="center", fontsize=8, fontweight="bold")
    for idx, (filename, title) in enumerate(INIT_IMAGES):
        label_ax = fig.add_subplot(grid[1, idx])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=6.3, fontweight="bold")
        ax = fig.add_subplot(grid[2, idx])
        _draw_image(ax, INPUT_DIR / filename)


def _draw_strategy_panel(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(3, 6, height_ratios=[0.18, 0.14, 1.0], hspace=0.03, wspace=0.05)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.5, "Strategies on one task", ha="left", va="center", fontsize=8, fontweight="bold")

    for start_col, title in zip((0, 2, 4), APPROACHES):
        label_ax = fig.add_subplot(grid[1, start_col : start_col + 2])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=6.5, fontweight="bold")

    for idx, (filename, approach, frame_id) in enumerate(STRATEGY_IMAGES):
        ax = fig.add_subplot(grid[2, idx])
        _draw_image(ax, INPUT_DIR / filename, label=frame_id)
        for spine in ax.spines.values():
            spine.set_color(COLORS[approach])
            spine.set_linewidth(1.25)


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(8.0, 3.95))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 2.0],
        left=0.055,
        right=0.99,
        bottom=0.10,
        top=0.90,
        wspace=0.18,
    )
    left = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.45)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.48)

    fig.text(0.055, 0.955, "(a) Sim-to-real success", fontsize=9.5, fontweight="bold", ha="left")
    fig.text(0.405, 0.955, "(b) Real-world throughput", fontsize=9.5, fontweight="bold", ha="left")

    success_ax = fig.add_subplot(left[0, 0])
    _plot_success_rates(success_ax)
    _draw_init_variation_panel(fig, left[1, 0])

    throughput_grid = right[0, 0].subgridspec(1, 3, wspace=0.34)
    throughput_axes = []
    for task_idx in range(len(TASKS)):
        ax = fig.add_subplot(throughput_grid[0, task_idx])
        _plot_task_throughput(ax, task_idx)
        throughput_axes.append(ax)
    _draw_strategy_panel(fig, right[1, 0])
    return fig


def main() -> None:
    _configure_rcparams()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = _draw_full_figure()
    png_path = OUT_DIR / "fig4_real_world_success_throughput.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
