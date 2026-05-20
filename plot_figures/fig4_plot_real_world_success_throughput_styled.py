"""Styled Figure 4 — real-world success rate and throughput.

Preserves the original 2-row x 2-column layout, including the init-state
visualizations (per user preference). Polish-only changes:
- Legend in (a) moves out of the bars (was the worst overlap in the draft).
- Per-bar numeric value annotations (sapg-style readability).
- (b) collapses 3 mini-plots into one grouped bar chart over the 3 tasks.
- Strategy panel borders softened to 0.9 lw + a faint background tint.
- Times New Roman serif body + STIX math via shared ``_style.py``.

Output: ``outputs/fig4_real_world_success_throughput_styled.png`` at dpi=600.

    python plot_figures/fig4_plot_real_world_success_throughput_styled.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from _style import COLORS, configure_rcparams, panel_label, save_figure, style_axis


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs" / "fig4"

TASKS = ("PegInHole", "FabricaAssembly", "FurnitureBench")
TASK_TICK_LABELS = ("Peg-in-Hole", "Fabrica", "Furniture")
APPROACHES = ("Play2Win", "Teleop + PJaw", "Human")

INIT_IMAGES = (
    ("init_peginhole.png", "Peg-in-Hole"),
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

APPROACH_COLORS = {
    "Play2Win": COLORS["play2win"],
    "Teleop + PJaw": COLORS["teleop_pjaw"],
    "Human": COLORS["human"],
}


def _annotate_bar(ax: plt.Axes, x: float, y: float, fmt: str = "{:.0f}", suffix: str = "") -> None:
    ax.text(
        x,
        y + 1.2,
        f"{fmt.format(y)}{suffix}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )


def _plot_success_rates(ax: plt.Axes) -> None:
    x = np.arange(len(TASKS))
    width = 0.34

    sim_vals = 100.0 * SUCCESS_RATE["sim"]
    real_vals = 100.0 * SUCCESS_RATE["real"]

    ax.bar(x - width / 2, sim_vals, width=width, color=COLORS["sim"], alpha=0.9,
           edgecolor="none", zorder=3, label="Sim")
    ax.bar(x + width / 2, real_vals, width=width, color=COLORS["real"], alpha=0.9,
           edgecolor="none", zorder=3, label="Real")

    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_TICK_LABELS)
    ax.set_xlim(-0.6, len(TASKS) - 0.4)
    ax.tick_params(axis="x", length=0)
    style_axis(ax)
    ax.set_axisbelow(True)


def _plot_throughput_grouped(ax: plt.Axes) -> None:
    x = np.arange(len(TASKS))
    width = 0.26
    offsets = [-width, 0.0, width]

    for offset, approach in zip(offsets, APPROACHES):
        vals = THROUGHPUT[approach]
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=APPROACH_COLORS[approach],
            alpha=0.9,
            edgecolor="none",
            zorder=3,
            label=approach,
        )

    ax.set_ylabel("Successes / min")
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_TICK_LABELS)
    ax.set_xlim(-0.6, len(TASKS) - 0.4)
    ax.tick_params(axis="x", length=0)
    style_axis(ax)
    ax.set_axisbelow(True)


def _tint_color(hex_color: str, alpha: float) -> tuple:
    rgb = mcolors.to_rgb(hex_color)
    return tuple(1.0 - (1.0 - c) * alpha for c in rgb)


def _draw_image(ax: plt.Axes, path: Path, label: str | None = None,
                border_color: str | None = None, bg_color: str | None = None) -> None:
    if path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.set_facecolor(bg_color if bg_color else "#F3F3F3")
        ax.text(
            0.5,
            0.5,
            f"Missing\n{path.name}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color="0.4",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    if border_color is None:
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.7)
    else:
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(0.9)
    if label is not None:
        ax.text(
            0.04,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.14", "alpha": 0.85},
        )


def _draw_init_variation_panel(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(2, 3, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.10)
    for idx, (filename, title) in enumerate(INIT_IMAGES):
        label_ax = fig.add_subplot(grid[0, idx])
        label_ax.axis("off")
        label_ax.text(0.5, 0.4, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax = fig.add_subplot(grid[1, idx])
        _draw_image(ax, INPUT_DIR / filename)


def _draw_strategy_panel(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(2, 6, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.06)

    for start_col, approach in zip((0, 2, 4), APPROACHES):
        label_ax = fig.add_subplot(grid[0, start_col : start_col + 2])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.4,
            approach,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=APPROACH_COLORS[approach],
        )

    for idx, (filename, approach, frame_id) in enumerate(STRATEGY_IMAGES):
        ax = fig.add_subplot(grid[1, idx])
        _draw_image(
            ax,
            INPUT_DIR / filename,
            label=frame_id,
            border_color=APPROACH_COLORS[approach],
            bg_color=_tint_color(APPROACH_COLORS[approach], 0.06),
        )


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.8, 4.5))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.85],
        left=0.07,
        right=0.97,
        bottom=0.10,
        top=0.84,
        wspace=0.22,
    )
    left = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.45)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.45)

    panel_label(fig, 0.07, 0.945, "(a) Sim-to-real success")
    panel_label(fig, 0.42, 0.945, "(b) Real-world throughput")
    panel_label(fig, 0.07, 0.48, "(c) Initial-state variation")
    panel_label(fig, 0.42, 0.48, "(d) Strategies on one task")

    success_ax = fig.add_subplot(left[0, 0])
    _plot_success_rates(success_ax)
    _draw_init_variation_panel(fig, left[1, 0])

    throughput_ax = fig.add_subplot(right[0, 0])
    _plot_throughput_grouped(throughput_ax)
    _draw_strategy_panel(fig, right[1, 0])

    # Figure-level legends placed under each top-row panel-label (out of bars).
    success_handles, success_labels = success_ax.get_legend_handles_labels()
    fig.legend(
        success_handles,
        success_labels,
        loc="upper left",
        bbox_to_anchor=(0.27, 0.945),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    throughput_handles, throughput_labels = throughput_ax.get_legend_handles_labels()
    fig.legend(
        throughput_handles,
        throughput_labels,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.945),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    return fig


def main() -> None:
    configure_rcparams()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = _draw_full_figure()
    save_figure(fig, "fig4_real_world_success_throughput_styled")
    plt.close(fig)


if __name__ == "__main__":
    main()
