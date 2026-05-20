"""Styled Figure 2 — Play2Win vs training from scratch.

Two-row layout (per user preference):
- Row (a): 5 task curves with CAD thumbnail strip above; shared top-right legend.
- Row (b/c/d): fixtured task curve | force robustness curve | qualitative strategy.

Inputs/outputs:
- Reads optional CAD/strategy images from ``inputs/fig2/``.
- Writes ``outputs/fig2_scratch_vs_play2win_styled.png`` at dpi=600.

    python plot_figures/fig2_plot_scratch_vs_play2win_styled.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# _style.py lives one directory up (plot_figures/), shared across all figs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import (
    COLORS,
    STYLES,
    TASK_COLORS,
    TASK_DISPLAY_NAMES,
    configure_rcparams,
    panel_label,
    save_figure,
    style_axis,
)


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs"
OUT_DIR = ROOT_DIR / "outputs"

METHODS = ("Play2Win", "Scratch task", "Scratch multi-stage")

TASKS = (
    "FMB Triangle Peg",
    "FMB Square-Circle Peg",
    "Fabrica Pillar",
    "Fabrica Beam",
    "FurnitureBench Screwing",
)
TASK_CAD_IMAGES = (
    "task_cad_fmb_triangle_peg.png",
    "task_cad_fmb_square_circle_peg.png",
    "task_cad_fabrica_pillar.png",
    "task_cad_fabrica_beam.png",
    "task_cad_furniturebench_screwing.png",
)
QUAL_IMAGES = (
    ("play2win_strategy.png", "Play2Win", "Play2Win"),
    ("scratch_strategy.png", "Scratch", "Scratch multi-stage"),
)

DUMMY_TOP_ROW = {
    "time_h": np.linspace(0.0, 2.0, 9),
    "play2win_final": np.array([0.95, 0.91, 0.89, 0.93, 0.87]),
    "scratch_task": np.array([0.02, 0.01, 0.03, 0.02, 0.01]),
    "scratch_multistage": np.array([0.06, 0.04, 0.05, 0.07, 0.03]),
}

DUMMY_FIXTURED = {
    "time_h": np.linspace(0.0, 40.0, 41),
    "Play2Win": 0.92 - 0.08 * np.exp(-np.linspace(0.0, 40.0, 41) / 2.5),
    "Scratch task": np.full(41, 0.02),
    "Scratch multi-stage": 0.94 / (1.0 + np.exp(-(np.linspace(0.0, 40.0, 41) - 25.0) / 4.0)),
}

ROBUSTNESS_FORCE = {
    "force_n": np.array([0, 1, 2, 4, 6, 8, 10]),
    "Play2Win": np.array([0.96, 0.95, 0.94, 0.91, 0.87, 0.82, 0.76]),
    "Scratch multi-stage": np.array([0.92, 0.80, 0.61, 0.33, 0.16, 0.06, 0.02]),
}


def _scratch_alpha(method: str) -> float:
    return 0.6 if method.startswith("Scratch") else 1.0


def _plot_method_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, method: str,
                       markevery: int | None = None) -> None:
    color, linestyle, marker = STYLES[method]
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=markevery,
        linewidth=1.8,
        markersize=3.5,
        markeredgewidth=0.0,
        alpha=_scratch_alpha(method),
        label=method,
    )


def _plot_top_task(ax: plt.Axes, task_idx: int) -> None:
    time_h = DUMMY_TOP_ROW["time_h"]
    final = DUMMY_TOP_ROW["play2win_final"][task_idx]
    play2win = final - (final - 0.50) * np.exp(-2.0 * time_h)
    scratch_task = DUMMY_TOP_ROW["scratch_task"][task_idx] + 0.004 * np.sin(3.0 * time_h)
    scratch_multi = DUMMY_TOP_ROW["scratch_multistage"][task_idx] + 0.006 * np.cos(2.0 * time_h)

    curves = {
        "Play2Win": play2win,
        "Scratch task": scratch_task,
        "Scratch multi-stage": scratch_multi,
    }
    for method in METHODS:
        _plot_method_curve(ax, time_h, 100.0 * curves[method], method)

    task = TASKS[task_idx]
    ax.set_title(TASK_DISPLAY_NAMES[task], fontsize=9, fontweight="bold",
                 color=TASK_COLORS[task], pad=3)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    style_axis(ax)
    _draw_task_cad_inset(ax, INPUT_DIR / TASK_CAD_IMAGES[task_idx], TASKS[task_idx])


def _draw_task_cad_inset(ax: plt.Axes, path: Path, task: str) -> None:
    inset = ax.inset_axes([0.62, 0.08, 0.33, 0.30])
    if path.exists():
        inset.imshow(plt.imread(path))
    else:
        inset.set_facecolor("#F3F3F3")
        inset.text(
            0.5,
            0.5,
            f"Missing\n{path.name}",
            transform=inset.transAxes,
            ha="center",
            va="center",
            fontsize=4.6,
            color="0.4",
        )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(TASK_COLORS[task])
        spine.set_linewidth(0.7)


def _plot_fixtured_task(ax: plt.Axes) -> None:
    time_h = DUMMY_FIXTURED["time_h"]
    for method in METHODS:
        _plot_method_curve(ax, time_h, 100.0 * DUMMY_FIXTURED[method], method, markevery=5)

    ax.axvline(2.0, color=COLORS["spine"], linewidth=0.8, linestyle=":")
    ax.text(2.5, 25.0, "2 h budget", fontsize=9, color="#444444", va="top", ha="left")
    ax.set_xlabel("Training time (h)")
    ax.set_ylabel("Success (%)")
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 2, 20, 40])
    ax.set_yticks([0, 50, 100])
    style_axis(ax)


def _plot_force_robustness(ax: plt.Axes) -> None:
    force_n = ROBUSTNESS_FORCE["force_n"]
    for method in ("Play2Win", "Scratch multi-stage"):
        _plot_method_curve(ax, force_n, 100.0 * ROBUSTNESS_FORCE[method], method)

    ax.set_xlabel("Perturbation force (N)")
    ax.set_ylabel("Success (%)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 50, 100])
    style_axis(ax)


def _draw_strategy_frame(ax: plt.Axes, path: Path, caption: str, method: str) -> None:
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
            color="0.4",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    color = STYLES[method][0]
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(0.9)


def _draw_qualitative_panel(fig: plt.Figure, spec) -> None:
    # Label row above the two cells so the captions can't bleed into each other
    # at any cell width.
    grid = spec.subgridspec(2, 2, height_ratios=[0.18, 1.0], hspace=0.06, wspace=0.10)
    for idx, (filename, caption, method) in enumerate(QUAL_IMAGES):
        label_ax = fig.add_subplot(grid[0, idx])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.3,
            caption,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=STYLES[method][0],
        )
        ax = fig.add_subplot(grid[1, idx])
        _draw_strategy_frame(ax, INPUT_DIR / filename, caption, method)


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.8, 4.8))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        left=0.07,
        right=0.97,
        bottom=0.11,
        top=0.83,
        hspace=0.85,
    )
    top_grid = outer[0].subgridspec(1, 5, wspace=0.42)
    bottom_grid = outer[1].subgridspec(1, 3, width_ratios=[1.30, 1.0, 1.0], wspace=0.48)

    panel_label(fig, 0.06, 0.96, "(a) Five assembly tasks, 2 h RL budget")
    panel_label(fig, 0.06, 0.475, "(b) Peg-in-Hole Fixtured")
    panel_label(fig, 0.46, 0.475, "(c) Force robustness")
    panel_label(fig, 0.74, 0.475, "(d) Qualitative strategy")

    top_axes = []
    for idx in range(len(TASKS)):
        ax = fig.add_subplot(top_grid[0, idx])
        _plot_top_task(ax, idx)
        if idx == 0:
            ax.set_ylabel("Success (%)")
        else:
            ax.set_yticklabels([])
        if idx == 2:
            ax.set_xlabel("Training time (h)")
        top_axes.append(ax)

    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.93),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    fixtured_ax = fig.add_subplot(bottom_grid[0, 0])
    _plot_fixtured_task(fixtured_ax)
    force_ax = fig.add_subplot(bottom_grid[0, 1])
    _plot_force_robustness(force_ax)
    _draw_qualitative_panel(fig, bottom_grid[0, 2])
    return fig


def main() -> None:
    configure_rcparams()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = _draw_full_figure()
    save_figure(fig, "fig2_scratch_vs_play2win_styled", out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
