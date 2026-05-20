"""Styled Figure 3 — pre-training ablations.

Keeps the 2x5 small-multiples grid (per user preference, no across-task
aggregate). Row (a) uses a sequential green ramp for the ordinal "1 / 10 /
100 / 1000 obj" axis; row (b) uses a categorical palette with distinct
markers per objective. Per-task titles use ``TASK_COLORS`` (consistent
with Fig 2).

Output: ``outputs/fig3_pretraining_ablations_styled.png`` at dpi=600.

    python plot_figures/fig3_plot_pretraining_ablations_styled.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    DIVERSITY_RAMP,
    STYLES,
    TASK_COLORS,
    TASK_DISPLAY_NAMES,
    configure_rcparams,
    panel_label,
    save_figure,
    style_axis,
)


ROOT_DIR = Path(__file__).resolve().parent

TASKS = (
    "FMB Triangle Peg",
    "FMB Square-Circle Peg",
    "Fabrica Pillar",
    "Fabrica Beam",
    "FurnitureBench Screwing",
)

DIVERSITY_METHODS = ("1 obj", "10 obj", "100 obj", "1000 obj")
OBJECTIVE_METHODS = ("Play2Win", "TranslationOnly", "RotationOnly", "SingleGoal")

OBJECTIVE_MARKERS = {
    "Play2Win": "o",
    "TranslationOnly": "s",
    "RotationOnly": "^",
    "SingleGoal": "D",
}

DUMMY_FINAL_SUCCESS = np.array([0.94, 0.90, 0.88, 0.92, 0.86])


def _efficiency_curve(time_h: np.ndarray, final: float, start: float, rate: float) -> np.ndarray:
    return final - (final - start) * np.exp(-rate * time_h)


def _plot_diversity_ablation(ax: plt.Axes, task_idx: int) -> None:
    time_h = np.linspace(0.0, 2.0, 9)
    final = DUMMY_FINAL_SUCCESS[task_idx]
    starts = {"1 obj": 0.05, "10 obj": 0.12, "100 obj": 0.28, "1000 obj": 0.50}
    rates = {"1 obj": 0.35, "10 obj": 0.75, "100 obj": 1.55, "1000 obj": 2.50}
    final_scales = {"1 obj": 0.38, "10 obj": 0.58, "100 obj": 0.82, "1000 obj": 1.00}

    for method, color in zip(DIVERSITY_METHODS, DIVERSITY_RAMP):
        curve = _efficiency_curve(time_h, final * final_scales[method], starts[method], rates[method])
        ax.plot(
            time_h,
            100.0 * curve,
            label=method,
            color=color,
            linestyle="-",
            linewidth=1.8,
        )

    task = TASKS[task_idx]
    ax.set_title(TASK_DISPLAY_NAMES[task], fontsize=9, fontweight="bold",
                 color=TASK_COLORS[task], pad=3)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    style_axis(ax)


def _plot_objective_ablation(ax: plt.Axes, task_idx: int) -> None:
    time_h = np.linspace(0.0, 2.0, 9)
    final = DUMMY_FINAL_SUCCESS[task_idx]
    starts = {"Play2Win": 0.50, "TranslationOnly": 0.30, "RotationOnly": 0.22, "SingleGoal": 0.16}
    rates = {"Play2Win": 2.40, "TranslationOnly": 1.25, "RotationOnly": 1.00, "SingleGoal": 0.70}
    final_scales = {"Play2Win": 1.00, "TranslationOnly": 0.78, "RotationOnly": 0.68, "SingleGoal": 0.55}

    for method in OBJECTIVE_METHODS:
        color, linestyle, _ = STYLES[method]
        curve = _efficiency_curve(time_h, final * final_scales[method], starts[method], rates[method])
        ax.plot(
            time_h,
            100.0 * curve,
            label=method,
            color=color,
            linestyle=linestyle,
            marker=OBJECTIVE_MARKERS[method],
            linewidth=1.7,
            markersize=3.5,
            markevery=2,
            markeredgewidth=0.0,
        )

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    style_axis(ax)


def _draw_diversity_legend(fig: plt.Figure, x: float, y: float) -> None:
    """Inline horizontal colorbar swatch + '1 -> 1000 obj' label."""
    from matplotlib.patches import Rectangle

    swatch_w = 0.018
    swatch_h = 0.018
    fig.text(x, y, "1 obj", fontsize=9, ha="right", va="center", color="#222222")
    for idx, color in enumerate(DIVERSITY_RAMP):
        rect = Rectangle(
            (x + 0.006 + idx * swatch_w, y - swatch_h / 2),
            swatch_w,
            swatch_h,
            transform=fig.transFigure,
            color=color,
            linewidth=0,
        )
        fig.add_artist(rect)
    fig.text(x + 0.012 + len(DIVERSITY_RAMP) * swatch_w, y, "1000 obj",
             fontsize=9, ha="left", va="center", color="#222222")


def _draw_objective_legend(fig: plt.Figure, x: float, y: float) -> None:
    """Inline 'method marker' legend, no box."""
    entries = []
    for method in OBJECTIVE_METHODS:
        color, _, _ = STYLES[method]
        marker = OBJECTIVE_MARKERS[method]
        entries.append((method, color, marker))
    spacing = 0.115
    for idx, (method, color, marker) in enumerate(entries):
        cx = x + idx * spacing
        fig.text(cx, y, method, fontsize=9, ha="left", va="center", color="#222222")


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.8, 3.6))
    grid = fig.add_gridspec(
        2,
        5,
        left=0.08,
        right=0.96,
        bottom=0.16,
        top=0.82,
        hspace=0.75,
        wspace=0.32,
    )

    axes = np.empty((2, 5), dtype=object)
    for task_idx in range(len(TASKS)):
        axes[0, task_idx] = fig.add_subplot(grid[0, task_idx])
        axes[1, task_idx] = fig.add_subplot(grid[1, task_idx])
        _plot_diversity_ablation(axes[0, task_idx], task_idx)
        _plot_objective_ablation(axes[1, task_idx], task_idx)

        if task_idx == 0:
            axes[0, task_idx].set_ylabel("Success (%)")
            axes[1, task_idx].set_ylabel("Success (%)")
        else:
            axes[0, task_idx].set_yticklabels([])
            axes[1, task_idx].set_yticklabels([])
        if task_idx == 2:
            axes[1, task_idx].set_xlabel("Fine-tuning time (h)")

    panel_label(fig, 0.08, 0.945, "(a) Object diversity")
    _draw_diversity_legend(fig, 0.52, 0.945)

    panel_label(fig, 0.08, 0.46, "(b) Play objective")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.47),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    return fig


def main() -> None:
    configure_rcparams()
    fig = _draw_full_figure()
    save_figure(fig, "fig3_pretraining_ablations_styled")
    plt.close(fig)


if __name__ == "__main__":
    main()
