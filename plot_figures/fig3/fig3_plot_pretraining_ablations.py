"""Figure 3 draft: ablations of pre-training choices.

This figure uses dummy values to show two expected trends over the same five
downstream assembly tasks:

- More object diversity during pre-training improves fine-tuning efficiency.
- The full Play2Win objective is more sample-efficient than objective ablations.

    python plot_figures/fig3_plot_pretraining_ablations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "outputs"

TASKS = (
    "FMB Triangle Peg",
    "FMB Square-Circle Peg",
    "Fabrica Pillar",
    "Fabrica Beam",
    "FurnitureBench Screwing",
)

DIVERSITY_METHODS = ("1 obj", "10 obj", "100 obj", "1000 obj")
OBJECTIVE_METHODS = ("Play2Win", "TranslationOnly", "RotationOnly", "SingleGoal")

DIVERSITY_STYLE = {
    "1 obj": {"color": "#D73027", "linestyle": "--", "marker": None},
    "10 obj": {"color": "#F46D43", "linestyle": "-.", "marker": None},
    "100 obj": {"color": "#74ADD1", "linestyle": ":", "marker": None},
    "1000 obj": {"color": "#1A9850", "linestyle": "-", "marker": "o"},
}

OBJECTIVE_STYLE = {
    "Play2Win": {"color": "#1A9850", "linestyle": "-", "marker": "o"},
    "TranslationOnly": {"color": "#2C7BB6", "linestyle": "--", "marker": None},
    "RotationOnly": {"color": "#7B3294", "linestyle": "-.", "marker": None},
    "SingleGoal": {"color": "#D73027", "linestyle": ":", "marker": None},
}

DUMMY_FINAL_SUCCESS = np.array([0.94, 0.90, 0.88, 0.92, 0.86])


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


def _efficiency_curve(time_h: np.ndarray, final: float, start: float, rate: float) -> np.ndarray:
    return final - (final - start) * np.exp(-rate * time_h)


def _plot_diversity_ablation(ax: plt.Axes, task_idx: int) -> None:
    time_h = np.linspace(0.0, 2.0, 9)
    final = DUMMY_FINAL_SUCCESS[task_idx]
    starts = {
        "1 obj": 0.05,
        "10 obj": 0.12,
        "100 obj": 0.28,
        "1000 obj": 0.50,
    }
    rates = {
        "1 obj": 0.35,
        "10 obj": 0.75,
        "100 obj": 1.55,
        "1000 obj": 2.50,
    }
    final_scales = {
        "1 obj": 0.38,
        "10 obj": 0.58,
        "100 obj": 0.82,
        "1000 obj": 1.00,
    }

    for method in DIVERSITY_METHODS:
        style = DIVERSITY_STYLE[method]
        curve = _efficiency_curve(time_h, final * final_scales[method], starts[method], rates[method])
        ax.plot(
            time_h,
            100.0 * curve,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.55,
            markersize=3.0,
            markeredgewidth=0.0,
        )

    ax.set_title(TASKS[task_idx], fontsize=7.4, fontweight="bold", pad=3)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    _style_axis(ax)


def _plot_objective_ablation(ax: plt.Axes, task_idx: int) -> None:
    time_h = np.linspace(0.0, 2.0, 9)
    final = DUMMY_FINAL_SUCCESS[task_idx]
    starts = {
        "Play2Win": 0.50,
        "TranslationOnly": 0.30,
        "RotationOnly": 0.22,
        "SingleGoal": 0.16,
    }
    rates = {
        "Play2Win": 2.40,
        "TranslationOnly": 1.25,
        "RotationOnly": 1.00,
        "SingleGoal": 0.70,
    }
    final_scales = {
        "Play2Win": 1.00,
        "TranslationOnly": 0.78,
        "RotationOnly": 0.68,
        "SingleGoal": 0.55,
    }

    for method in OBJECTIVE_METHODS:
        style = OBJECTIVE_STYLE[method]
        curve = _efficiency_curve(time_h, final * final_scales[method], starts[method], rates[method])
        ax.plot(
            time_h,
            100.0 * curve,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.55,
            markersize=3.0,
            markeredgewidth=0.0,
        )

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    _style_axis(ax)


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(8.0, 4.15))
    grid = fig.add_gridspec(
        2,
        5,
        left=0.058,
        right=0.99,
        bottom=0.12,
        top=0.86,
        hspace=0.58,
        wspace=0.34,
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

    fig.text(
        0.058,
        0.955,
        "(a) Object diversity during pre-training",
        ha="left",
        va="center",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.058,
        0.500,
        "(b) Play objective during pre-training",
        ha="left",
        va="center",
        fontsize=9.4,
        fontweight="bold",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.965),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.0,
    )

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.99, 0.505),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        borderaxespad=0.0,
    )
    return fig


def main() -> None:
    _configure_rcparams()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = _draw_full_figure()

    png_path = OUT_DIR / "fig3_pretraining_ablations.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
