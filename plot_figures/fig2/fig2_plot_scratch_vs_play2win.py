"""Figure 2 draft: Play2Win versus training from scratch.

Dummy values encode the intended story:

- Across five assembly tasks, scratch policies do not learn within a 2 h RL
  budget, while Play2Win transfers quickly.
- On an easier toy task, Scratch + multi-stage rewards eventually solves the
  task, but only after a much longer horizon.
- The learned scratch strategy is qualitatively odd and brittle to small
  physics variations.

    python plot_figures/fig2_plot_scratch_vs_play2win.py
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

METHODS = ("Play2Win", "Scratch task", "Scratch multi-stage")
STYLE = {
    "Play2Win": {"color": "#1A9850", "linewidth": 1.8, "linestyle": "-", "marker": "o"},
    "Scratch task": {"color": "#D73027", "linewidth": 1.5, "linestyle": "--", "marker": None},
    "Scratch multi-stage": {"color": "#6A51A3", "linewidth": 1.5, "linestyle": "-.", "marker": None},
}

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
    ("scratch_strategy.png", "Scratch multi-stage", "Scratch multi-stage"),
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


def _configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", labelsize=7.2, width=0.75, length=2.7)


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
        style = STYLE[method]
        ax.plot(
            time_h,
            100.0 * curves[method],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2,
            markeredgewidth=0.0,
            label=method,
        )

    ax.set_title(TASKS[task_idx], fontsize=7.6, fontweight="bold", pad=3)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 50, 100])
    _style_axis(ax)
    _draw_task_cad_inset(ax, INPUT_DIR / TASK_CAD_IMAGES[task_idx])


def _draw_task_cad_inset(ax: plt.Axes, path: Path) -> None:
    inset = ax.inset_axes([0.63, 0.08, 0.32, 0.27])
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
            color="0.38",
        )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("0.70")
        spine.set_linewidth(0.55)


def _plot_fixtured_task(ax: plt.Axes) -> None:
    time_h = DUMMY_FIXTURED["time_h"]
    for method in METHODS:
        style = STYLE[method]
        ax.plot(
            time_h,
            100.0 * DUMMY_FIXTURED[method],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=5,
            markersize=3.0,
            markeredgewidth=0.0,
            label=method,
        )

    ax.axvline(2.0, color="0.35", linewidth=0.8, linestyle=":")
    ax.text(2.6, 86, "2 h budget", fontsize=6.8, color="0.25", va="center")
    ax.set_title("Peg in Hole Fixtured", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.set_xlabel("Training time (h)")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(0.0, 102.0)
    ax.set_xticks([0, 2, 20, 40])
    ax.set_yticks([0, 50, 100])
    _style_axis(ax)


def _draw_strategy_frame(ax: plt.Axes, path: Path, caption: str, method: str) -> None:
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
            fontsize=5.8,
            color="0.35",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(STYLE[method]["color"])
        spine.set_linewidth(1.6)
    ax.set_title(caption, fontsize=6.6, fontweight="bold", pad=2)


def _draw_qualitative_panel(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(2, 2, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.08)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.5, "Qualitative strategy", ha="left", va="center", fontsize=8, fontweight="bold")

    for idx, (filename, caption, method) in enumerate(QUAL_IMAGES):
        ax = fig.add_subplot(grid[1, idx])
        _draw_strategy_frame(ax, INPUT_DIR / filename, caption, method)


def _plot_force_robustness(ax: plt.Axes) -> None:
    force_n = ROBUSTNESS_FORCE["force_n"]
    for method in ("Play2Win", "Scratch multi-stage"):
        style = STYLE[method]
        ax.plot(
            force_n,
            100.0 * ROBUSTNESS_FORCE[method],
            label=method,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.0,
        )

    ax.set_title("Force robustness", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.set_xlabel("Perturbation force (N)")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 50, 100])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.00, 0.02), handlelength=1.6, borderaxespad=0.0)
    _style_axis(ax)


def _draw_full_figure() -> plt.Figure:
    fig = plt.figure(figsize=(8.0, 4.95))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.18],
        left=0.055,
        right=0.99,
        bottom=0.08,
        top=0.875,
        hspace=0.62,
    )
    top = outer[0].subgridspec(1, 5, wspace=0.32)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.25, 1.10, 1.06], wspace=0.36)

    fig.text(0.055, 0.965, "(a) Five assembly tasks, 2 h RL budget", fontsize=9.5, fontweight="bold", ha="left")
    fig.text(0.055, 0.492, "(b) Easy toy task exposes scratch-policy limits", fontsize=9.5, fontweight="bold", ha="left")

    top_axes = []
    for idx in range(len(TASKS)):
        ax = fig.add_subplot(top[0, idx])
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
        bbox_to_anchor=(0.987, 0.947),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    fixtured_ax = fig.add_subplot(bottom[0, 0])
    _plot_fixtured_task(fixtured_ax)
    _draw_qualitative_panel(fig, bottom[0, 1])

    robustness_ax = fig.add_subplot(bottom[0, 2])
    _plot_force_robustness(robustness_ax)
    return fig


def main() -> None:
    _configure_rcparams()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = _draw_full_figure()
    png_path = OUT_DIR / "fig2_scratch_vs_play2win.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
