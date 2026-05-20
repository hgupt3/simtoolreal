"""Styled Figure 1 — contact-tolerance evaluation for peg-in-hole.

Same one-row layout as the original draft, but with re-planned dimensions:

- Figure: 7.8 x 3.6 in (balanced height).
- Real-world cells: 4:3 landscape aspect (wider than tall) so the workspace
  crop fits naturally; the image grid only occupies the upper portion of
  the left panel so the cells stay proportional.
- Right column plots use the full figure height, giving each stacked plot
  more room than the original.
- Blue/orange palette via ``_style.py``; no gridlines; frameless legend.

Output: ``outputs/fig1_contact_tolerances_styled.png`` at dpi=600.

    python plot_figures/fig1_plot_contact_tolerances_styled.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import NullFormatter

# _style.py lives one directory up (plot_figures/), shared across all figs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import COLORS, configure_rcparams, panel_label, save_figure, style_axis


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "inputs"
OUT_PATH = ROOT_DIR / "outputs" / "fig1_contact_tolerances_styled.png"

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

METHOD_COLOR = {
    "Play-only": COLORS["play_only"],
    "Play2Win": COLORS["play2win"],
}

# Real sim measurements are loaded from ``inputs/fig1/sim_results.json``
# (produced by ``fig1_run_lpeg_tol_sweep.py``). Method-key mapping between
# the JSON ("play2win" / "play_only") and the figure labels ("Play2Win" /
# "Play-only").
JSON_METHOD_KEY = {
    "Play-only": "play_only",
    "Play2Win": "play2win",
}

# The sweep stores tolerance as **per-side clearance** (peg wall to hole wall),
# matching the asset naming `hole_tolXmm`. In the paper figure we report
# *diametral* clearance (slot width minus peg width = 2x per-side), since that
# is the conventional way mechanical fits are described in the literature.
TOLERANCE_PAPER_SCALE = 2.0
TOLERANCE_PAPER_LABEL = "Clearance (mm)"


def _load_sim_results(path: Path, *, prefer_filtered: bool = True) -> dict:
    """Load the aggregate sweep JSON. Returns dict with keys:

      - ``tolerances_mm`` (list[float])
      - ``Play-only``  / ``Play2Win`` -> {tolerances_mm, success_rate,
        throughput_per_min} with ``None`` entries filtered out per series.

    When ``prefer_filtered`` is True (default), we prefer the ``filtered``
    sub-block of each method, which excludes envs whose first episode
    terminated very early (init-fall artifacts). Falls back to the raw
    series if the filtered block isn't present.
    """
    if not path.exists():
        return {"tolerances_mm": [], "Play-only": None, "Play2Win": None}
    raw = json.loads(path.read_text())
    tols_per_side = list(raw.get("tolerances_mm") or [])
    tols = [TOLERANCE_PAPER_SCALE * float(t) for t in tols_per_side]
    out: dict = {"tolerances_mm": tols}
    for label, key in JSON_METHOD_KEY.items():
        block = raw.get(key) or {}
        if prefer_filtered and isinstance(block.get("filtered"), dict):
            src = block["filtered"]
        else:
            src = block
        sr = src.get("success_rate") or []
        tp = src.get("throughput_per_min") or []
        if not sr:
            out[label] = None
            continue
        kept_tols, kept_sr, kept_tp = [], [], []
        for t, s, p in zip(tols, sr, tp):
            # ``tols`` is already scaled to the paper convention above.
            if s is None or p is None:
                continue
            kept_tols.append(float(t))
            kept_sr.append(float(s))
            kept_tp.append(float(p))
        out[label] = {
            "tolerances_mm": np.asarray(kept_tols, dtype=float),
            "success_rate": np.asarray(kept_sr, dtype=float),
            "throughput_per_min": np.asarray(kept_tp, dtype=float),
        }
    return out

# ----------------------------------------------------------------------------
# Layout constants (all in figure-relative coordinates unless noted)
# ----------------------------------------------------------------------------
FIG_W_IN = 9.5
FIG_H_IN = 3.7

# Left panel width is tuned so the cells come out roughly square at the same
# vertical extent as the right plots' axes boxes (1.27 in tall). Right panel
# is unchanged.
LEFT_X0, LEFT_X1 = 0.05, 0.74
RIGHT_X0, RIGHT_X1 = 0.82, 0.99

# Image cell grid is aligned vertically with the right axes boxes (top=0.87,
# bottom=0.13), so the two halves of the figure read as a single horizontal
# band. Column headers ("Hole", "Play-only", "Play2Win") sit above the cells
# in their own strip, analogous to how the legend rides above the right plots.
CELL_TOP = 0.87
CELL_BOTTOM = 0.13
HEADER_TOP = 0.95
HEADER_BOTTOM = 0.88

# Tighter intra-grid gaps reduce inter-cell whitespace.
INTRA_ROW_GAP = 0.10
INTRA_COL_GAP = 0.10
GRID_BOTTOM = CELL_BOTTOM


_X_TICKS = [40.0, 20.0, 10.0, 4.0, 2.0, 1.0, 0.4, 0.2]
_X_TICK_LABELS = ["40", "20", "10", "4", "2", "1", "0.4", "0.2"]


def _style_tolerance_axis(ax: plt.Axes, all_tols: np.ndarray, *, show_xlabel: bool) -> None:
    """Shared x-axis treatment for both right-column plots."""
    ax.set_xscale("log")
    if all_tols.size:
        x_lo = max(float(all_tols.min()) * 0.9, 1e-3)
        x_hi = float(all_tols.max()) * 1.1
    else:
        x_lo, x_hi = 0.1, 5.0
    # High tol on the left, tight tol on the right: easier -> harder.
    ax.set_xlim(x_hi, x_lo)
    ax.set_xticks(_X_TICKS)
    ax.set_xticklabels(_X_TICK_LABELS)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if show_xlabel:
        ax.set_xlabel(TOLERANCE_PAPER_LABEL)


def _plot_success_rate(ax: plt.Axes, sim_results: dict) -> None:
    series = [(label, sim_results.get(label)) for label in METHODS]
    all_tols = np.concatenate(
        [s["tolerances_mm"] for _, s in series if s is not None],
        axis=0,
    ) if any(s is not None for _, s in series) else np.asarray([])

    for method, payload in series:
        if payload is None:
            continue
        ax.plot(
            payload["tolerances_mm"],
            100.0 * payload["success_rate"],
            color=METHOD_COLOR[method],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            markeredgewidth=0.0,
            label=method,
            zorder=3,
        )

    ax.set_title("Success Rate (Sim)", fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0.0, 104.0)
    ax.set_yticks([0, 50, 100])
    _style_tolerance_axis(ax, all_tols, show_xlabel=True)
    style_axis(ax)


def _plot_throughput_line(ax: plt.Axes, sim_results: dict) -> None:
    series = [(label, sim_results.get(label)) for label in METHODS]
    all_tols = np.concatenate(
        [s["tolerances_mm"] for _, s in series if s is not None],
        axis=0,
    ) if any(s is not None for _, s in series) else np.asarray([])
    all_thr = np.concatenate(
        [s["throughput_per_min"] for _, s in series if s is not None],
        axis=0,
    ) if any(s is not None for _, s in series) else np.asarray([])

    for method, payload in series:
        if payload is None:
            continue
        ax.plot(
            payload["tolerances_mm"],
            payload["throughput_per_min"],
            color=METHOD_COLOR[method],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            markeredgewidth=0.0,
            label=method,
            zorder=3,
        )

    ax.set_title("Throughput (Sim)", fontsize=11, fontweight="bold", pad=4)
    ax.set_ylabel("Successes / min")
    y_max = float(all_thr.max()) if all_thr.size else 1.0
    # Nudge the top so the highest marker isn't flush with the spine; choose a
    # tick step that gives ~4 ticks regardless of throughput magnitude.
    ax.set_ylim(0.0, y_max * 1.15 + 1e-6)
    _style_tolerance_axis(ax, all_tols, show_xlabel=True)
    style_axis(ax)


def _draw_image_cell(
    ax: plt.Axes,
    path: Path,
    border_color: str | None = None,
    show_placeholder_image: bool = False,
) -> None:
    """Draw a real-world cell. By default the placeholder PNG is *not* shown
    (placeholders bake in text like ``Play-only 2 mm / 1`` that overlaps the
    cell once rendered). Real images can be wired in later by setting
    ``show_placeholder_image=True`` once final renders land in ``inputs/fig1``.
    """
    if show_placeholder_image and path.exists():
        ax.imshow(plt.imread(path), aspect="auto")
    else:
        # clean light tint based on the border color (or neutral gray if none).
        if border_color is None:
            ax.set_facecolor("#F3F3F3")
        else:
            # 8%-saturation tint of the border color: keeps the orange/blue
            # encoding while letting real workspace photos drop in cleanly.
            import matplotlib.colors as mcolors

            rgb = mcolors.to_rgb(border_color)
            tinted = tuple(1.0 - (1.0 - c) * 0.10 for c in rgb)
            ax.set_facecolor(tinted)
    ax.set_xticks([])
    ax.set_yticks([])

    if border_color is None:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(1.0)


def _build_left_panel(fig: plt.Figure) -> None:
    # Separate gridspecs for headers vs cells so the cell grid can be
    # pinned to the right plots' vertical extent without the header strip
    # eating into that range.
    header_grid = fig.add_gridspec(
        1,
        5,
        left=LEFT_X0,
        right=LEFT_X1,
        top=HEADER_TOP,
        bottom=HEADER_BOTTOM,
        wspace=INTRA_COL_GAP,
    )
    cell_grid = fig.add_gridspec(
        2,
        5,
        left=LEFT_X0,
        right=LEFT_X1,
        top=CELL_TOP,
        bottom=CELL_BOTTOM,
        hspace=INTRA_ROW_GAP,
        wspace=INTRA_COL_GAP,
    )

    headers = [
        (0, 1, "Hole", "#222222"),
        (1, 3, "Play-only", METHOD_COLOR["Play-only"]),
        (3, 5, "Play2Win", METHOD_COLOR["Play2Win"]),
    ]
    for col_start, col_end, title, color in headers:
        ax = fig.add_subplot(header_grid[0, col_start:col_end])
        ax.axis("off")
        ax.text(0.5, 0.4, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

    for row_idx, (_label, suffix) in enumerate(TOLERANCE_ROWS):
        hole_ax = fig.add_subplot(cell_grid[row_idx, 0])
        # Hole cells: show the placeholder image (it carries the tolerance label
        # baked into the render); real hole-CAD images can drop in unchanged.
        _draw_image_cell(
            hole_ax,
            INPUT_DIR / EXPECTED_IMAGES[f"hole_{suffix}"],
            show_placeholder_image=True,
        )

        policy_specs = [
            (1, "play_only", "Play-only", "1"),
            (2, "play_only", "Play-only", "2"),
            (3, "play2win", "Play2Win", "1"),
            (4, "play2win", "Play2Win", "2"),
        ]
        for col, file_prefix, method, _image_label in policy_specs:
            ax = fig.add_subplot(cell_grid[row_idx, col])
            # Policy cells: clean tinted background until real workspace photos
            # are added. Avoids the placeholder PNG's baked-in text overlapping
            # the cell. Flip ``show_placeholder_image=True`` once real renders
            # are in ``inputs/fig1/``.
            _draw_image_cell(
                ax,
                INPUT_DIR / EXPECTED_IMAGES[f"{file_prefix}_{suffix}_{_image_label}"],
                border_color=METHOD_COLOR[method],
                show_placeholder_image=False,
            )


def _build_right_panel(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    grid = fig.add_gridspec(
        2,
        1,
        left=RIGHT_X0,
        right=RIGHT_X1,
        # Top of the gridspec sits below the legend with breathing room.
        # Vertical stack from figure top down: legend (y≈0.96) → gap →
        # title (above the top plot) → top plot.
        top=0.87,
        bottom=GRID_BOTTOM,
        # Gap now only holds: top plot's x-axis ticks + label + bottom title.
        hspace=0.7,
    )
    sim_ax = fig.add_subplot(grid[0, 0])
    throughput_ax = fig.add_subplot(grid[1, 0])
    return sim_ax, throughput_ax


def _draw_hole_policy_separator(fig: plt.Figure, sim_ax: plt.Axes, throughput_ax: plt.Axes) -> None:
    fig.canvas.draw()
    image_cells = []
    for ax in fig.axes:
        if ax is sim_ax or ax is throughput_ax or not ax.axison:
            continue
        if len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0:
            image_cells.append(ax.get_position())
    if not image_cells:
        return
    xs = sorted({round(b.x0, 4) for b in image_cells})
    if len(xs) < 2:
        return
    hole_cells = [b for b in image_cells if abs(b.x0 - xs[0]) < 1e-3]
    policy_cells = [b for b in image_cells if abs(b.x0 - xs[1]) < 1e-3]
    sep_x = 0.5 * (max(b.x1 for b in hole_cells) + min(b.x0 for b in policy_cells))
    y_top = max(b.y1 for b in image_cells)
    y_bot = min(b.y0 for b in image_cells)
    fig.add_artist(
        plt.Line2D(
            [sep_x, sep_x],
            [y_bot, y_top],
            transform=fig.transFigure,
            color="#CCCCCC",
            linestyle=":",
            linewidth=0.6,
        )
    )


def _annotate_reversed_axis(fig: plt.Figure, sim_ax: plt.Axes) -> None:
    fig.canvas.draw()
    bbox = sim_ax.get_position()
    y = bbox.y0 - 0.075
    x_left = bbox.x0 + 0.005
    x_right = bbox.x1 - 0.005
    arrow = FancyArrowPatch(
        (x_left, y),
        (x_right, y),
        transform=fig.transFigure,
        arrowstyle="<|-|>",
        mutation_scale=8,
        color=COLORS["spine"],
        linewidth=0.8,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(arrow)
    fig.text(x_left, y - 0.038, "easier", fontsize=10, color="#444444", ha="left", va="top")
    fig.text(x_right, y - 0.038, "harder", fontsize=10, color="#444444", ha="right", va="top")


def main() -> None:
    configure_rcparams()
    # Plot-axes typography is dialed down vs the global rcParams so the rotated
    # y-labels and tick labels don't dominate the small plot column. Header /
    # panel-level text in `_style.py` stays at the user-requested large sizes.
    plt.rcParams.update(
        {
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))

    _build_left_panel(fig)
    sim_ax, throughput_ax = _build_right_panel(fig)

    sim_results = _load_sim_results(INPUT_DIR / "sim_results.json")
    _plot_success_rate(sim_ax, sim_results)
    _plot_throughput_line(throughput_ax, sim_results)
    # Align the two y-axis labels — without this, "Success rate (%)" sits
    # further left than "Successes / min" because the y-tick numbers "100"
    # are wider than "20".
    fig.align_ylabels([sim_ax, throughput_ax])

    # Divider between the qualitative section (left, image grid) and the
    # quantitative section (right, sim plots). Sits just past the right
    # edge of the image grid (LEFT_X1) so it lands to the LEFT of the right
    # plots' y-axis labels and tick labels (which live in [LEFT_X1, RIGHT_X0]).
    _div_x = LEFT_X1 + 0.003
    # Y-extent is symmetric about the figure center so the divider has equal
    # padding from the topmost and bottommost edges of the saved PNG.
    # bbox_inches="tight" pads pad_inches on each side equally, so as long as
    # the line is the topmost/bottommost figure-edge artist (or matches the
    # outermost ink on both sides), the visible top/bottom padding matches.
    _div_pad = 0.02
    fig.add_artist(
        plt.Line2D(
            [_div_x, _div_x],
            [_div_pad, 1.0 - _div_pad],
            transform=fig.transFigure,
            color="#888888",
            linestyle="-",
            linewidth=1.2,
        )
    )

    # Legend sits at the very top of the right column, above the top plot's
    # "Success Rate (Sim)" title. HEADER_TOP - 0.025 leaves a small padding
    # above the legend and keeps it inside the figure bounds.
    handles, labels = sim_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5 * (RIGHT_X0 + RIGHT_X1) - 0.015, 0.96),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    _draw_hole_policy_separator(fig, sim_ax, throughput_ax)
    # Removed the `easier <- -> harder` arrow annotation: it consumed a chunk
    # of vertical space between the two plots. The tolerance-axis ordering
    # (2 -> 0.2 mm) is shared with the throughput-bar x-axis below, so the
    # reversal reads naturally without an explicit arrow.

    fig.savefig(
        OUT_PATH,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
