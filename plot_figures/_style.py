"""Shared style module for the paper figures.

Pulls conventions from /share/portal/kk837/sapg/plot_figures (Times New Roman
serif body, STIX math, hidden top/right spines, soft #333 spine/tick color,
no-frame fig-level legends, dpi=600 saves). Font sizes are scaled up ~1.25x
from sapg's 8/9/10/7.5pt because our target is a 6.5 in single-column figure
rather than sapg's 3.5 in two-column figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def configure_rcparams() -> None:
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Nimbus Roman",
        "Liberation Serif",
    ]
    matplotlib.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


COLORS: dict[str, str] = {
    "play2win": "#2C7BB6",   # blue (hero)
    "play_only": "#E08214",  # orange (foil)
    "scratch_task": "#E08214",
    "scratch_multistage": "#9970AB",
    "teleop_pjaw": "#E08214",
    "human": "#4D4D4D",
    "sim": "#92C5DE",
    "real": "#2C7BB6",
    "separator": "#888888",
    "spine": "#333333",
}

DIVERSITY_RAMP: list[str] = ["#BAE4B3", "#74C476", "#31A354", "#006D2C"]

# Cool blue-leaning ramp for the 5 assembly tasks. Since Play2Win is the
# only method being measured across these tasks, the palette stays
# tonally consistent with our hero color (blue) rather than spanning the
# full rainbow.
TASK_COLORS: dict[str, str] = {
    "FMB Triangle Peg": "#08519C",        # deep blue
    "FMB Square-Circle Peg": "#3182BD",   # medium blue
    "Fabrica Pillar": "#6BAED6",          # light blue
    "Fabrica Beam": "#41B6C4",            # teal
    "FurnitureBench Screwing": "#7A0177", # cool purple accent
}

# Shorter display titles for the same 5 tasks. Use these for per-panel titles
# in small-multiple layouts where the full names visually collide.
TASK_DISPLAY_NAMES: dict[str, str] = {
    "FMB Triangle Peg": "FMB Triangle",
    "FMB Square-Circle Peg": "FMB Sq.-Circle",
    "Fabrica Pillar": "Pillar",
    "Fabrica Beam": "Beam",
    "FurnitureBench Screwing": "Furniture Screw",
}

STYLES: dict[str, tuple[str, str, str | None]] = {
    "Play2Win": (COLORS["play2win"], "-", "o"),
    "Play-only": (COLORS["play_only"], "--", "s"),
    "Scratch task": (COLORS["scratch_task"], "--", "s"),
    "Scratch multi-stage": (COLORS["scratch_multistage"], "-.", "^"),
    "TranslationOnly": ("#2C7BB6", "--", "s"),
    "RotationOnly": ("#7B3294", "-.", "^"),
    "SingleGoal": ("#D6604D", ":", "D"),
}


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(COLORS["spine"])
    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors=COLORS["spine"])


def panel_label(fig: plt.Figure, x: float, y: float, text: str) -> None:
    fig.text(x, y, text, ha="left", va="center", fontsize=12, fontweight="bold", color="#222222")


def save_figure(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Wrote {path}")
    return path
