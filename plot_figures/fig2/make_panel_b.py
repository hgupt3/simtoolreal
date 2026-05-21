"""Render panel (b) of fig 2: training-time success curves on the toy fixtured task.

Methods:
    Play2Win (blue)
    Scratch (dense reward) (orange)
    Scratch (task reward) (red)

Annotation "Matches Scratch in 4.1h (Nx faster)" is drawn in the bottom-left
interior at (x=40h, y=13%). A short dashed vertical at the crossing time marks
where Play2Win first reaches Scratch (dense reward)'s terminal success rate.

Reads:
    outputs/fig2_panel_bcd/panel_b_curves.json

Writes:
    plot_figures/fig2/outputs/panel_b_draft.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "fig2_panel_bcd"
OUT = REPO / "plot_figures" / "fig2" / "outputs"

LABELS = {
    "Play2Win":           ("Play2Win",                "#2C7BB6"),
    "Scratch_multistage": ("Scratch (dense reward)",  "#E08214"),
    "Scratch_task":       ("Scratch (task reward)",   "#D6604D"),
}
X_MAX = 140.0
LW = 1.5


def main():
    curves = json.load(open(DATA_DIR / "panel_b_curves.json"))
    target_ratio = curves["Scratch_multistage"][-1][1]
    x_cross = next(t for t, s in curves["Play2Win"] if s >= target_ratio)
    speedup = curves["Scratch_multistage"][-1][0] / x_cross
    y_cross = target_ratio * 100

    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    for key, (label, c) in LABELS.items():
        pts = curves[key]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        ax.plot(xs, ys, color=c, linewidth=LW, label=label)
        if key == "Play2Win" and xs[-1] < X_MAX:
            ax.plot([xs[-1], X_MAX], [ys[-1], ys[-1]], color=c, linewidth=LW)

    ax.plot([x_cross, x_cross], [-2, y_cross], color="#555555", linestyle="--", linewidth=0.8)
    ax.text(
        40, 13,
        f"Matches Scratch in\n{x_cross:.1f}h ({speedup:.0f}× faster)",
        fontsize=8.5, color="#333333", ha="center", va="center", multialignment="center",
    )

    ax.set_xlabel("Training wall-clock time", fontsize=9)
    ax.set_ylabel("Success rate", fontsize=9)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=8)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_xticklabels([f"{v} h" for v in (0, 40, 80, 120)], fontsize=8)
    ax.legend(
        loc="center right", bbox_to_anchor=(0.99, 0.50),
        frameon=False, fontsize=8,
        handlelength=1.4, handletextpad=0.5, borderaxespad=0.2,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "panel_b_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_b_draft.png'}")


if __name__ == "__main__":
    main()
