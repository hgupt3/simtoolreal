"""Render panel (a) of fig 2: 1x4 row of training-curve mini-plots.

Tasks (left to right):
    Peg-In-Hole, Asm-Pillar, Asm-Beam, Screw-Leg

Each subplot shows:
    Play2Win (blue, solid)
    Scratch (dense reward) (orange, solid at 0%)
    Scratch (task reward) (red, solid at 2% with a small offset so both show in legend)

The Play2Win curves are normalized by (1 - terminal done_fall) — i.e. success
conditional on feasible initialization. A shared 3-entry legend sits below the
row.

Reads:
    outputs/fig2_panel_bcd/panel_a_curves.json

Writes:
    outputs/fig2_panel_bcd/panel_a_draft.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "fig2_panel_bcd"

COLORS = {
    "Play2Win": "#2C7BB6",
    "Scratch_dense": "#E08214",
    "Scratch_task": "#D6604D",
}
TASKS = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]


def main():
    panel_a = json.load(open(OUT / "panel_a_curves.json"))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    xticks = [0, 6, 12, 18, 24]
    yticks = [0, 25, 50, 75, 100]
    lw_main, lw_base = 1.5, 1.3

    handles = None
    for i, task in enumerate(TASKS):
        ax = axes[i]
        pts = panel_a[task]["pts"]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        h0, = ax.plot(xs, ys, color=COLORS["Play2Win"], linewidth=lw_main, label="Play2Win")
        if xs[-1] < 24.0:
            ax.plot([xs[-1], 24.0], [ys[-1], ys[-1]], color=COLORS["Play2Win"], linewidth=lw_main)
        h1, = ax.plot(
            [0, 24.0], [0.0, 0.0], color=COLORS["Scratch_dense"], linewidth=lw_base,
            label="Scratch (dense reward)",
        )
        h2, = ax.plot(
            [0, 24.0], [2.0, 2.0], color=COLORS["Scratch_task"], linewidth=lw_base,
            label="Scratch (task reward)",
        )
        if handles is None:
            handles = [h0, h1, h2]
        ax.set_title(task, fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_ylim(-3, 104)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v} h" for v in xticks], fontsize=7)
        ax.set_xlabel("Training time", fontsize=8.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v}%" for v in yticks], fontsize=7)
        ax.set_ylabel("Success rate", fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", bbox_to_anchor=(0.5, 0.04),
        ncol=3, frameon=False, fontsize=8,
        handlelength=1.6, handletextpad=0.5, columnspacing=2.0,
    )

    plt.tight_layout(w_pad=1.0, rect=[0, 0.10, 1, 1])
    plt.savefig(OUT / "panel_a_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_a_draft.png'}")


if __name__ == "__main__":
    main()
