"""Render panel (c) of fig 2: force-perturbation robustness sweep.

Methods (each is one curve with hollow circle markers at the 11 force scales
{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50} N):
    Play2Win (blue)
    Scratch (dense reward) (orange)

Each data point is a single forceScale eval with 128 parallel envs (no
error bars per point).

Reads (per method, per force scale):
    outputs/fig2_panel_bcd/panel_c_{play2win,scratch_multistage}_fs{N}.json

Writes:
    plot_figures/fig2/outputs/panel_c_draft.png
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "outputs" / "fig2_panel_bcd"
OUT = REPO / "plot_figures" / "fig2" / "outputs"

LABELS = {
    "play2win":           ("Play2Win",                "#2C7BB6"),
    "scratch_multistage": ("Scratch (dense reward)",  "#E08214"),
}


def load_curve(method):
    pts = []
    for p in sorted(
        DATA_DIR.glob(f"panel_c_{method}_fs*.json"),
        key=lambda q: int(re.search(r"fs(\d+)", q.stem).group(1)),
    ):
        d = json.load(open(p))
        fs = int(re.search(r"fs(\d+)", p.stem).group(1))
        sr = next(iter(d["results"].values()))["success_rate"] * 100
        pts.append((fs, sr))
    return pts


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    for key, (label, c) in LABELS.items():
        pts = load_curve(key)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys, color=c, linewidth=1.7, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.3, label=label,
        )

    ax.set_xlabel("Random force perturbation", fontsize=9)
    ax.set_ylabel("Success rate", fontsize=9)
    ax.set_xlim(-1, 51)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=8)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels([f"{v} N" for v in (0, 10, 20, 30, 40, 50)], fontsize=8)
    ax.legend(
        loc="center right", bbox_to_anchor=(0.99, 0.55),
        frameon=False, fontsize=8,
        handlelength=1.4, handletextpad=0.5, borderaxespad=0.2,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "panel_c_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_c_draft.png'}")


if __name__ == "__main__":
    main()
