"""Fig 4 panel (a): sim vs real success rate per task (half-column bar chart).

Stub numbers. Drop real numbers into SIM_SUCCESS / REAL_SUCCESS below.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "plot_figures" / "fig4" / "outputs"

TASKS = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]

# Stub success rates (%). Swap in real numbers later.
SIM_SUCCESS = np.array([99.0, 99.0, 98.0, 92.0])
REAL_SUCCESS = np.array([88.0, 90.0, 85.0, 75.0])

COLOR_SIM = "#92C5DE"
COLOR_REAL = "#2C7BB6"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.6))

    x = np.arange(len(TASKS))
    bar_w = 0.35
    h_sim, = (ax.bar(x - bar_w / 2, SIM_SUCCESS, bar_w, color=COLOR_SIM, label="Sim"),)
    h_real, = (ax.bar(x + bar_w / 2, REAL_SUCCESS, bar_w, color=COLOR_REAL, label="Real"),)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_ylabel("Success rate", fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=8)
    ax.set_ylim(0, 105)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Match fig 2 / fig 3 panel-(a) shared-legend style: bottom-centered, no
    # frame, fontsize=8, generous handle spacing.
    fig.legend([h_sim, h_real], ["Sim", "Real"],
               loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=2, frameon=False, fontsize=8,
               handlelength=1.6, handletextpad=0.5, columnspacing=2.0)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(OUT / "panel_a_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_a_draft.png'}")


if __name__ == "__main__":
    main()
