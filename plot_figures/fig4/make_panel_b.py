"""Fig 4 panel (b): real-world throughput per task (half-column bar chart).

3 bars per task: Play2Win / Parallel-jaw teleop / Human demo.
y-axis: successes per minute. Stub numbers — swap in real later.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "plot_figures" / "fig4" / "outputs"

TASKS = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]

# Stub throughput numbers (successes / minute).
THROUGHPUT = {
    "Play2Win":      np.array([11.0, 5.5, 4.0, 2.7]),
    "PJaw teleop":   np.array([ 2.0, 1.0, 0.5, 0.4]),
    "Human demo":    np.array([13.5, 6.4, 5.0, 3.1]),
}
COLORS = {
    "Play2Win":      "#2C7BB6",
    "PJaw teleop":   "#E08214",
    "Human demo":    "#4D4D4D",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    methods = list(THROUGHPUT.keys())
    x = np.arange(len(TASKS))
    bar_w = 0.25
    handles = []
    for i, m in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_w
        h = ax.bar(x + offset, THROUGHPUT[m], bar_w, color=COLORS[m], label=m)
        handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_ylabel("Successes / min", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.legend(handles, methods,
               loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=3, frameon=False, fontsize=8,
               handlelength=1.6, handletextpad=0.5, columnspacing=2.0)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(OUT / "panel_b_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_b_draft.png'}")


if __name__ == "__main__":
    main()
