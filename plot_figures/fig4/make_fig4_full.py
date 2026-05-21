"""Fig 4 combined: panel (a) success rate + panel (b) throughput, side by side.

Total figure: 8.0 × 2.6 inches.
- Panel (a) gets 3.5 inches wide (sim/real success across 4 tasks).
- Panel (b) gets 4.5 inches wide (Play2Win / PJaw teleop / Human throughput).

Stub numbers are inline in this script — swap for real numbers when ready.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "plot_figures" / "fig4" / "outputs"

TASKS = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]

# Panel (a): sim vs real success (%)
SIM_SUCCESS = np.array([99.0, 99.0, 98.0, 92.0])
REAL_SUCCESS = np.array([88.0, 90.0, 85.0, 75.0])
COLOR_SIM = "#92C5DE"
COLOR_REAL = "#2C7BB6"

# Panel (b): real-world throughput (successes / minute)
THROUGHPUT = {
    "Play2Win":    np.array([11.0, 5.5, 4.0, 2.7]),
    "PJaw teleop": np.array([ 2.0, 1.0, 0.5, 0.4]),
    "Human demo":  np.array([13.5, 6.4, 5.0, 3.1]),
}
COLORS_B = {
    "Play2Win":    "#2C7BB6",
    "PJaw teleop": "#E08214",
    "Human demo":  "#4D4D4D",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 2.0))
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[3.5, 4.5],
        # Tight margins so the 2.0" total height leaves max plotting room.
        # Plotting area height ≈ 2.0 * (top - bottom) = 2.0 * (0.96 - 0.28) = 1.36".
        wspace=0.20, left=0.085, right=0.99, top=0.96, bottom=0.28,
    )

    # ===== Panel (a): success rate =====
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(TASKS))
    bar_w = 0.35
    h_sim = ax_a.bar(x - bar_w / 2, SIM_SUCCESS, bar_w, color=COLOR_SIM, label="Sim")
    h_real = ax_a.bar(x + bar_w / 2, REAL_SUCCESS, bar_w, color=COLOR_REAL, label="Real")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(TASKS, fontsize=6.5)
    ax_a.set_ylabel("Success rate", fontsize=8.5)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=7)
    ax_a.set_ylim(0, 105)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    ax_a.legend(
        [h_sim, h_real], ["Sim", "Real"],
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, frameon=False, fontsize=7.5,
        handlelength=1.6, handletextpad=0.5, columnspacing=2.0,
    )

    # ===== Panel (b): throughput =====
    ax_b = fig.add_subplot(gs[0, 1])
    methods = list(THROUGHPUT.keys())
    bar_w_b = 0.25
    handles_b = []
    for i, m in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_w_b
        h = ax_b.bar(x + offset, THROUGHPUT[m], bar_w_b, color=COLORS_B[m], label=m)
        handles_b.append(h)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(TASKS, fontsize=6.5)
    ax_b.set_ylabel("Successes / min", fontsize=8.5)
    ax_b.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.legend(
        handles_b, methods,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, frameon=False, fontsize=7.5,
        handlelength=1.6, handletextpad=0.5, columnspacing=2.0,
    )

    out_path = OUT / "fig4_full.png"
    plt.savefig(out_path, dpi=240, facecolor="white")
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
