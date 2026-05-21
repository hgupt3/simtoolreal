"""Compose the full fig 2 in a single 7.2 x 5.6 inch canvas.

Layout (3 rows):
    Row 1: panel (a) — 4 mini training-time curves (Peg-In-Hole, Asm-Pillar,
           Asm-Beam, Screw-Leg). Shared 3-entry legend below.
    Row 2: panel (b) (left, 1/4 width)  |  panel (d) Play2Win strip (right, 3/4 width)
    Row 3: panel (c) (left, 1/4 width)  |  panel (d) Scratch (dense) strip (right, 3/4 width)

Panel (d) is rendered as flat rectangular placeholders (4 frames per method,
tinted blue/orange) until real episode snapshots are dropped in.

Reads:
    outputs/fig2_panel_bcd/panel_a_curves.json
    outputs/fig2_panel_bcd/panel_b_curves.json
    outputs/fig2_panel_bcd/panel_c_{play2win,scratch_multistage}_fs*.json

Writes:
    outputs/fig2_panel_bcd/fig2_full.png
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "fig2_panel_bcd"

COLORS = {
    "Play2Win": "#2C7BB6",
    "Scratch_dense": "#E08214",
    "Scratch_task": "#D6604D",
}
TASKS_A = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]
NFRAMES = 4

LABEL_PB = {
    "Play2Win":           ("Play2Win",                "#2C7BB6"),
    "Scratch_multistage": ("Scratch (dense reward)",  "#E08214"),
    "Scratch_task":       ("Scratch (task reward)",   "#D6604D"),
}
LABEL_PC = {
    "play2win":           ("Play2Win",                "#2C7BB6"),
    "scratch_multistage": ("Scratch (dense reward)",  "#E08214"),
}


def load_panel_c(method):
    pts = []
    for p in sorted(
        OUT.glob(f"panel_c_{method}_fs*.json"),
        key=lambda q: int(re.search(r"fs(\d+)", q.stem).group(1)),
    ):
        d = json.load(open(p))
        fs = int(re.search(r"fs(\d+)", p.stem).group(1))
        sr = next(iter(d["results"].values()))["success_rate"] * 100
        pts.append((fs, sr))
    return pts


def main():
    panel_a = json.load(open(OUT / "panel_a_curves.json"))
    panel_b = json.load(open(OUT / "panel_b_curves.json"))

    fig = plt.figure(figsize=(7.2, 5.6))
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.9, 1.4, 1.4],
        hspace=0.65, left=0.07, right=0.985, top=0.955, bottom=0.07,
    )

    # ============ Row 1: Panel (a) ============
    row_a = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.18)
    xticks_a = [0, 6, 12, 18, 24]
    yticks_a = [0, 25, 50, 75, 100]
    lw_a_main, lw_a_base = 1.5, 1.3
    handles_a = None
    for i, task in enumerate(TASKS_A):
        ax = fig.add_subplot(row_a[0, i])
        pts = panel_a[task]["pts"]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        h0, = ax.plot(xs, ys, color=COLORS["Play2Win"], linewidth=lw_a_main, label="Play2Win")
        if xs[-1] < 24.0:
            ax.plot([xs[-1], 24.0], [ys[-1], ys[-1]], color=COLORS["Play2Win"], linewidth=lw_a_main)
        h1, = ax.plot(
            [0, 24.0], [0.0, 0.0], color=COLORS["Scratch_dense"], linewidth=lw_a_base,
            label="Scratch (dense reward)",
        )
        h2, = ax.plot(
            [0, 24.0], [2.0, 2.0], color=COLORS["Scratch_task"], linewidth=lw_a_base,
            label="Scratch (task reward)",
        )
        if handles_a is None:
            handles_a = [h0, h1, h2]
        ax.set_title(task, fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_ylim(-3, 104)
        ax.set_xticks(xticks_a)
        ax.set_xticklabels([f"{v} h" for v in xticks_a], fontsize=7)
        ax.set_xlabel("Training time", fontsize=8.5)
        ax.set_yticks(yticks_a)
        if i == 0:
            ax.set_yticklabels([f"{v}%" for v in yticks_a], fontsize=7)
            ax.set_ylabel("Success rate", fontsize=8.5)
        else:
            ax.tick_params(axis='y', labelleft=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.legend(
        handles_a, [h.get_label() for h in handles_a],
        loc="lower center", bbox_to_anchor=(0.5, 0.640),
        ncol=3, frameon=False, fontsize=8,
        handlelength=1.6, handletextpad=0.5, columnspacing=2.0,
    )

    # ============ Rows 2 / 3: (b)/(c) on left, panel (d) on right (3/4 width) ============
    row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1, 3], wspace=0.40)
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[1, 3], wspace=0.40)

    # Panel (b)
    X_MAX = 140.0
    target_ratio = panel_b["Scratch_multistage"][-1][1]
    x_cross = next(t for t, s in panel_b["Play2Win"] if s >= target_ratio)
    speedup = panel_b["Scratch_multistage"][-1][0] / x_cross
    y_cross = target_ratio * 100

    ax_b = fig.add_subplot(row2[0, 0])
    for key, (label, c) in LABEL_PB.items():
        pts = panel_b[key]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        ax_b.plot(xs, ys, color=c, linewidth=0.9)
        if key == "Play2Win" and xs[-1] < X_MAX:
            ax_b.plot([xs[-1], X_MAX], [ys[-1], ys[-1]], color=c, linewidth=0.9)
    ax_b.plot([x_cross, x_cross], [-2, y_cross], color="#555555", linestyle="--", linewidth=0.8)
    ax_b.text(
        0.55, 0.13,
        f"{speedup:.0f}× faster\n({x_cross:.1f}h)",
        fontsize=7, color="#333333", ha="center", va="center", multialignment="center",
        transform=ax_b.transAxes,
    )
    ax_b.set_xlabel("Training time", fontsize=8)
    ax_b.set_ylabel("Success rate", fontsize=8)
    ax_b.set_xlim(0, X_MAX)
    ax_b.set_ylim(-2, 104)
    ax_b.set_yticks([0, 50, 100])
    ax_b.set_yticklabels([f"{v}%" for v in (0, 50, 100)], fontsize=7)
    ax_b.set_xticks([0, 70, 140])
    ax_b.set_xticklabels([f"{v} h" for v in (0, 70, 140)], fontsize=7)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)

    # Panel (c)
    ax_c = fig.add_subplot(row3[0, 0])
    for key, (label, c) in LABEL_PC.items():
        pts = load_panel_c(key)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_c.plot(
            xs, ys, color=c, linewidth=1.3, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.1,
        )
    ax_c.set_xlabel("Force perturbation", fontsize=8)
    ax_c.set_ylabel("Success rate", fontsize=8)
    ax_c.set_xlim(-1, 51)
    ax_c.set_ylim(-2, 104)
    ax_c.set_yticks([0, 50, 100])
    ax_c.set_yticklabels([f"{v}%" for v in (0, 50, 100)], fontsize=7)
    ax_c.set_xticks([0, 25, 50])
    ax_c.set_xticklabels([f"{v} N" for v in (0, 25, 50)], fontsize=7)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)

    # Panel (d) placeholders
    d_top = gridspec.GridSpecFromSubplotSpec(1, NFRAMES, subplot_spec=row2[0, 1], wspace=0.06)
    d_bot = gridspec.GridSpecFromSubplotSpec(1, NFRAMES, subplot_spec=row3[0, 1], wspace=0.06)
    PLAY_TINT = (0.86, 0.93, 0.99)
    SCR_TINT = (0.99, 0.93, 0.84)

    for k in range(NFRAMES):
        ax = fig.add_subplot(d_top[0, k])
        ax.set_facecolor(PLAY_TINT)
        ax.add_patch(Rectangle(
            (0.0, 0.0), 1.0, 1.0, facecolor=PLAY_TINT, edgecolor="#888888",
            linewidth=0.7, transform=ax.transAxes, zorder=-1,
        ))
        ax.text(
            0.5, 0.5, f"Play2Win\nframe {k+1}",
            ha="center", va="center", fontsize=7.5, color="#444444", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(False)
        if k == 0:
            ax.set_ylabel("Play2Win", fontsize=8.5, color=COLORS["Play2Win"], labelpad=4, rotation=90)

    for k in range(NFRAMES):
        ax = fig.add_subplot(d_bot[0, k])
        ax.set_facecolor(SCR_TINT)
        ax.add_patch(Rectangle(
            (0.0, 0.0), 1.0, 1.0, facecolor=SCR_TINT, edgecolor="#888888",
            linewidth=0.7, transform=ax.transAxes, zorder=-1,
        ))
        ax.text(
            0.5, 0.5, f"Scratch (dense)\nframe {k+1}",
            ha="center", va="center", fontsize=7.5, color="#444444", transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(False)
        if k == 0:
            ax.set_ylabel(
                "Scratch (dense)", fontsize=8.5, color=COLORS["Scratch_dense"], labelpad=4, rotation=90,
            )

    def panel_letter(x, y, letter):
        fig.text(x, y, f"({letter})", fontsize=11, fontweight="bold", ha="left", va="top")

    panel_letter(0.010, 0.985, "a")
    panel_letter(0.010, 0.580, "b")
    panel_letter(0.010, 0.290, "c")
    panel_letter(0.260, 0.580, "d")

    plt.savefig(OUT / "fig2_full.png", dpi=240, facecolor="white")
    plt.close()
    print(f"wrote {OUT / 'fig2_full.png'}")


if __name__ == "__main__":
    main()
