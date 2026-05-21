"""Render panel (b) of fig 3: pretraining-objective ablation.

1x4 row of training-curve mini-plots, one per downstream task. Each subplot
shows 4 curves:

    Play2Win        — REAL curve from fig 2 panel (a)
    TranslationOnly — synthetic dummy
    RotationOnly    — synthetic dummy
    SingleGoal      — synthetic dummy

The Play2Win curve IS the full pretrain. The other 3 are synthesized as
exponential-saturation toward lower asymptotes until real ablation pretrain
runs land.

Reads (Play2Win only):
    outputs/fig2_panel_bcd/panel_a_curves.json

Writes:
    plot_figures/fig3/outputs/panel_b_draft.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG2_DATA = REPO / "outputs" / "fig2_panel_bcd" / "panel_a_curves.json"
OUT = REPO / "plot_figures" / "fig3" / "outputs"

# (display name, color)
VARIANTS = [
    ("Play2Win (random pose goals)", "#2C7BB6"),   # real data
    ("Translation goals only",       "#E08214"),
    ("Rotation goals only",          "#009E73"),
    ("Single goal",                  "#4D4D4D"),
]
TASKS = ["Peg-In-Hole", "Asm-Pillar", "Asm-Beam", "Screw-Leg"]

# Per-(variant, task) outcome. Bimodal regime: fine-tuning either converges
# to ~100% or stays dead at 0%. "succ_<N>" = succeeds with rise-time N hours.
# Hypothesis:
#   Translation only: solves translation-dominated tasks (Peg, Pillar),
#                     dead on rotation-heavy ones (Beam, Screw-Leg).
#   Rotation only:    opposite — solves rotation-heavy, dead on the others.
#   Single goal:      no preInsertAndFinal scaffolding → dead on all.
OUTCOMES = {
    "Play2Win (random pose goals)": {t: "real" for t in TASKS},
    "Translation goals only":       {"Peg-In-Hole": "succ_12", "Asm-Pillar": "succ_15",
                                     "Asm-Beam": "dead",       "Screw-Leg": "dead"},
    "Rotation goals only":          {"Peg-In-Hole": "dead",    "Asm-Pillar": "dead",
                                     "Asm-Beam": "succ_14",    "Screw-Leg": "succ_18"},
    "Single goal":                  {t: "dead" for t in TASKS},
}
RNG = np.random.default_rng(1)


def synth_curve(xs, regime, cap=1.0, noise_succ=0.010, noise_dead=0.010):
    """Synthesise a dummy success curve in [0, cap]; cap = Play2Win peak."""
    xs = np.asarray(xs)
    if regime == "dead":
        base = np.zeros_like(xs)
        jitter = np.abs(RNG.normal(0, noise_dead, size=xs.shape))
        return np.clip(base + jitter, 0, cap)
    assert regime.startswith("succ_"), regime
    t_rise = float(regime.split("_", 1)[1])
    rate = -np.log(0.20) / t_rise
    base = cap * (1.0 - np.exp(-rate * xs))
    jitter = RNG.normal(0, noise_succ, size=xs.shape)
    return np.clip(base + jitter, 0, cap)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel_a_fig2 = json.load(open(FIG2_DATA))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    xticks = [0, 6, 12, 18, 24]
    yticks = [0, 25, 50, 75, 100]
    handles = None

    for i, task in enumerate(TASKS):
        ax = axes[i]
        p2w_pts = panel_a_fig2[task]["pts"]
        xs_real = np.array([p[0] for p in p2w_pts])
        ys_real = np.array([p[1] for p in p2w_pts])
        p2w_asymptote = float(ys_real.max())

        ablation_handles = []
        for label, color in VARIANTS:
            regime = OUTCOMES[label][task]
            if regime == "real":
                xs, ys = xs_real, ys_real
                if xs[-1] < 24.0:
                    xs = np.append(xs, 24.0)
                    ys = np.append(ys, ys[-1])
            else:
                xs = np.linspace(0, 24, 200)
                ys = synth_curve(xs, regime=regime, cap=p2w_asymptote)
            h, = ax.plot(xs, 100 * ys, color=color, linewidth=1.3, label=label, alpha=0.95)
            ablation_handles.append(h)

        if handles is None:
            handles = ablation_handles

        ax.set_title(task, fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_ylim(-3, 104)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v} h" for v in xticks], fontsize=7)
        ax.set_xlabel("Training time", fontsize=8.5)
        ax.set_yticks(yticks)
        if i == 0:
            ax.set_yticklabels([f"{v}%" for v in yticks], fontsize=7)
            ax.set_ylabel("Success rate", fontsize=8.5)
        else:
            ax.tick_params(axis='y', labelleft=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", bbox_to_anchor=(0.5, 0.04),
        ncol=4, frameon=False, fontsize=8,
        handlelength=1.6, handletextpad=0.5, columnspacing=1.8,
    )
    plt.tight_layout(w_pad=0.6, rect=[0, 0.10, 1, 1])
    plt.savefig(OUT / "panel_b_draft.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'panel_b_draft.png'}")


if __name__ == "__main__":
    main()
