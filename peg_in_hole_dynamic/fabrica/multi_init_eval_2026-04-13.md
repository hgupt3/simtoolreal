# Multi-Init-State Eval: beam_multi_init_no_dr vs beam_no_dr

## Setup

- **Assembly:** beam (5 parts: 6, 2, 0, 3, 1 in assembly order)
- **Policies:**
  - `beam_multi_init_no_dr` — trained with multi-init-state scenes (`scenes.npz`, 100 scenes, seed 0), no domain randomization
  - `beam_no_dr` — trained single-init (from `pick_place.json`), no domain randomization
- **Eval scenes:**
  - **Train split:** first **10** scenes from `scenes.npz` (same scenes seen during training)
  - **Val split:** first **5** scenes from `scenes_val.npz` (20 held-out scenes, seed 42 — disjoint from training)
- **Per policy:** 50 train evals + 25 val evals = 75 episodes
- **Collision method:** coacd
- **Total wall time:** 130 min

## Overall Summary

| Policy                  | Train goal % | Train retract % | Train n | Val goal % | Val retract % | Val n |
|:------------------------|-------------:|----------------:|--------:|-----------:|--------------:|------:|
| `beam_multi_init_no_dr` |       95.3 % |          86.0 % |      50 |     98.4 % |        84.0 % |    25 |
| `beam_no_dr`            |        7.0 % |           4.0 % |      50 |      3.9 % |         0.0 % |    25 |

## Per-Part Breakdown — Train Split (avg goal % [retract %])

| Policy                  |      part 0 |      part 1 |       part 2 |       part 3 |      part 6 |
|:------------------------|------------:|------------:|-------------:|-------------:|------------:|
| `beam_multi_init_no_dr` | 99.0 % [80] | 99.0 % [80] | 100.0 % [100] | 100.0 % [100] | 78.7 % [70] |
| `beam_no_dr`            | 11.0 % [0]  |  0.0 % [0]  |   10.0 % [10] |   14.0 % [10] |  0.0 % [0]  |

## Per-Part Breakdown — Val Split (avg goal % [retract %])

| Policy                  |       part 0 |       part 1 |       part 2 |       part 3 |      part 6 |
|:------------------------|-------------:|-------------:|-------------:|-------------:|------------:|
| `beam_multi_init_no_dr` | 100.0 % [80] | 100.0 % [60] | 100.0 % [100] | 100.0 % [100] | 91.8 % [80] |
| `beam_no_dr`            |   0.0 % [0]  |  18.3 % [0]  |   0.0 % [0]  |   0.0 % [0]  |  1.2 % [0]  |

## Key Takeaways

- **Multi-init training works as intended.** With diverse start poses, `beam_multi_init_no_dr` hits 95–98 % goal completion and ~85 % retract success across train and val. The single-init policy collapses to near-zero on both splits, as expected — it was never exposed to these start poses.
- **No overfitting to training scenes.** Val performance (98.4 %) matches or exceeds train (95.3 %). Multi-init coverage appears to be generalizing rather than memorizing.
- **Part 6 is the hardest case for multi-init.** 78.7 % goals on train, 91.8 % on val — consistent with it being the first part in the assembly order (no previously-assembled parts in the fixture to physically constrain the motion). Everything else scores ≥ 99 % on both splits.
- **Retract rates lag goal rates** by ~10–15 pp across parts — worth flagging if retract success is a hard requirement.

## Artifacts

- Raw JSON: `fabrica/eval_outputs/multi_init_eval_results_2026-04-13_22-46-17.json`
- Eval scripts: `fabrica/fabrica_multi_init_eval.py` (interactive viser), `fabrica/fabrica_multi_init_eval_all.py` (batch)
- Val scenes: `assets/urdf/fabrica/beam/scenes_val.npz` (20 scenes, seed 42)
