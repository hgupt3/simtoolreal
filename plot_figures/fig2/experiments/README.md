# Fig 2 — Panel (a) experiments

4 teacher (Play2Win finetune) runs, one per task. Each mirrors the latest
`teachers_final/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_dr_*`
recipe:

* preInsertAndFinal goals, SAPG, finetune from the shared pretrained
  checkpoint at `pretrained_policy/model.pth`.
* `random_goal_fraction = 0.0`.
* Full domain randomization ON (obs/action delay + object-state delay +
  object-state XYZ/rotation noise + object-scale noise).
* Perturbation forces/torques OFF during training.

## Run inventory

| # | Task | URDF problem key | Submission script |
|---|------|------------------|--------------------|
| 1 | L-peg @ 0.5 mm clearance | `Lpeg_matchedmass.tol0p5mm` | `panel_a_lpeg_tol0p5mm_finetune_rgf0_dr.sub` |
| 2 | Fabrica beam_3x part 0 | `fabrica.beam_3x.part_0_matchedmass` | `panel_a_beam_3x_part_0_finetune_rgf0_dr.sub` |
| 3 | Fabrica beam_3x part 2 | `fabrica.beam_3x.part_2_matchedmass` | `panel_a_beam_3x_part_2_finetune_rgf0_dr.sub` |
| 4 | FurnitureBench one_leg (1×) | `furniture_bench.one_leg_matchedmass_sdf_hybrid` | `panel_a_furniture_bench_one_leg_finetune_rgf0_dr.sub` |

## Outputs

Each run writes to:
```
train_dir/fig2/panel_a_teachers/<EXPERIMENT_TAG>_<DATETIME>/
    .hydra/         # config snapshot
    0_<NAME>/       # rl_games artifacts (nn/, summaries/, best/, last/)
    interactive_viewer/
    slurm.log, slurm.err, gpu_usage.log, ram_usage.log
```

W&B project / group: `fig2 / panel_a_teachers`.

## Submitting

```bash
# Run 1: L-peg
sbatch plot_figures/fig2/experiments/panel_a_lpeg_tol0p5mm_finetune_rgf0_dr.sub

# Run 2-4: Beam parts + FurnitureBench
sbatch plot_figures/fig2/experiments/panel_a_beam_3x_part_0_finetune_rgf0_dr.sub
sbatch plot_figures/fig2/experiments/panel_a_beam_3x_part_2_finetune_rgf0_dr.sub
sbatch plot_figures/fig2/experiments/panel_a_furniture_bench_one_leg_finetune_rgf0_dr.sub
```

Override checkpoint:
```bash
CHECKPOINT=/path/to/other.pth sbatch panel_a_lpeg_tol0p5mm_finetune_rgf0_dr.sub
```

## Notes

* All 4 runs use the same `Isaacsimenvs-PegInHole-Direct-v0` task — only
  `env.peg_in_hole.problem` differs. The env auto-loads the inserter +
  receptive URDF via `PROBLEM_REGISTRY`.
* For run #1 (L-peg), the canonical teacher already exists at
  `train_dir/teachers_final/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40/`.
  Re-launch this only if you want a fresh 48 h curve for the panel-(a)
  plot starting from the matched-mass URDF state.
* For run #4 (FurnitureBench), the problem key references the
  `_matchedmass_sdf_hybrid` variant so the threaded tip uses SDF
  collision and the body uses CoACD hulls. The fixture URDF carries the
  3 inactive-corner cylinder approximations.
* Multi-stage goal_mode and Scratch baselines are deferred — those need
  `multiStageTrajectory` to be implemented and from-scratch launchers
  added (see TODO in the panel-(a) plan).
