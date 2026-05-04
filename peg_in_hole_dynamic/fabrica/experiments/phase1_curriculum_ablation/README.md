# Phase 1: FGT Curriculum Ablation

**Branch**: `2026_03_18_phase1_curriculum_ablation`
**WandB Group**: `fabrica_phase1_curriculum_ablation`

## Question

Is the FGT (Final Goal Tolerance) curriculum necessary, or can we directly set a tight final goal tolerance?

## Background

The current training pipeline uses a curriculum that starts at 0.01m tolerance and decrements by 0.001m each time the agent achieves consecutive successes, down to a target of 0.002m. This phase tests whether we can skip the curriculum and train directly at the final tolerance.

## Experiment Matrix (all on beam part 2)

| Experiment | FGT Curriculum | Final Goal Tolerance | Tag | `.sub` file |
|-----------|---------------|---------------------|-----|-------------|
| Control | ON (0.01 -> 0.002) | 0.002m | `ablation_curriculum` | `baseline.sub` (in `train_scripts/`) |
| 1a | OFF | 0.002m direct | `ablation_direct_0.002` | `no_curriculum_0.002.sub` |
| 1b | OFF | 0.005m direct | `ablation_direct_0.005` | `no_curriculum_0.005.sub` |
| 1c | OFF | 0.01m direct | `ablation_direct_0.01` | `no_curriculum_0.01.sub` |

## How to run

```bash
# Submit all 4 experiments (control uses existing baseline.sub with updated WANDB_GROUP)
sbatch fabrica/experiments/phase1_curriculum_ablation/no_curriculum_0.002.sub
sbatch fabrica/experiments/phase1_curriculum_ablation/no_curriculum_0.005.sub
sbatch fabrica/experiments/phase1_curriculum_ablation/no_curriculum_0.01.sub
# Control: update baseline.sub WANDB_GROUP and EXPERIMENT_TAG, then submit
```

## Expected Outcomes

- **1c (0.01m direct)**: Should work — this is the starting tolerance of the curriculum anyway (sanity check)
- **1b (0.005m direct)**: Interesting boundary — may or may not converge
- **1a (0.002m direct)**: Likely fails or converges much slower, validating the curriculum approach

## Metrics to Compare

- `consecutive_successes` vs training frames (4-curve WandB plot)
- `final_goal_tolerance` over time for the curriculum run
- Wall-clock time to reach 12 consecutive successes

## Results

| Experiment | Goals Reached | Frames to 12/12 | Final FGT | Notes |
|-----------|--------------|-----------------|-----------|-------|
| Control (curriculum) | | | | |
| 1a (direct 0.002) | | | | |
| 1b (direct 0.005) | | | | |
| 1c (direct 0.01) | | | | |

## Conclusion

_To be filled after experiments complete._
