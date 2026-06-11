# MaxRL Advantage Reweighting — Experiment Guide

## What This Tests

Three PPO training runs with different advantage reweighting strategies, inspired by the MaxRL paper (arxiv 2602.02710). All code changes are already implemented in `a2c_common.py`.

- **Baseline**: Standard PPO (no reweighting) — control run
- **Value**: Weight each transition's advantage by inverse percentile of V(s) — upweights hard states
- **Return EMA**: Weight each env's advantages by inverse percentile of its running average episode return — upweights hard objects

## Running the Experiments

```bash
cd /share/portal/kk837/simtoolreal

# Submit all 3 jobs
sbatch experiments/maxrl/maxrl_baseline.sub
sbatch experiments/maxrl/maxrl_value.sub
sbatch experiments/maxrl/maxrl_return_ema.sub
```

All runs log to wandb under project `simtoolreal`, group `maxrl_ablation`. Tags: `maxrl_baseline`, `maxrl_value`, `maxrl_return_ema`.

## What to Compare in Wandb

- Overall success rate and reward curves
- Per-object success rates (do hard objects improve more with reweighting?)
- `maxrl/weight_std` — confirms reweighting is active (should be > 0 for value/return_ema)
- `maxrl/return_ema_mean` — tracks running average returns (return_ema mode)
- Training stability (KL divergence, entropy, gradient norms)

## Config Knobs

All configurable via Hydra overrides in the sub files:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `advantage_reweight_mode` | `none` | `none` / `value` / `return_ema` |
| `advantage_reweight_eps` | `0.1` | Floor on percentile. Lower = more aggressive (0.01 → 100x max weight, 0.1 → 10x) |
| `advantage_reweight_ema_decay` | `0.99` | EMA smoothing for return tracker. Higher = slower adaptation. |
