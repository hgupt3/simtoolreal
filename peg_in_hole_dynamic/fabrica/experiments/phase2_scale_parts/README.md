# Phase 2: Scale to All 5 Beam Parts

**Branch**: `2026_03_18_phase2_scale_parts`
**WandB Group**: `fabrica_phase2_scale_parts`

## Goal

Train insertion for all 5 beam parts independently, using the best tolerance strategy from Phase 1.

## Per-Part Details

| Part | Geometry | Rotation | Final z | Difficulty | `.sub` file |
|------|----------|----------|---------|------------|-------------|
| 6 | 152×18×13mm (largest) | None (identity quat) | 0.537 | Easy | `beam_6.sub` |
| 2 | 76×12×12mm | 90° single-axis | 0.568 | **Baseline (works)** | (existing baseline) |
| 3 | 76×12×12mm | 90° single-axis | ~0.568 | Easy (mirrors part 2) | `beam_3.sub` |
| 0 | 76×15×13mm | 120° 3-axis | 0.602 | Hard | `beam_0.sub` |
| 1 | 76×15×13mm | Similar to 0 | ~0.602 | Hard | `beam_1.sub` |

## How to run

```bash
sbatch fabrica/experiments/phase2_scale_parts/beam_0.sub
sbatch fabrica/experiments/phase2_scale_parts/beam_1.sub
sbatch fabrica/experiments/phase2_scale_parts/beam_3.sub
sbatch fabrica/experiments/phase2_scale_parts/beam_6.sub
```

## Configuration

- Uses FGT curriculum (0.01 → 0.002m) — update if Phase 1 results suggest otherwise
- DR off (same as baseline)
- 24K envs, pretrained checkpoint

## Risk

Parts 0/1 have complex 3-axis rotations — if they fail, try:
1. More trajectory waypoints
2. Looser FGT target (0.005m first)

## Results

| Part | Goals Reached | Frames to 12/12 | Final FGT | Notes |
|------|--------------|-----------------|-----------|-------|
| 6 | | | | |
| 2 | 12/12 | — | 0.002m | Baseline (already done) |
| 3 | | | | |
| 0 | | | | |
| 1 | | | | |

## Conclusion

_To be filled after experiments complete._
