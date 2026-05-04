# Collision Method Comparison for Beam Assembly

**Date:** 2025-03-14 advisor meeting
**Assembly:** beam (parts 2 and 6)

## Overview

Compared 3 collision mesh methods in IsaacGym for thin beam assembly parts:
- **VHACD** — IsaacGym's built-in convex decomposition
- **CoACD** — Approximate convex decomposition (default and tuned params)
- **SDF** — Signed distance field (exact mesh collision via `useSDF` config)

All renders use `cam_props.use_collision_geometry = True` so we see the actual collision shapes, not visual meshes.

## CoACD Parameter Tuning

### Parameter sets

| Parameter | Default (worst) | Tuned v1 (OK) | Tuned v2 (best) |
|---|---|---|---|
| threshold | 0.05 | 0.03 | 0.03 |
| preprocess_resolution | 30 | 50 | 100 |
| resolution | 2000 | 3000 | 8000 |
| mcts_nodes | 20 | 25 | 25 |
| mcts_iterations | 150 | 250 | 250 |
| mcts_max_depth | 3 | 4 | 4 |

- **Default**: Too coarse for thin parts. Part 6 only got 6 hulls.
- **Tuned v1**: Good general improvement. Part 6 went to 15 hulls. Used for all parts. However, part 3 insertion had 3.9mm collision delta due to part 6's hulls filling narrow channels.
- **Tuned v2**: Applied to part 6 only. Higher `preprocess_resolution` (100) preserves narrow channels in the manifold repair step. Higher `resolution` (8000) gives tighter Hausdorff fit per hull. Part 6 went to 12 hulls — fewer but better-shaped, no longer blocking part 3's insertion path.

### Hull counts

| Part | Default | Tuned v1 | Tuned v2 |
|---|---|---|---|
| Part 2 | 17 | 18 | — (unchanged, uses v1) |
| Part 6 | 6 | 15 | 12 |

### Insertion test results (final timestep delta, all parts)

Tested with `debug_insertion_training.py` using full trajectory (all 12 waypoints).

**Tuned v1 (all parts at v1 params):**

| Part | Final Delta |
|---|---|
| Part 0 | 1.2mm |
| Part 1 | 1.0mm |
| Part 2 | 1.3mm |
| Part 3 | 3.9mm |
| Part 6 | 0.4mm |

**Tuned v2 (part 6 at v2 params, rest at v1):**

| Part | Final Delta | Change |
|---|---|---|
| Part 0 | 1.2mm | same |
| Part 1 | 1.0mm | same |
| Part 2 | 1.0mm | improved |
| Part 3 | 0.6mm | fixed (was 3.9mm) |
| Part 6 | 0.4mm | same |

All parts now under 1.2mm. The ~2mm constant baseline delta visible throughout trajectories is a GPU pipeline physics offset (present even in free air), not real collision.

### Experiment: v2 params on all parts (branch `2026_03_18_FineGrainedInsertions`)

Tested applying v2 params to all 5 parts (not just part 6). Hull counts changed: Part 2: 18→14, Part 3: 17→19, others unchanged.

| Part | v1 all | v2 part6-only | v2 all parts |
|---|---|---|---|
| Part 0 | 1.2mm | 1.2mm | 1.3mm |
| Part 1 | 1.0mm | 1.0mm | 1.4mm |
| Part 2 | 1.3mm | 1.0mm | 0.4mm |
| Part 3 | 3.9mm | 0.6mm | 1.9mm |
| Part 6 | 0.4mm | 0.4mm | 0.4mm |

**Decision:** Keep v2 on part 6 only (max 1.2mm across all parts). Applying v2 to all parts regressed part 3 from 0.6mm to 1.9mm due to CoACD non-determinism producing a different decomposition for part 3 (19 vs 17 hulls). Per-part tuning is more reliable than blanket parameter changes.

## Quality Ranking

**SDF > CoACD (tuned) > CoACD (default) > VHACD**

- **SDF**: Most faithful to original mesh. No decomposition artifacts. Best for debugging/validation.
- **CoACD (tuned)**: Close to SDF quality. Good hull coverage on thin parts. Suitable for GPU pipeline.
- **CoACD (default)**: Acceptable for chunky parts but too coarse for thin geometry (e.g., Part 6 with only 6 hulls).
- **VHACD**: Coarsest approximation. Adequate for simple shapes but poor on thin/detailed parts.

## Comparison Images

![Part 2](../debug_output/comparison/collision_comparison_part_2.png)

![Part 6](../debug_output/comparison/collision_comparison_part_6.png)

Columns: VHACD, CoACD (default), CoACD (tuned), SDF

## Debug Pipeline

1. `fabrica/run_coacd.py` — generate CoACD decomposition with configurable params
2. `fabrica/debug_coacd.py` — IsaacGym headless drop test rendering collision geometry
   - `--coacd-dir` to override decomposition directory
   - `--output-dir` to control output location
   - `--empty-table` for clean background
3. `fabrica/make_comparison.py` — assemble grid image from frame captures

## Next Steps

- Test collision methods with actual assembly tasks (insert part into fixture)
- Try CoACD `extrude=True` for inflated collision margins on thin parts
- Evaluate performance impact of SDF vs CoACD in multi-part assembly scenes
- Consider per-part parameter tuning for other assemblies beyond beam
