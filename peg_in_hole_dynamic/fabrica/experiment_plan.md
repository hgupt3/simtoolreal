# Fabrica Assembly Policy Evaluation Plan

## Date: 2026-03-26

## Objective

Systematically evaluate 3 RL policies across all 38 Fabrica insertion tasks to understand
how training data composition affects insertion performance and generalization.

## Hypotheses

1. **Pretrained policy alone can't do most insertions** — the pretrained policy was trained
   on generic manipulation, not assembly-specific insertion tasks.
2. **Beam assembly policy perfectly does beam assembly** — fine-tuned on all 5 beam parts,
   it should achieve near-100% success on beam tasks.
3. **Beam assembly policy generalizes to other assemblies** better than the pretrained policy,
   because it has learned general insertion skills during beam fine-tuning.
4. **All-assemblies policy is worse than beam policy on beam** (less specialization) **but
   better on other assemblies** (broader training distribution).

## Policies Under Evaluation

| Policy | Description | Config | Checkpoint |
|--------|-------------|--------|------------|
| `pretrained` | Base manipulation policy | `pretrained_policy/config.yaml` | `pretrained_policy/model.pth` |
| `beam_policy` | Fine-tuned on 5 beam parts | `train_dir/.../fabrica_phase4_unified/all_beam_parts_.../config.yaml` | `last_..._ep_243000_rew_13263.113.pth` |
| `all_assemblies_policy` | Fine-tuned on all 6 assemblies (2-GPU) | `train_dir/.../fabrica_phase5_all_assemblies/all_6_assemblies_v2_2gpu_.../config.yaml` | `last_..._ep_243000_rew_12387.344.pth` |

## Tasks (38 total)

| Assembly | Parts | Count |
|----------|-------|-------|
| beam | 0, 1, 2, 3, 6 | 5 |
| car | 0, 1, 2, 3, 4, 5 | 6 |
| cooling_manifold | 0, 1, 2, 3, 4, 5, 6 | 7 |
| gamepad | 0, 1, 2, 3, 4, 5 | 6 |
| plumbers_block | 0, 1, 2, 3, 4 | 5 |
| stool_circular | 0, 1, 2, 3, 4, 5, 6, 7, 8 | 9 |

Note: duct assembly excluded (no trajectories/environments available).

## Evaluation Protocol

- **Collision method**: coacd
- **Insertions**: Each part evaluated independently from robot home position (no chaining)
- **Episodes per task**: 1 (deterministic policy, deterministic initial conditions)
- **Noise**: All randomization disabled (position, rotation, velocity, force perturbations)
- **Success tolerance**: 0.01m
- **Retract**: Enabled — after insertion, robot must retract hand from the part

## Metrics

Per task:
- **Goal %**: Percentage of subgoals reached (0–100%)
- **Steps**: Episode length in simulation steps
- **Retract OK**: Whether the robot successfully retracted after insertion
- **Full success**: Goal% = 100% AND retract OK

Aggregated:
- Per-policy average goal%, retract rate, full success count
- Per-assembly average goal% for each policy
- Per-part detail table

## How to Run

```bash
# Full evaluation (all 3 policies x 38 tasks = 114 evaluations)
python fabrica/fabrica_eval_all.py

# Subset evaluation
python fabrica/fabrica_eval_all.py --policies pretrained --assemblies beam

# Custom collision method
python fabrica/fabrica_eval_all.py --collision sdf
```

Results saved to `fabrica/eval_results_<timestamp>.json`.
