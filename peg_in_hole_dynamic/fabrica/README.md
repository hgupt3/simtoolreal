# Fabrica

RL-based robotic assembly using depth-based manipulation. Processes raw Fabrica benchmark meshes into training-ready assets and provides evaluation tools.

## Directory Structure

```
fabrica/
├── benchmark_processing/     # Asset pipeline (steps 0-6)
│   ├── fabrica_setup.md      # Pipeline documentation
│   ├── step0_import_assets.py
│   ├── step1_define_assembly_order.py
│   ├── step2_compute_canonical.py
│   ├── step3_generate_trajectories.py
│   ├── step4_run_coacd.py
│   ├── step5_generate_table_urdfs.py
│   └── step6_validate_insertions.py
├── objects.py                # Auto-registers assembly parts for training
├── viser_utils.py            # Shared viser viewer utilities
├── fabrica_eval.py           # Interactive policy evaluation
└── experiments/              # SLURM templates + experiment configs
```

## Asset Pipeline

See `benchmark_processing/fabrica_setup.md` for the full pipeline. Quick start for beam:

```bash
python fabrica/benchmark_processing/step0_import_assets.py --assemblies beam
python fabrica/benchmark_processing/step1_define_assembly_order.py --assembly beam
python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam
python fabrica/benchmark_processing/step3_generate_trajectories.py --assembly beam
for part in 0 1 2 3 6; do
    python fabrica/benchmark_processing/step4_run_coacd.py --assembly beam --part $part
done
python fabrica/benchmark_processing/step5_generate_table_urdfs.py --assembly beam
```

Generated assets live under `assets/urdf/fabrica/{assembly}/`:
```
beam/
├── assembly_order.json
├── canonical_transforms.json
├── {part_id}/              # meshes, URDFs, coacd/
├── trajectories/{part_id}/ # pick_place.json
└── environments/{part_id}/ # scene URDFs (vhacd, sdf, coacd)
```

## Evaluation

```bash
python fabrica/fabrica_eval.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --collision coacd
```

Web-based viser GUI for running policy rollouts. Select assembly/part from dropdowns, load the environment, and run episodes.

## Training

```bash
sbatch fabrica/experiments/baseline.sub
```

Edit the config section at the top of `baseline.sub` to set assembly, part, collision method, and training hyperparameters.
