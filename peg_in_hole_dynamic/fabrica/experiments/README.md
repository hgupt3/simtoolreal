# Beam Assembly Post-Training Experiments

Post-training SimToolReal for contact-rich beam assembly insertion.
- Robot: Kuka IIWA 14 + SHARPA (29 DOF)
- Simulator: Isaac Gym, 24K envs, LSTM asymmetric PPO
- [WandB Dashboard](https://wandb.ai/kk837/FABRICA_TRAINING)

## Phases
1. [Curriculum ablation](phase1_curriculum_ablation/README.md)
2. [Scale to all parts](phase2_scale_parts/README.md)
3. [Retract after insertion](phase3_retract/README.md)
4. [Cross-part generalization](phase4_generalization/README.md)

## Combined Results
| Part | Insertion | FGT | Retract | Generalization |
|------|----------|-----|---------|---------------|
| 6    |          |     |         |               |
| 2    | 12/12    | 0.002m | | |
| 0    |          |     |         |               |
| 3    |          |     |         |               |
| 1    |          |     |         |               |
