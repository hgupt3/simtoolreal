# SimToolReal SAPG Transformer Study

This directory records the decisions behind the five-run GCP study. The launch
entry point is `python isaacgymenvs/launch_transformer_study.py --variant ...`.

## Matrix

| Variant | Backend | Temporal policy | Horizon / sequence / context |
| --- | --- | --- | --- |
| `rlgames-lstm-sapg` | vendored rl_games | historical 1024-unit LSTM | 16 / 16 / n/a |
| `legacy-simplerl-lstm-sapg` | April simple_rl snapshot | 1024-unit LSTM | 16 / 16 / n/a |
| `current-simplerl-lstm-sapg` | current simple_rl | 1024-unit LSTM | 16 / 16 / n/a |
| `current-simplerl-rolling16-sapg` | current simple_rl | 2-layer Qwen-style rolling cache | 16 / 16 / 16 |
| `current-simplerl-loco128-sapg` | current simple_rl | 6-layer Transformer-XL segment memory | 128 / 128 / 128 |

Every run uses seed 0, six SAPG policies, leader/follower experience sharing,
one off-policy block, conditioning width 32, entropy scale 0.005, an asymmetric
MLP critic, two mini-epochs, and a nominal minibatch of 98,304. SAPG's shared
remainder is absorbed by the last minibatch, matching the reference code.
Current recurrent policies explicitly select `recurrent_experience_sharing:
reuse_state`; this reuses source-policy memory under the relabelled conditioning
and is therefore an intentional off-policy approximation.

The April snapshot is extracted from SimToolReal commit `5e831aaa` at launch.
`legacy_simple_rl_batch_compat.patch` only backports remainder-batch and
asymmetric-critic dataset sizing so the legacy run can use exactly the same
98,304 minibatch setting. It does not backport current model or PPO behavior.

## Environment

All five runs use `objectName=allegro_kuka_cuboids`: the ordered 654-shape small
cuboid pool from Allegro-Kuka, generated deterministically from a 5 cm base,
normalized volumes 1.0–2.5, density 400 kg/m^3, side lengths 2.5–15 cm, and
masses 50–125 g. Tool heads, cylinders, big cuboids, and sticks are excluded.
This is an easier controlled debugging distribution, so the old heavy-tool W&B
curve is context rather than a direct reproduction target.

## Scale and stopping

Start with 12,288 environments. A full-memory smoke must leave at least 10% of
the A100's 40 GB free after one collection and update. If any architecture does
not, lower all five runs to the same largest value that is divisible by six and
keeps `num_envs * 16` divisible by 98,304 (6,144 is the next candidate).

Runs stop at the first of 70 billion frames or seven days. Current simple_rl
does this at an epoch boundary and writes a final checkpoint. Infrastructure
also enforces the seven-day limit for legacy/rl_games. Checkpoints, resolved
configs, logs, and a heartbeat are synchronized externally.

## Deliberate non-choices

- No vanilla PPO, EPO, or no-sharing ablation is in this matrix.
- Goal changes inside an episode do not clear recurrent memory; only the true
  outer `done` boundary does. This is required for LocoFormer-style in-context
  adaptation across locomotion/task trials.
- Transformer dropout is zero so rollout and PPO replay use identical models.
