# SimToolReal SAPG Transformer Study

This directory records the decisions behind the seven-run GCP study. The launch
entry point is `python isaacgymenvs/launch_transformer_study.py --variant ...`.

## Matrix

| Variant | Backend | Temporal policy | Horizon / sequence / context |
| --- | --- | --- | --- |
| `rlgames-lstm-sapg` | vendored rl_games | historical 1024-unit LSTM | 16 / 16 / n/a |
| `legacy-simplerl-lstm-sapg` | April simple_rl snapshot | 1024-unit LSTM | 16 / 16 / n/a |
| `current-simplerl-lstm-sapg` | current simple_rl | 1024-unit LSTM | 16 / 16 / n/a |
| `current-simplerl-rolling16-sapg` | current simple_rl | 2-layer Qwen-style rolling cache | 16 / 16 / 16 |
| `current-simplerl-loco128-sapg` | current simple_rl | 6-layer Transformer-XL segment memory | 128 / 128 / 128 |
| `legacy-simplerl-lstm-sapg-parity` | April simple_rl snapshot | 1024-unit LSTM, rl_games optimizer settings | 16 / 16 / n/a |
| `current-simplerl-lstm-sapg-parity` | current simple_rl | 1024-unit LSTM, rl_games optimizer settings | 16 / 16 / n/a |

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

The rl_games and legacy simple_rl LSTM runs are not strict optimizer-parity
runs. See [`rlgames_vs_simplerl_parity.md`](rlgames_vs_simplerl_parity.md) for
the fully resolved three-way configuration, intentional differences, fixed
actor-value-loss regression, reward-metric caveat, recommended defaults, and
the two explicit `*-parity` follow-up variants.

See [`transformer_reference_audit.md`](transformer_reference_audit.md) for the
detailed comparison against Denys88's IsaacGymEnvs transformer example,
rl_games PR #163 and issue #194, and the LocoFormer paper. It also records the
remaining implementation limitations and a ranked next-experiment plan.

## Environment

All seven runs use `objectName=allegro_kuka_cuboids`: the ordered 654-shape small
cuboid pool from Allegro-Kuka, generated deterministically from a 5 cm base,
normalized volumes 1.0–2.5, density 400 kg/m^3, side lengths 2.5–15 cm, and
masses 50–125 g. Tool heads, cylinders, big cuboids, and sticks are excluded.
This is an easier controlled debugging distribution, so the old heavy-tool W&B
curve is context rather than a direct reproduction target.

## Scale and stopping

The production matrix uses 6,144 environments. A 12,288-environment LocoFormer
cannot fit. The full 6,144-environment LocoFormer smoke completed collection and
updates, peaking at about 38.5 GiB on the 40 GiB A100. This is tighter than the
preferred 10% reserve, but 6,144 is the smallest environment count compatible
with both six equal policy blocks and the unchanged 98,304 minibatch. Reducing
again would also change the optimizer batch and confound the comparison, so the
validated exact-batch configuration was retained.

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

## Parity bridge interpretation

The two parity workers were added after resolving the hidden optimizer defaults.
They use adaptive/standard actor LR with KL target `0.016`, bounds coefficient
`1e-4`, central value clip `0.2`, and the actor-side auxiliary value loss. This
matches the existing rl_games run while changing only the backend generation.

- If both parity curves approach rl_games, configuration explains most of the
  original gap.
- If legacy parity stays near legacy simple_rl, the remaining gap is primarily
  implementation or random-stream behavior.
- If current parity separates from legacy parity, a post-April simple_rl change
  remains behaviorally important.

Seed 0 is a diagnostic bridge, not a statistical conclusion. Only after seeing
which branch of this decision tree occurs should the most informative pair be
repeated at seeds 1 and 2.

## GCP operation

`study/gcp/startup.sh` installs the run and ten-minute synchronization services.
Workers receive `study-variant`, `study-num-envs`, `study-seed`, and
`study-bucket` as instance metadata. Logs, TensorBoard events, configs,
checkpoints, GPU status, and heartbeats are mirrored to
`gs://gcp-gentoolreal-simtoolreal-transformer` every ten minutes. A failed
service resumes the newest local checkpoint. The worker maps its intentional
seven-day timeout to a successful service exit, so completion is not restarted;
current simple_rl gets a ten-minute margin to checkpoint at an epoch boundary.

No W&B credential was present on the workstation or source image. Runs still
activate W&B with project `simtoolreal_transformer` in offline mode, preserving
complete run directories for later `wandb sync` without baking a secret into a
shared image. GCS and TensorBoard are the live durable telemetry meanwhile.
