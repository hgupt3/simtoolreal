# rl_games versus legacy simple_rl parity audit

This note records the audit performed on 2026-08-12 after the seed-0 reward
curves showed legacy simple_rl learning earlier than vendored rl_games. It
applies to these study variants:

- `rlgames-lstm-sapg-seed0`
- `legacy-simplerl-lstm-sapg-seed0`

## Conclusion

These runs test substantially similar LSTM SAPG agents, but they are not strict
implementation-parity runs. In particular, they use different actor learning
rate schedules. Their curve difference must not be classified as ordinary
seed noise until a strict optimizer-parity comparison is run.

The current experiment remains useful as a historical-backend comparison: it
tests the original rl_games configuration against the April simple_rl snapshot
as each was configured. It does not isolate the effect of the code cleanup.

## Confirmed similarities

Both runs use:

- 6,144 environments split into six equal SAPG policy blocks of 1,024
  environments;
- a learned 32-dimensional conditioning for each of the six policies;
- leader/follower experience sharing with one off-policy block;
- policy-dependent action standard deviations;
- a one-layer, 1,024-unit LSTM before an `[1024, 1024, 512, 512]` MLP;
- LSTM layer normalization, horizon 16, sequence length 16, and recurrent state
  reset on true environment `done`;
- an asymmetric central critic with the same MLP widths;
- two actor and critic mini-epochs and a nominal 98,304-sample minibatch;
- reward scale 0.01, gamma 0.99, GAE lambda 0.95, PPO clip 0.1, critic
  coefficient 4.0, and gradient norm 1.0;
- input, value, and advantage normalization; and
- seed 0 and the same deterministic cuboid task distribution.

The SAPG remainder is not silently dropped. Both dataset implementations extend
the final recurrent minibatch to the actual number of returns, so the augmented
remainder is included in the update.

## Confirmed differences

### Actor learning-rate schedule

rl_games inherits the following from `SimToolRealPPO.yaml`:

```yaml
learning_rate: 1e-4
lr_schedule: adaptive
schedule_type: standard
kl_threshold: 0.016
```

The legacy simple_rl study configuration only sets:

```yaml
learning_rate: 1e-4
```

Consequently, legacy simple_rl uses a constant actor learning rate while
rl_games adapts it from measured PPO KL. TensorBoard data collected during the
run confirms the difference:

| Environment frames | legacy simple_rl | rl_games |
| ---: | ---: | ---: |
| 0 | 0.0001000 | 0.0001000 |
| 100M | 0.0001000 | 0.0002250 |
| 300M | 0.0001000 | 0.0001500 |
| 500M | 0.0001000 | 0.0003375 |
| 700M | 0.0001000 | 0.0000444 |

Across the observed run, rl_games ranged from approximately `1.98e-5` to
`3.84e-3`; legacy simple_rl remained exactly at `1e-4`. This is large enough to
be a plausible cause of different learning-transition timing.

### Bounds loss

rl_games inherits `bounds_loss_coef: 0.0001`; the legacy simple_rl study sets
`bounds_loss_coef: 0.0`. This is probably smaller than the learning-rate
difference but is still a real optimizer mismatch.

### Random-number streams and numerical implementation

Using the same seed does not make the runs bitwise equivalent. Module
construction order, parameter initialization calls, rollout code, shuffling,
normalization updates, and kernel execution differ between implementations.
They therefore begin from different sampled parameters and do not encounter
identical trajectories. Multiple seeds are required to estimate the size of
this residual stochastic variation.

## Reward logging caveat

For mixed exploration, rl_games excludes the exploratory blocks from its
`rewards/step` statistic and reports the last 1,024-environment block: policy 5,
the zero-entropy exploitation/follower policy. simple_rl's `rewards/step`
aggregates completed episodes from all six blocks and separately records
`block_rewards/block_0` through `block_rewards/block_5`.

Therefore the original all-policy plot was not metric-equivalent. The matching
evaluation-policy comparison is:

- rl_games: `rewards/step`
- simple_rl: `block_rewards/block_5`

The corrected plot is
`results/reward_curves_eval_policy_seed0_20260812.png`. The rl_games learning
delay remains in that comparison, so the observed delay is not explained by
the logging mismatch.

## Interpretation of the current curves

The evidence supports these statements:

1. The initial reward plot contained a metric mismatch.
2. Correcting that mismatch does not remove the curve difference.
3. The two runs have a major measured learning-rate mismatch and a smaller
   bounds-loss mismatch.
4. One seed is insufficient to measure residual implementation variance.

The evidence does **not** support attributing the curve difference specifically
to the simple_rl cleanup, nor does it support calling the difference only noise.

## Recommended strict-parity follow-up

For a regression test of the cleanup itself, run at least three matched seeds
with:

- adaptive actor LR in both backends;
- `schedule_type: standard` and `kl_threshold: 0.016` in both;
- `bounds_loss_coef: 0.0001` in both;
- the existing matching architecture, SAPG, batch, task, and seed settings; and
- policy-5 reward as the primary evaluation curve, while retaining all-policy
  and per-block curves for diagnosis.

Log actor LR, KL, bounds loss, policy loss, critic losses, action sigma, and
normalization statistics from both backends. First compare their distributions
over frames; do not expect step-by-step identity because their random streams
and numerical implementations remain different.
