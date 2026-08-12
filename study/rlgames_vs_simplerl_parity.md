# rl_games, legacy simple_rl, and current simple_rl parity audit

Audit date: 2026-08-12. The immutable pre-fix study anchors are commit
`1fbee38b` in this repository and commit `ef910a8` in standalone simple_rl.
The five active workers were not modified or restarted by this audit.

This comparison uses the workers' saved `config_resolved.yaml` files (and the
rl_games saved Hydra config), the complete Hydra defaults chain, dataclass/code
fallbacks, launcher overrides, and training-entry-point mutations. It therefore
describes runtime behavior, not merely text that appears in one YAML file.

## Conclusion

The three LSTM SAPG runs use the same task, policy size, rollout shape, SAPG
population, and nearly all PPO coefficients. They are not strict backend-parity
runs, however. Four optimizer behaviors differ:

1. rl_games uses adaptive actor LR; both simple_rl runs use constant LR.
2. rl_games uses bounds coefficient `1e-4`; both simple_rl runs use zero.
3. rl_games' central value clip defaults to `0.2`; simple_rl explicitly uses
   `0.1`.
4. rl_games and legacy simple_rl train an auxiliary actor-side value head;
   pre-fix current simple_rl does not.

The first three are accidental study-configuration mismatches caused by
asymmetric inheritance/defaults, not core algorithm bugs. The fourth is a
current simple_rl parity regression: its comment claimed rl_games parity, but
rl_games continuous PPO defaults the auxiliary loss on. It is now explicit and
defaults on again in the post-anchor code.

Consequently, the reward difference is not evidence that cleanup alone improves
or hurts learning, and it should not be dismissed as seed noise. The measured
LR mismatch is large, and current-versus-legacy also differs in backbone value
supervision.

## How configuration actually resolves

### rl_games

Hydra composes:

```text
SimToolRealPPO
  -> SimToolRealLSTMAsymmetricPPO
     -> SimToolRealStudyRLGamesLSTMSAPG
        -> launch_transformer_study.py command-line overrides
        -> train.py preprocess_train_config
```

`SimToolRealLSTMAsymmetricPPO.yaml` and the study SAPG YAML do **not** override
the base PPO actor scheduler or bounds coefficient. They therefore retain
`adaptive`, `standard`, KL `0.016`, and bounds `1e-4`. The nested central critic
sets `clip_value: true` but no `e_clip`, so `central_value.py` supplies `0.2`.
The launcher changes `expl_coef_block_size` to `num_envs / 6 = 1024` and
`train.py` injects device/PBT metadata; neither changes the optimizer values.

### legacy and current simple_rl

The two simple_rl study YAMLs are complete `ppo:` mappings rather than children
of `SimToolRealPPO.yaml`. `train_simple_rl.py` converts the resolved mapping to
typed dataclasses, then only replaces the device from `cfg.rl_device`; it does
not copy rl_games defaults or mutate schedule, bounds, or critic clipping.

Missing simple_rl fields come from dataclass defaults. In particular,
`lr_schedule=None` means identity/constant LR and a missing `schedule_type` is
`legacy` but inert. Both study YAMLs explicitly set bounds to zero and central
`e_clip` to `0.1`.

### Exact effective training values

`default` below means a code/dataclass fallback after Hydra composition.

| Effective value | rl_games | legacy simple_rl | pre-fix current simple_rl |
| --- | ---: | ---: | ---: |
| Environments / SAPG blocks | 6144 / 6 | 6144 / 6 | 6144 / 6 |
| Horizon / sequence | 16 / 16 | 16 / 16 | 16 / 16 |
| Base / augmented batch | 98,304 / 114,688 | same | same |
| Requested minibatch / mini-epochs | 98,304 / 2 | same | same |
| Actor LR | `1e-4` adaptive | constant `1e-4` | constant `1e-4` |
| Schedule timing / KL target | standard / 0.016 | inert defaults | inert defaults |
| Bounds coefficient | `1e-4` | `0.0` | `0.0` |
| Actor PPO clip | 0.1 | 0.1 | 0.1 |
| Central value clip | default 0.2 | explicit 0.1 | explicit 0.1 |
| Auxiliary actor value loss | on (default) | on (unconditional) | off (hard-coded) |
| Actor/central critic mini-epochs | 2 / 2 | 2 / 2 | 2 / 2 |
| Reward scale / gamma / GAE lambda | .01 / .99 / .95 | same | same |
| Critic coefficient / grad norm | 4 / 1 | same | same |
| Input/value/advantage normalization | on/on/on | same | same |
| Mixed precision / gradient clipping | on/on | same | same |
| Episode value bootstrap | default false | default false | default false |
| Recurrent reset on `done` | default true | explicit true | explicit true |
| Initial log sigma / per-block sigma | 0 / yes | 0 / yes | 0 / yes |
| SAPG sharing / off-policy blocks | leader-follower / 1 | same | same |
| Entropy coefficient range | .0025 to 0 | same | same |

The augmented remainder is not dropped. Each implementation lets the final
minibatch absorb all 114,688 samples. Legacy simple_rl receives only the narrow
compatibility patch needed to support that batch and its asymmetric critic.

## Differences classified

### Accidental experimental mismatches, not library bugs

- **Adaptive versus constant LR.** Adaptive LR is an intentional rl_games
  feature, and constant LR is an intentional simple_rl default. Failing to set
  them identically in a claimed baseline comparison was the mistake. Observed
  rl_games LR ranged from about `1.98e-5` to `3.84e-3`, while simple_rl stayed at
  `1e-4`; this can readily alter the learning transition.
- **Bounds loss.** The rl_games `1e-4` value is inherited historical tuning.
  Zero is valid because actions are clipped before the environment and the
  coefficient is tiny. This is a parity mismatch, not evidence either backend
  computes the loss incorrectly.
- **Central critic clip 0.2 versus 0.1.** Both implementations correctly apply
  clipped value loss. The discrepancy comes from an omitted nested rl_games key
  falling back to `0.2`; simple_rl explicitly chose the main PPO clip of `0.1`.

### Fixed parity bug

Current simple_rl had unconditionally zeroed the policy network's value loss
whenever a privileged asymmetric critic existed. That is a defensible ablation,
but it was neither historical simple_rl behavior nor rl_games parity, and there
was no configuration switch. It also removes value-learning gradients from an
LSTM/Transformer policy backbone.

Post-anchor simple_rl now exposes:

```yaml
train_actor_value_with_asymmetric_critic: true
```

It defaults to `true`, restoring legacy/rl_games behavior. Setting it `false`
cleanly makes the external critic own all value learning. This is especially
important for interpreting the transformer runs: the LocoFormer paper trains a
value decoder from the temporal representation, while the pre-fix active
transformer receives policy/entropy gradients but no value-loss gradient in its
temporal backbone.

### Intentional or behaviorally irrelevant differences

- rl_games identifies SAPG blocks with fixed float IDs and then looks up learned
  rows; simple_rl appends integer IDs and performs the same learned-row lookup.
  Both avoid normalizing the ID and both initialize learned rows with `randn`.
- Current simple_rl makes recurrent SAPG state reuse explicit, fixes shuffling
  for `seq_length < horizon`, and supports `memory_reset`. With the active LSTM
  `seq_length == horizon` and ordinary `done` resets, these fixes do not explain
  the legacy/current curve gap.
- Current has a wall-clock stop; legacy/rl_games use the launcher's external
  deadline. This changes termination, not updates before termination.
- Player determinism settings affect evaluation only, not these training jobs.
- Framework construction order and kernels consume random numbers differently.
  Equal integer seeds cannot make independent implementations bitwise equal.

## Reward metric caveat

rl_games `rewards/step` excludes exploratory blocks and reports policy 5, the
zero-entropy exploitation/follower policy. simple_rl `rewards/step` aggregates
all blocks and separately logs `block_rewards/block_0` through block 5.

The matching comparison is rl_games `rewards/step` versus simple_rl
`block_rewards/block_5`, plotted in
`results/reward_curves_eval_policy_seed0_20260812.png`. The delay remains after
this correction, so it is not merely a logging artifact.

## Defaults moving forward

There are two distinct goals and they should use named configurations:

1. **Production/exploration default:** keep constant actor LR `1e-4`, bounds
   disabled, central clip `0.1`, and the auxiliary actor value loss enabled.
   Constant LR has been substantially less volatile in this run and the two
   simple_rl curves transitioned earlier, although more seeds are still needed.
   Bounds zero and matching 0.1 actor/critic clips are the simpler choices.
2. **Backend regression parity:** match historical rl_games exactly with
   adaptive/standard LR, KL `0.016`, bounds `1e-4`, central clip `0.2`, and the
   auxiliary actor value loss enabled. New
   `SimToolRealStudyLegacyLSTMSAPGParity.yaml` and
   `SimToolRealStudyCurrentLSTMSAPGParity.yaml` encode these values instead of
   relying on hidden defaults.

Do not replace the original run configurations: they are evidence of what the
active curves actually tested. For the strict parity follow-up, run the existing
rl_games variant and both new parity variants for at least three seeds, compare
policy-5 reward, and log LR, KL, actor/central value losses, bounds loss, sigma,
and normalization statistics.

The existing single-seed evidence supports keeping the stable production
defaults above; it does not yet establish that they are universally superior.
