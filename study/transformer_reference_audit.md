# Transformer and recurrent-policy reference audit

Audit date: 2026-08-12. This is a read-only audit of the code used by the five
active GCP runs. No running process or resolved configuration was changed.

## Sources and revisions

- LocoFormer v1 PDF: `/juno/u/tylerlum/Downloads/2509.23745v1.pdf`, SHA-256
  `248ba53b1feb2ede28fa1b50deee0c75d4e8e76e548097a2d37e042f418b07ca`.
- [LocoFormer project page](https://generalist-locomotion.github.io/) and
  [paper](https://arxiv.org/abs/2509.23745).
- [Denys88/IsaacGymEnvs](https://github.com/Denys88/IsaacGymEnvs), commit
  `a9e8a2d5ecd8fab7716ad0b980aae1feb580ea58`.
- [AntPPO_tr.yaml](https://github.com/Denys88/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/train/AntPPO_tr.yaml)
  and [ig_networks.py](https://github.com/Denys88/IsaacGymEnvs/blob/main/isaacgymenvs/learning/networks/ig_networks.py).
- [rl_games PR #163](https://github.com/Denys88/rl_games/pull/163), head
  `3907e0e610a7a364b7952e2cdb54e0545d7235c9`.
- [rl_games issue #194](https://github.com/Denys88/rl_games/issues/194).
- Vendored rl_games and the April/current simple_rl implementations in this
  branch.

## Bottom line

The current segment implementation is a sound, PPO-compatible approximation of
the Transformer-XL mechanism described by LocoFormer. Its causal masking,
detached per-layer memory, segment alignment, rollout/replay equivalence, and
episode reset behavior are internally consistent and covered by targeted tests.

It is not an exact reproduction of the unpublished LocoFormer code. The paper
does not report most PPO or model-width hyperparameters, and our experiment
does not reproduce the paper's most important training construction: multiple
physical trials, including failures, inside a longer episode whose memory
persists. The active SimToolReal policy keeps memory across successful goal
changes, but clears it on object/hand failures, timeouts, or any other true
environment `done`.

The historical IsaacGym transformer is a different idea. It is a fixed
four-frame observation encoder, not Transformer-XL and not a recurrent policy.
It cannot support long-context or cross-trial adaptation.

## Three transformer designs

| Property | IsaacGym example | simple_rl rolling | simple_rl segment / Loco variant |
| --- | --- | --- | --- |
| Temporal source | Fixed four-frame stack | Recurrent sliding history | Previous TXL segment plus current segment |
| Persistent policy state | No (`is_rnn=False`) | Yes | Yes |
| Training sequence | Four stacked frames in one observation | PPO sequence, currently 16 | PPO segment, currently 128 |
| Causal | No explicit causal mask; pooled encoder | Yes | Yes |
| Memory gradient | Not applicable | Initial cache detached | Previous segment detached |
| Reset boundary | Frame stack follows env reset | `memory_reset` | `memory_reset` with episode-aware mask |
| Position scheme | Learnable positions for Ant | RoPE | RoPE |
| Inference cache | Rebuilds fixed stack | Hidden-state ring | Hidden-state ring; not a true KV cache |
| Direct receptive field | Four frames | 16 prior states plus current | 128 prior states plus up to 128 current states |
| Layer-expanded field | None | Fixed window | Up to `(layers + 1) * segment_length` |

### Denys88 IsaacGym example

IsaacGym's modified `VecTask` stacks four observations by default without
flattening. `TransformerModel` projects each of the four observation vectors to
256 dimensions, applies a two-layer/two-head Transformer for Ant, pools the
sequence, and jointly decodes action mean and value. Ant uses learnable
positions. The `input_split` list in `AntPPO_tr.yaml` is unused because
`split_features` remains false.

The wrapper returns `is_rnn=False`, `get_default_rnn_state=None`, and does not
use a causal mask. It therefore has a short temporal view but no state from one
policy call to the next. Its PPO settings are useful historical tuning clues
(`lr=3e-4`, horizon 16, minibatch 4,096, four mini-epochs, clip 0.2, no LR
schedule, no mixed precision), but they are not evidence for long-context
online RL.

### simple_rl rolling transformer

The rolling mode stores each layer's input hidden state in a length-16 ring.
During replay, query `i` sees the surviving initial-memory slots and current
tokens through `i`, reproducing the eviction that occurred step by step during
rollout. It is causal and rollout output matches sequence replay in tests.

This mode is a compact recency model, not an approximation to a 16-token
stateless encoder. It has an effective window of 16 prior observations plus the
current observation at inference.

### simple_rl segment transformer

For each layer, the state stores the hidden inputs needed as keys and values for
the next segment. Training concatenates the detached previous 128-state segment
with the causally masked current 128-state segment. Higher layers propagate
older information through the cached representations. With six layers and
length 128, the theoretical largest receptive field is 896 timesteps, matching
the number stated by LocoFormer.

The global segment phase is intentionally not reset per environment. Vectorized
environments must remain on the same segment boundary, while their memory
content and validity masks reset independently. This is why reset clears content
but preserves the phase tensor.

## LocoFormer: reported versus current experiment

### What the paper actually reports

- PPO in large-scale physical simulation.
- Transformer-XL with observation MLP embedding and separate actor/value MLP
  heads.
- Segment length 128 and an illustrated six-layer model.
- Detached cached hidden states from the preceding segment.
- An effective field of 896 timesteps: about 18 seconds at 50 Hz.
- A deployment KV cache retaining the most recent `2L - 1` prior timesteps.
- Multiple physical trials inside one episode. Memory persists between trials
  but attention is masked across outer episode boundaries.
- A sampled number of trials, later expressed as an adaptation-time budget
  `u ~ Uniform(0, U)` before a final trial.
- Two training phases: short trials/small adaptation budget first, then longer
  trials/larger adaptation budget.
- Roughly 100,000 procedurally generated robots with broad morphology and
  dynamics randomization.

The paper does **not** report the observation embedding width, number of heads,
FFN width, actor/critic head widths, PPO learning rate, clip, batch/minibatch
size, mini-epochs, GAE lambda, discount, entropy coefficient, optimizer, or
initial action standard deviation. Those values cannot be copied faithfully
from the paper. A third-party package named `locoformer` is not the authors'
training code and must not be treated as a source for their hyperparameters.

### What matches

- Six Transformer layers and segment length 128.
- Standard multi-head attention and feed-forward blocks for the Loco variant.
- Observation MLP embedding and separate actor/critic output heads.
- Causal current-segment attention to detached previous-segment layer states.
- No gradient through the cached segment.
- Per-environment validity/reset masking.
- Layer-expanded context, tested explicitly beyond one segment.
- Rollout versus PPO-replay output equivalence, including resets.

At SimToolReal's 60 Hz, 896 steps are about 14.9 seconds rather than the
paper's 18 seconds at 50 Hz. The directly retained `2L - 1 = 255` prior states
cover about 4.25 seconds.

### What differs or remains unknown

1. **Trial construction is not reproduced.** SimToolReal changes the target
   after a success without `done`, so memory persists across goals. Object/hand
   failures, extreme separation, timeouts, and other true resets emit `done`
   and clear memory. LocoFormer specifically learns from failed physical trials
   retained inside a longer episode.
2. **The paper's task diversity is absent.** Object-shape variation and SAPG
   exploration are not equivalent to 100,000 bodies plus aggressive dynamics
   randomization. Architecture alone does not create in-context adaptation.
3. **The cache is functionally correct but not the paper's optimized KV
   cache.** simple_rl stores hidden states and recomputes their K/V projections
   on each step. This preserves the intended attention result at dropout zero
   but costs more inference compute.
4. **Positional and normalization details are underspecified by the paper.** We
   use RoPE, pre-layer normalization, GELU FFNs, and no dropout. These are
   reasonable choices, not confirmed reproductions.
5. **The active run uses a privileged external critic.** The paper describes a
   value head decoded from the TXL output. Current simple_rl sets that actor-side
   value loss to zero when the asymmetric critic exists, leaving the TXL
   backbone without a value-loss gradient.
6. **SAPG sharing is an additional off-policy approximation.** Relabelled
   follower samples reuse memory generated under their source conditioning.
   The code makes this explicit as `recurrent_experience_sharing: reuse_state`,
   but it is less principled for a 128-step history than for an MLP.
7. **The model capacity is locally chosen.** The paper gives no widths. With the
   live 140-dimensional observation and 29 actions, the current actor policy has
   approximately 1.30M trainable parameters; the current LSTM policy has 7.81M
   and rolling transformer 0.38M. These curves are not parameter-matched.

## Current study confounders

The architecture curves intentionally explore working configurations, not a
single-variable architecture ablation:

| Variant | Actor parameters | Horizon / sequence | LR | Initial action std |
| --- | ---: | ---: | ---: | ---: |
| Current LSTM | 7.81M | 16 / 16 | `1e-4` | `exp(0)=1.0` |
| Rolling | 0.38M | 16 / 16 | `3e-4` | `exp(-1)=0.368` |
| Loco/TXL | 1.30M | 128 / 128 | `1e-4` | `exp(0)=1.0` |

Loco/TXL also collects eight times as many frames per PPO epoch and processes
nine augmented minibatches per mini-epoch instead of one oversized augmented
minibatch for the 16-step variants. Frame-normalized reward plots remain useful,
but they do not isolate architecture, context, model size, exploration scale, or
optimizer cadence.

## rl_games PR #163 and issue #194

PR #163 is not transformer code. It fixed/refactored optimizer stepping so that
mixed-precision scaling and multi-GPU synchronization work whether gradient
clipping is enabled or disabled. The required order is:

1. backpropagate the scaled loss;
2. synchronize gradients for multi-GPU;
3. unscale before clipping, when clipping is enabled;
4. call scaler step and update in every configuration.

simple_rl follows those essential semantics. The active jobs are single-GPU,
so the PR's distributed corner case does not affect them.

Issue #194's requested capabilities map as follows:

- custom networks: simple_rl accepts network dataclasses directly;
- external recurrent-state manipulation: state is explicit and resettable, but
  there is not yet a supported corruption callback;
- checkpointed test/player mode: present and tested;
- dropout: deliberately rejected for temporal policies because independent
  rollout/replay masks break PPO likelihood equivalence unless masks are stored
  or made deterministic; and
- single/multi-GPU and memory-limit tests: single-GPU correctness exists, but
  multi-GPU transformer and automated peak-memory coverage are still missing.

### Heavy-network mixed-precision check

A synthetic optimizer microbenchmark was run on an RTX 4090 with 32,768 tokens
per update, gradient scaling, unscale, clipping, and Adam:

| Network | FP32 | AMP | FP32 peak | AMP peak |
| --- | ---: | ---: | ---: | ---: |
| 6-layer rolling Qwen, context 16 | 46.7 ms | 27.3 ms | 3.70 GiB | 2.85 GiB |
| 6-layer TXL, segment 128 | 34.1 ms | 20.2 ms | 2.61 GiB | 1.77 GiB |

This is not an end-to-end Isaac Gym throughput result, but it confirms the PR
discussion's important qualification: AMP may do little for small MLPs while
being materially useful for larger attention models. A precision-only replay
diagnostic measured small initial Gaussian KL (`~2e-5` rolling and `~1e-6`
segment), so FP32 rollout versus AMP update is not currently an obvious failure,
though pre-update KL should remain logged.

## Legacy versus current simple_rl

Current simple_rl adds or fixes:

- rolling and segment transformer state plumbing;
- explicit `memory_reset` distinct from environment `done`;
- sequence-level SAPG shuffling that keeps memory snapshots paired with data;
- explicit opt-in for recurrent SAPG state reuse;
- recurrent reset code that preserves singleton time/batch axes instead of
  unsafe bare `squeeze()` calls;
- removal of the old `bptt_len` argument, which was explicitly ignored;
- validation that horizons/minibatches divide into full sequences;
- last-minibatch absorption of augmented SAPG remainders;
- EPO replacement reset of only the killed policies' memory; and
- checkpoint/statistics and wall-time termination fixes.

For all active variants `seq_length == horizon_length`, so the legacy recurrent
shuffle issue and ignored `bptt_len` do not change these particular LSTM runs.

One current change is not a pure parity fix: current simple_rl disables the
actor value loss when the external asymmetric critic is enabled. Both legacy
simple_rl and this vendored rl_games train that auxiliary actor value head. The
current comment claiming rl_games parity is incorrect. Make this behavior an
explicit configuration option before the next parity matrix.

## Correctness checks completed

- All 49 simple_rl tests pass.
- Tests cover causal masking, rollout/replay equality across chunks, independent
  resets, context beyond one segment, segment-phase preservation, transformer
  PPO/checkpoint/player paths, SAPG sharing for both transformer modes and LSTM,
  sequence/state shuffling, remainder minibatches, per-policy sigma, EPO memory
  reset, and singleton dimensions.
- Current study and standalone simple_rl copies of `agent.py`, `network.py`, and
  `transformer.py` have identical SHA-256 hashes.
- The four current-code VMs use tracked commit `125ead58`; the legacy worker uses
  tracked commit `76657dae`. Their only untracked paths are runtime/evaluation
  outputs. The active core `agent.py`, `network.py`, and `transformer.py` files
  exactly match the committed local copies.

Still missing are multi-GPU transformer tests, an AMP rollout/replay regression
test, a resume test taken at every segment phase, and an end-to-end test where a
policy demonstrably improves over several failed physical trials.

## Ranked next experiments

Do not alter or stop the active five-run matrix. For the next matrix:

1. **Establish PPO parity first.** Match adaptive LR, KL threshold, bounds loss,
   and actor-side value-loss behavior across rl_games, legacy, and current
   simple_rl. Use at least three seeds and compare policy-5 reward.
2. **Create true multi-trial episodes.** Put several physical object-reset
   trials—including failure/drop trials—inside one outer episode. Preserve
   memory at trial reset and clear it only when object/dynamics identity changes
   at the outer boundary. This tests the central LocoFormer claim.
3. **Run a reset ablation.** Compare clearing memory on every physical reset,
   preserving it across trials, and the existing goal-only persistence. Log
   reward by trial number, not only whole-episode reward.
4. **Parameter-match temporal policies.** Match actor parameter count, initial
   sigma, LR schedule, horizon/update ratio, and number of optimizer sample
   passes before attributing differences to LSTM versus attention.
5. **Keep SAPG sharing, but measure its approximation.** Retain the requested
   leader/follower sharing as the main run. Add one inexpensive diagnostic seed
   comparing reused source history against target-conditioned history
   recomputation, and log on/off-policy KL and gradient alignment.
6. **Sweep context economically.** Try segment lengths 64 and 128 with two,
   four, and six layers. The relevant quantity is both direct cache `2L-1` and
   layer-expanded field `(N+1)L`; increasing only `L` is expensive.
7. **Optimize after correctness.** Implement a real per-layer KV cache and
   consider AMP rollout/state storage only after an equivalence test. Current
   hidden-state caching is correct but leaves inference speed and several GiB of
   state memory on the table.

Highest-value tuning/logging parameters are context length, layer count,
embedding/FFN width, actor LR and schedule, initial sigma, PPO clip, number of
mini-epochs, gradient norm, SAPG entropy scale, off-policy ratio, reset policy,
and number/duration of trials per outer episode. For in-context learning, reset
semantics and task variation are more important than blindly making the
Transformer wider.
