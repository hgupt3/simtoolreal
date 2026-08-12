# Eigengrasp action-space experiment log

Launch date: 2026-08-12. Code revision: `f6965fe5` on
`2026-08-12_Eigengrasp_Action_Space`.

These workers extend the original seven-run transformer study without stopping
or modifying its workers. At launch, the project had exactly ten running A100
instances: the original seven and the following three non-preemptible
`a2-highgpu-1g` workers in `us-central1-f`:

| Instance | Variant | Hand action width | Decoder |
| --- | --- | ---: | --- |
| `simtoolreal-study-run-pca22-clamp` | `current-simplerl-lstm-sapg-pca22-clamp` | 22 | literal P1/P99 PCA, then joint clamp |
| `simtoolreal-study-run-pca22-gauge` | `current-simplerl-lstm-sapg-pca22-gauge` | 22 | full-rank radial gauge map |
| `simtoolreal-study-run-pca5-clamp` | `current-simplerl-lstm-sapg-pca5-clamp` | 5 | literal top-five P1/P99 PCA |

All three use 6,144 environments, seed 0, the current simple_rl 1,024-unit
LSTM, six-policy SAPG, `use_others_experience=true`, and
`recurrent_experience_sharing=reuse_state`. The task uses the same deterministic
Allegro-Kuka cuboid distribution as the original study. The arm retains seven
delta actions and the hand retains the stock `handMovingAverage=0.1` target
filter. Thus the full-rank workers expose 29 total actions and PCA-5 exposes 12.

## Launch validation

The literal PCA-22 worker was held as a smoke test before allocating the final
two machines. It passed the live left-Sharpa joint-order contract, reported a
29-dimensional action box, loaded all 22 ARCTIC components, enabled recurrent
SAPG sharing, and trained for more than 7.4 million frames with finite rewards.

Afterward, both remaining workers passed the same integration checks. PCA-22
gauge reported a 29-dimensional action box and PCA-5 reported a 12-dimensional
action box with 0.713657 retained variance. Both produced finite aggregate and
zero-entropy-follower rewards and sustained about 69--73k total frames/second in
their initial epochs. GPU memory use settled near 13--14 GiB. Logs, event files,
resolved configs, checkpoints, GPU status, and heartbeats sync to
`gs://gcp-gentoolreal-simtoolreal-transformer/<run-name>/`.

The first few million frames are only a health check, not evidence that one
parameterization is better. Compare curves at a common frame count. If PCA-5 is
promising, repeat its best comparison with the entropy coefficient adjusted per
action dimension so lower dimensionality is not confounded with a smaller
summed Gaussian entropy bonus.
