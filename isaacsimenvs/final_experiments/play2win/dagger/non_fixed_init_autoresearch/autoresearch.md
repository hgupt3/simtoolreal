# Autoresearch: DAgger non-fixed-init hyperparameter optimization

You are a Claude Code agent. Your mission is to **find the combination of `NUM_ENVS`, `MINIBATCH_SIZE`, `mini_epochs`, and `learning_rate` that maximizes `max(successes)` achieved within a fixed 1-hour (3600 s) training-time budget** on the non-fixed-init DAgger distillation task in `/share/portal/kk837/depthbasedRL`.

Baseline: `baseline.sub` is a copy of `../distill_lpeg_tol0p5mm_bc_only_det_n512.sub` with `MINI_EPOCHS` and `LR` exposed as bash vars (yaml defaults `mini_epochs=2`, `learning_rate=1e-4`). Run it once as your first action to **measure the true baseline max-successes on this hardware** before starting the optimization loop. Higher is better.

You **MUST loop forever** until externally interrupted. Do not stop because you ran out of ideas or because the user might be asleep — think harder, try a new region of the search space.

---

## Setup (one-shot, before any experiments)

```bash
cd /share/portal/kk837/depthbasedRL

# Use the existing autoresearch branch (same branch the fixed-init loop ran on).
# If you're not already on it, switch over.
git checkout autoresearch/dagger-hpopt 2>/dev/null || true

# Initialize the experiment ledger if it doesn't exist.
TSV=isaacsimenvs/final_experiments/play2win/dagger/non_fixed_init_autoresearch/results.tsv
if [ ! -s "$TSV" ]; then
  printf "timestamp\tcommit\texperiment_tag\tnum_envs\tminibatch_size\tmini_epochs\tlearning_rate\texpl_block_size\tmax_successes\tlast_elapsed_s\tstatus\tdescription\n" > "$TSV"
fi
```

Read these three files end-to-end before touching anything:
- `isaacsimenvs/final_experiments/play2win/dagger/non_fixed_init_autoresearch/baseline.sub` — the baseline you'll be optimizing.
- `isaacsimenvs/cfg/train/PegInHoleDepthStudentSAPG.yaml` — confirms yaml defaults (`mini_epochs: 2`, `learning_rate: 1e-4`).
- This file (`autoresearch.md`) — re-read at the start of every iteration in case you forgot a rule.

### Measure the true baseline (one experiment, single slot)

Before opening the 2-slot loop, run `baseline.sub` once on this hardware and record its `max(successes)` over the first 3600 s of training. **Do not trust any prior expectation** — measure it.

```bash
JOBID=$(sbatch isaacsimenvs/final_experiments/play2win/dagger/non_fixed_init_autoresearch/baseline.sub | awk '{print $NF}')
echo "baseline job: $JOBID"
# Wait for the run dir to appear:
while true; do
  RUN_DIR=$(ls -dt /share/portal/kk837/depthbasedRL/train_dir/dagger_nonfixed_autoresearch/bc_only_det/baseline_n512_mb1024_me2_lr1e-4_* 2>/dev/null | head -1)
  [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] && break
  sleep 30
done
```

Poll via the `max_successes_in_budget` Python snippet below until the training-time elapsed exceeds 3600 s OR the job crashes. Once 3600 s of training time has elapsed:

```bash
# Get the metric:
#   MAX_SUCC = max successes value over events with wall_time ≤ t_train_start + 3600s
# Then:
BASELINE_COMMIT=$(git rev-parse --short HEAD)
printf "%s\t%s\tbaseline\t512\t1024\t2\t1e-4\t128\t%s\t%s\tkeep\tmeasured baseline on this hardware\n" \
  "$(date -Iseconds)" "$BASELINE_COMMIT" "$MAX_SUCC" "$LAST_ELAPSED" >> "$TSV"
# scancel the job — it will keep running otherwise.
scancel "$JOBID"
```

This row's `max_successes` becomes the **initial "current best"** that every subsequent experiment is compared against. Only after this row is written do you enter the 2-slot loop. If the baseline run fails (crash, OOM, etc), **halt and report** — something is broken at the infrastructure level.

---

## Scope of allowed edits

**You may ONLY edit files matching the glob**
`isaacsimenvs/final_experiments/play2win/dagger/non_fixed_init_autoresearch/distill_*.sub`.

You may create new variant subs in the same directory. You may not touch:
- any `.py` file, anywhere in the repo;
- any `.yaml` file, anywhere in the repo;
- `.gitignore`, `pyproject.toml`, the `rl_games/` or `isaaclab/` submodules;
- subs outside `non_fixed_init_autoresearch/` (the parent `dagger/` folder is off-limits).

If a knob you'd like to tune isn't exposed as a `agent.params.config.*` / `env.*` override in the baseline sub, **stop and add it via the sub's CLI override block** — never edit the yaml directly.

---

## The four tunable knobs

| Knob | Bash var in sub | Hydra override line | Yaml default | Baseline value |
|------|-----------------|---------------------|--------------|----------------|
| Envs per rollout | `NUM_ENVS` | `env.scene.num_envs=$NUM_ENVS` | — | 512 |
| Minibatch size | `MINIBATCH_SIZE` | `agent.params.config.minibatch_size=$MINIBATCH_SIZE` *and* `agent.params.config.central_value_config.minibatch_size=$MINIBATCH_SIZE` | 16384 | 1024 |
| Mini-epochs | `MINI_EPOCHS` | `agent.params.config.mini_epochs=$MINI_EPOCHS` | 2 | 2 |
| Learning rate | `LR` | `agent.params.config.learning_rate=$LR` | 1e-4 | 1e-4 |

All four are pre-exposed as bash variables at the top of `baseline.sub` with their corresponding `agent.params.config.*` CLI lines wired up. Do **not** add a fifth CLI override line — every knob the autoresearch loop should touch is already plumbed.

---

## Hard constraints — verify before EVERY `sbatch`

1. `NUM_ENVS % EXPL_COEF_BLOCK_SIZE == 0`. The baseline uses 512 / 128 = 4 SAPG blocks. If you change `NUM_ENVS`, scale `EXPL_BLOCK` to keep ~4 blocks (or any divisor of `NUM_ENVS`).
2. `(NUM_ENVS * 16) % MINIBATCH_SIZE == 0` — horizon is 16, total batch = `NUM_ENVS × 16`, must be cleanly divisible by minibatch.
3. `#SBATCH --mem` scales roughly linearly with `NUM_ENVS`. Reference points: 100 GB @ 512 envs; 400 GB @ 4096 envs. Bump it before submitting larger runs; drop for smaller ones.
4. The following overrides MUST remain in every variant (else the run is invalid — discard immediately if you forgot):
   - `agent.params.config.lr_schedule=identity` (adaptive scheduler crushes LR to 1e-6 for BC; this fix is non-negotiable);
   - `USE_OBS_DELAY=True`, `USE_ACTION_DELAY=True`, `USE_CAMERA_DELAY=True` (all max=3); delays are part of the testbed and are not under tuning;
   - all `RESET_POSITION_NOISE_*`, `RESET_DOF_POS_NOISE_*`, `RESET_DOF_VEL_NOISE`, `TABLE_RESET_Z_RANGE` knobs unchanged from baseline (these define the "non-fixed-init" task; tuning them changes the task).
   - `WANDB_PROJECT=dagger_nonfixed_autoresearch`, `WANDB_GROUP=bc_only_det` (so all variants plot together).
5. `EXPERIMENT_TAG` must be unique vs. every prior row in `results.tsv`. Suggested format: `nf_n${NUM_ENVS}_mb${MINIBATCH_SIZE}_me${MINI_EPOCHS}_lr${LR}`.

---

## Experiment lifecycle (the core algorithm)

Run **at most 2 concurrent jobs**. Each experiment runs for **exactly 3600 s of training time** (measured from t_train_start), then is scancelled. We record `max(successes)` over the budget. Loop forever:

```text
while True:
  # 1. Fill empty slots.
  while len(in_flight) < 2:
    base_sub = current_best_variant.sub
    perturbed_sub = apply_perturbation(base_sub, picked_knobs)
    write to variant_<tag>.sub
    verify_constraints(variant_<tag>.sub)
    git add + git commit -m "autoresearch-nf: <one-line desc>"
    JOBID = sbatch variant_<tag>.sub
    in_flight.append({jobid, commit, sub_path, run_dir, sbatch_time, desc})

  # 2. Poll every 60–1800s.
  sleep N
  for exp in in_flight[:]:
    state = check_state(exp)
    if state == "training_started":
      exp.t_train_start = first_tfevents_walltime(exp.run_dir)
    elif state == "budget_reached":   # (now - t_train_start) ≥ 3600s
      exp.max_succ, exp.last_elapsed = max_succ_within_budget(exp.run_dir, 3600)
      scancel exp.jobid
      keep = exp.max_succ > best_max_so_far
      append_tsv(exp, max_succ=exp.max_succ, last_elapsed=exp.last_elapsed,
                 status="keep" if keep else "discard")
      if not keep: git revert <exp.commit> --no-edit
      in_flight.remove(exp)
    elif state == "crashed":          # squeue empty AND slurm.err has traceback
      append_tsv(exp, max_succ=0, last_elapsed=0, status="crash")
      git revert <exp.commit> --no-edit
      in_flight.remove(exp)
    # else: still running, leave alone
```

Two-slot pipelining is asynchronous: when one slot frees up, fill it from the **current best** commit (not from whatever the other slot is doing).

### Picking perturbations

Greedy local search — perturb one or two knobs at a time from the current best. Aggressive early, refine once near a local optimum.

A few sane prior expectations from the user (treat as hypotheses):
- Higher `mini_epochs` may give more gradient reuse per rollout (good if data is bottleneck) but risks fitting to off-policy data.
- `learning_rate` interacts with all of the above — tune it alongside others.
- Lower `NUM_ENVS` → higher throughput (more epochs in 3600s) but noisier gradients per epoch.
- Higher `MINIBATCH_SIZE` → more stable updates, slower per-step.

When two slots free at once, prefer **two distinct hypotheses** (different knobs or opposite directions on the same knob) over near-duplicates.

---

## Metric extraction — TRAINING time only

Large `NUM_ENVS` (e.g. 4096) spends ~20 min on env init before training starts. **Do not count that against the budget.**

For each in-flight experiment, the run dir is:
```
train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}/
```
i.e. `train_dir/dagger_nonfixed_autoresearch/bc_only_det/<EXPERIMENT_NAME>/`. Tensorboard summaries live at `<run_dir>/0_lpeg_tol0p5mm_sapg_dagger/summaries/events.out.tfevents.*`.

The metric scalar is **`successes`** (mean across envs, written each epoch by `EnvStatsAlgoObserver`). `env_max_goals = 2` for `goal_mode=preInsertAndFinal`, so values fall in [0, 2]. Use this exact scalar key.

```python
# Requires: source .venv_isaacsim/bin/activate
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def max_successes_in_budget(run_dir: str, budget_s: float = 3600.0, key: str = "successes"):
    """Return (t_train_start, max_succ_in_budget, last_elapsed_s).
    - t_train_start: wall_time of first scalar event (= start of training)
    - max_succ_in_budget: max value of `key` across events with wall_time <= t_train_start + budget_s
    - last_elapsed_s: elapsed training-time at the last event seen so far
    Returns (None, None, None) if tfevents not yet written."""
    summ = next(Path(run_dir).rglob("0_*/summaries"), None)
    if summ is None:
        return None, None, None
    ea = EventAccumulator(str(summ), size_guidance={"scalars": 0})
    ea.Reload()
    if key not in ea.Tags()["scalars"]:
        return None, None, None
    events = ea.Scalars(key)
    if not events:
        return None, None, None
    t0 = events[0].wall_time
    deadline = t0 + budget_s
    in_budget = [e for e in events if e.wall_time <= deadline]
    if not in_budget:
        return t0, None, None
    return t0, max(e.value for e in in_budget), in_budget[-1].wall_time - t0
```

**Budget = 3600 s training time**, measured from `t_train_start` (the first tfevents write). Your polling loop has two phases per experiment:
1. Pre-training: wait for the first tfevents scalar to appear.
2. Post-training: poll until `last_elapsed_s >= 3600`, then scancel and finalize.

---

## Job monitoring commands

```bash
# Queue state.
squeue -j "$JOBID" -h -o '%T'
# Final state.
sacct -j "$JOBID" -X -o State,ExitCode -P -n
# Kill.
scancel "$JOBID"
# Forensics on crash.
tail -100 "$RUN_DIR/slurm.err"
```

A job is "crashed" iff `squeue -j` returns empty AND `sacct -j ... -o State -P -n -X` returns something other than `COMPLETED`.

Resolve the `RUN_DIR` from the tag + datetime:
```
${REPO_ROOT}/train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_TAG}_${DATETIME}
```
Recover by `ls -dt train_dir/dagger_nonfixed_autoresearch/bc_only_det/${EXPERIMENT_TAG}_*` and taking the first match.

---

## `results.tsv` schema

Tab-separated, header row already written by Setup. One row per experiment, appended atomically (single `printf >> $TSV`). Columns:

| # | Column | Notes |
|---|--------|-------|
| 1 | `timestamp` | ISO 8601 — `date -Iseconds` |
| 2 | `commit` | 7-char hash of the experiment's commit on `autoresearch/dagger-hpopt` |
| 3 | `experiment_tag` | exactly the `EXPERIMENT_TAG` you put in the sub |
| 4 | `num_envs` | |
| 5 | `minibatch_size` | |
| 6 | `mini_epochs` | |
| 7 | `learning_rate` | as written, e.g. `3e-4` |
| 8 | `expl_block_size` | scales with `num_envs`, recorded for transparency |
| 9 | `max_successes` | float in [0, 2], max value within 3600 s training-time budget |
| 10 | `last_elapsed_s` | training-time at the last logged event (≤ 3600 + epoch-period; useful for sanity) |
| 11 | `status` | `keep` \| `discard` \| `crash` |
| 12 | `description` | one line, no tabs/newlines — what you changed and why |

A `keep` advances the "current best" pointer (highest `max_successes` across all `keep` rows). A `discard` / `crash` row stays in the TSV (for history) but the commit is `git revert`ed.

---

## Sanity checklist before EVERY sbatch

- [ ] `(NUM_ENVS * 16) % MINIBATCH_SIZE == 0`.
- [ ] `NUM_ENVS % EXPL_BLOCK == 0`.
- [ ] `#SBATCH --mem` appropriate for `NUM_ENVS` (≥ ~100 GB per 512 envs; scale linearly).
- [ ] `agent.params.config.lr_schedule=identity` line is present.
- [ ] `USE_OBS_DELAY=True`, `USE_ACTION_DELAY=True`, `USE_CAMERA_DELAY=True`; `OBS_DELAY_MAX=3`, `ACTION_DELAY_MAX=3`, `CAMERA_DELAY_MAX=3`.
- [ ] All reset-noise vars unchanged from baseline:
      `RESET_POSITION_NOISE_X=0.1`, `Y=0.1`, `Z=0.02`,
      `RESET_DOF_POS_NOISE_ARM=0.1`, `FINGERS=0.1`,
      `RESET_DOF_VEL_NOISE=0.5`, `TABLE_RESET_Z_RANGE=0.01`.
- [ ] `LAMBDA_D_START=1.0`, `LAMBDA_D_FLOOR=1.0` (pure BC).
- [ ] `WANDB_PROJECT="dagger_nonfixed_autoresearch"`, `WANDB_GROUP="bc_only_det"`.
- [ ] `EXPERIMENT_TAG` is unique vs. every prior row in `results.tsv`.

---

## Autonomy clause

The human might be asleep. Once you start the loop:
- **Do not** ask for human approval on individual experiment ideas.
- **Do not** stop when you "run out of ideas" — look at the TSV for patterns, try an under-probed region.
- **Do not** stop on a crash — diagnose, fix the divisibility / memory issue, try again.
- **Cancel runs as soon as they reach budget** — the slurm `-t` is long; without `scancel` the GPU stays occupied.
- **Do** stop only on an external interrupt (Ctrl+C, the user explicitly telling you to halt, or a hard infrastructure failure you can't recover from).

The loop runs until the human interrupts you, period.
