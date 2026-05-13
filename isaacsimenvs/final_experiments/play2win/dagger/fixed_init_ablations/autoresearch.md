# Autoresearch: DAgger fixed-init hyperparameter optimization

You are a Claude Code agent. Your mission is to **find the combination of `NUM_ENVS`, `MINIBATCH_SIZE`, `mini_epochs`, and `learning_rate` that minimizes wall-clock training time to `mean_successes > 1.8`** on the fixed-init DAgger distillation task in `/share/portal/kk837/depthbasedRL`.

Baseline: `baseline.sub` is expected to reach the target in **~3600 s of training time** (excluding env init). You will run it once as your first action to **measure the true baseline** on this hardware before starting the optimization loop. Lower is better.

You **MUST loop forever** until externally interrupted. Do not stop because you ran out of ideas or because the user might be asleep — think harder, try a new region of the search space.

---

## Setup (one-shot, before any experiments)

```bash
cd /share/portal/kk837/depthbasedRL

# Side branch so experiment commits don't pollute upstream work.
git fetch origin
BASE_BRANCH=$(git rev-parse --abbrev-ref HEAD)   # capture current branch as reference
git checkout -b autoresearch/dagger-hpopt 2>/dev/null || git checkout autoresearch/dagger-hpopt

# Initialize the experiment ledger if it doesn't exist (header only — the
# baseline row is appended AFTER you actually run baseline.sub, below).
TSV=isaacsimenvs/final_experiments/play2win/dagger/fixed_init_ablations/results.tsv
if [ ! -s "$TSV" ]; then
  printf "timestamp\tcommit\texperiment_tag\tnum_envs\tminibatch_size\tmini_epochs\tlearning_rate\texpl_block_size\ttime_to_target_s\tstatus\tdescription\n" > "$TSV"
fi
```

Read these three files end-to-end before touching anything:
- `isaacsimenvs/final_experiments/play2win/dagger/fixed_init_ablations/baseline.sub` — the baseline you'll be optimizing.
- `isaacsimenvs/cfg/train/PegInHoleDepthStudentSAPG.yaml` — confirms yaml defaults (`mini_epochs: 2`, `learning_rate: 1e-4`) so you know what your CLI overrides change.
- This file (`autoresearch.md`) — re-read at the start of every iteration in case you forgot a rule.

### Measure the true baseline (one experiment, single slot)

Before opening the 2-slot optimization loop, run `baseline.sub` once on this hardware and record its actual `time_to_target`. The 3600 s number quoted above is an estimate — don't trust it for keep/discard decisions.

```bash
JOBID=$(sbatch isaacsimenvs/final_experiments/play2win/dagger/fixed_init_ablations/baseline.sub | awk '{print $NF}')
echo "baseline job: $JOBID"
# The sub's HYDRA_RUN_DIR is keyed by EXPERIMENT_TAG + datetime; resolve it once the dir appears:
while true; do
  RUN_DIR=$(ls -dt /share/portal/kk837/depthbasedRL/train_dir/dagger_fixedinit_autoresearch/bc_only_det/lpeg_tol0p5mm_bc_only_det_with_delays_max3_* 2>/dev/null | head -1)
  [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] && break
  sleep 30
done
echo "baseline run dir: $RUN_DIR"
```

Then poll via the `time_to_target` Python snippet below (every 60 s) until it returns a finite value, the 5400 s training-time timeout fires, or the job crashes. Once you have the baseline number:

```bash
# Use the commit hash currently at HEAD on autoresearch/dagger-hpopt
# (or "-------" if it equals upstream HEAD without any local commits).
BASELINE_COMMIT=$(git rev-parse --short HEAD)
printf "%s\t%s\tbaseline\t512\t2048\t2\t1e-4\t128\t%s\tkeep\tmeasured baseline on this hardware\n" \
  "$(date -Iseconds)" "$BASELINE_COMMIT" "$TIME_TO_TARGET_S" >> "$TSV"
```

This row's `time_to_target_s` becomes the **initial "current best"** that every subsequent experiment is compared against. Only after this row is written do you enter the 2-slot loop. If the baseline run fails (timeout, crash, OOM), **halt and report** — something is broken at the infrastructure level before any optimization can be meaningful.

---

## Scope of allowed edits

**You may ONLY edit files matching the glob**
`isaacsimenvs/final_experiments/play2win/dagger/fixed_init_ablations/distill_*.sub`.

You may create new variant subs in the same directory. You may not touch:
- any `.py` file, anywhere in the repo;
- any `.yaml` file, anywhere in the repo;
- `.gitignore`, `pyproject.toml`, the `rl_games/` or `isaaclab/` submodules, etc.;
- subs outside `fixed_init_ablations/` (the parent `dagger/` folder is off-limits).

If a configuration knob you'd like to tune isn't exposed as a `agent.params.config.*` / `env.*` override in the baseline sub, **stop and add it via the sub's CLI override block** — never edit the yaml directly.

---

## The four tunable knobs

| Knob | Bash var in sub | Hydra override line | Yaml default | Baseline value |
|------|-----------------|---------------------|--------------|----------------|
| Envs per rollout | `NUM_ENVS` | `env.scene.num_envs=$NUM_ENVS` | — | 512 |
| Minibatch size | `MINIBATCH_SIZE` | `agent.params.config.minibatch_size=$MINIBATCH_SIZE` *and* `agent.params.config.central_value_config.minibatch_size=$MINIBATCH_SIZE` | 16384 | 2048 |
| Mini-epochs | `MINI_EPOCHS` | `agent.params.config.mini_epochs=$MINI_EPOCHS` | 2 | **4** (overridden — empirically ~2x faster wall-clock convergence than yaml's PPO-tuned default of 2) |
| Learning rate | `LR` | `agent.params.config.learning_rate=$LR` | 1e-4 | 1e-4 |

All four are pre-exposed as bash variables at the top of `baseline.sub` with their corresponding `agent.params.config.*` CLI lines wired up. To tune one, edit the bash var. Do **not** add a fifth CLI override line — every knob the autoresearch loop should touch is already plumbed.

---

## Hard constraints — verify before EVERY `sbatch`

1. `NUM_ENVS % EXPL_COEF_BLOCK_SIZE == 0`. The baseline uses 512 / 128 = 4 SAPG blocks. If you change `NUM_ENVS`, scale `EXPL_BLOCK` to keep ~4 blocks (or any divisor of `NUM_ENVS`).
2. `(NUM_ENVS * 16) % MINIBATCH_SIZE == 0` — horizon is 16, total batch = `NUM_ENVS × 16`, must be cleanly divisible by minibatch.
3. `#SBATCH --mem` scales roughly linearly with `NUM_ENVS`. Reference points: 120 GB @ 512 envs; 400 GB @ 4096 envs. Bump it before submitting larger runs.
4. The following overrides MUST remain in every variant (else the run is invalid — discard immediately if you forgot):
   - `agent.params.config.lr_schedule=identity` (adaptive scheduler crushes LR to 1e-6 for BC; this fix is non-negotiable);
   - `USE_OBS_DELAY=True`, `USE_ACTION_DELAY=True`, `USE_CAMERA_DELAY=True` (all max=3) — delays are part of the testbed, not under tuning.
   - `WANDB_PROJECT=dagger_fixedinit_autoresearch`, `WANDB_GROUP=bc_only_det` (so all variants plot together).
5. `EXPERIMENT_TAG` must be unique vs. every prior row in `results.tsv`. Suggested format: `lpeg_v_n${NUM_ENVS}_mb${MINIBATCH_SIZE}_me${MINI_EPOCHS}_lr${LR}` then suffix with a short hash if you've used that combo before.

---

## Experiment lifecycle (the core algorithm)

Run **at most 2 concurrent jobs**. Loop forever:

```text
while True:
  # 1. Fill empty slots.
  while len(in_flight) < 2:
    base_sub = git_show(best_commit_so_far,
                        "isaacsimenvs/.../fixed_init_ablations/baseline.sub")
    perturbed_sub = apply_perturbation(base_sub, picked_knobs)   # see "Picking perturbations"
    write to variant_<timestamp>.sub
    verify_constraints(variant_<timestamp>.sub)                  # hard assertions
    git add + git commit -m "autoresearch: <one-line desc>"
    JOBID = sbatch variant_<timestamp>.sub
    in_flight.append({jobid, commit, sub_path, run_dir, sbatch_time, desc})

  # 2. Poll every 60s.
  sleep 60
  for exp in in_flight[:]:
    state = check_state(exp)
    if state == "training_started":
      exp.t_train_start = first_tfevents_walltime(exp.run_dir)
    elif state == "target_hit":
      exp.time_to_target = t_target_walltime - exp.t_train_start
      scancel exp.jobid
      keep = exp.time_to_target < best_time_so_far
      append_tsv(exp, time=exp.time_to_target, status="keep" if keep else "discard")
      if not keep: git revert <exp.commit> --no-edit
      in_flight.remove(exp)
    elif state == "timeout":               # > 5400s training time
      scancel exp.jobid
      append_tsv(exp, time=now-t_train_start, status="timeout")
      git revert <exp.commit> --no-edit
      in_flight.remove(exp)
    elif state == "crashed":               # squeue empty AND slurm.err has a traceback
      append_tsv(exp, time=0, status="crash")
      git revert <exp.commit> --no-edit
      in_flight.remove(exp)
    # else: still running, leave alone
```

Two-slot pipelining is asynchronous: you finish one slot, immediately fill it from the **current best** commit (not from whatever the other slot is doing). The other slot may have been launched off an earlier best — that's fine.

### Picking perturbations

You decide. Greedy local search is the simplest model — perturb one or two knobs at a time from the current best. You're encouraged to explore aggressively at first (orders-of-magnitude jumps on `learning_rate`, `NUM_ENVS`) and refine once you're near a local optimum.

A few sane prior expectations from the user (treat as hypotheses to test, not gospel):
- Lower `NUM_ENVS` → higher throughput (faster iterations) but noisier gradients.
- Higher `MINIBATCH_SIZE` → more stable updates, slower per-step.
- More `mini_epochs` → more gradient reuse per rollout (good if data is the bottleneck) but risks fitting to off-policy data.
- `learning_rate` interacts with all of the above — don't tune it last.

When two slots free up at the same time, prefer to pick **two distinct hypotheses** (different knobs or opposite directions on the same knob) rather than running two near-duplicates.

---

## Metric extraction — TRAINING time, not slurm wall time

Large `NUM_ENVS` (e.g. 4096) spends ~20 min on env init before training starts. **Do not count that against the run.**

For each in-flight experiment, the run dir is:
```
train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}/
```
i.e. for our settings: `train_dir/dagger_fixedinit_autoresearch/bc_only_det/<EXPERIMENT_NAME>/`. The tensorboard summaries are under `<run_dir>/0_lpeg_tol0p5mm_sapg_dagger/summaries/events.out.tfevents.*`.

The metric scalar is **`successes`** (mean across envs, written each epoch by `EnvStatsAlgoObserver`). `env_max_goals = 2` for `goal_mode=preInsertAndFinal`, so a `successes` value of `1.8` corresponds to 90% success rate. Use this exact scalar key — `successes_max` and `successes_median` are different summaries.

Use this snippet (Python, via the project venv):

```python
# Requires: source .venv_isaacsim/bin/activate
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def time_to_target(run_dir: str, threshold: float = 1.8, key: str = "successes"):
    """Return (t_train_start, t_target, time_to_target_s) or (t_train_start, None, None)
    if target not yet hit. Returns (None, None, None) if tfevents not yet written."""
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
    t_train_start = events[0].wall_time
    for ev in events:
        if ev.value > threshold:
            return t_train_start, ev.wall_time, ev.wall_time - t_train_start
    return t_train_start, None, None
```

`wall_time` is float-seconds since epoch — directly usable for time deltas.

**Training-time timeout: 5400 s (1.5 h)**, measured from `t_train_start` (the first tfevents write), **not** from sbatch submission. Your polling loop has two phases per experiment:
1. Pre-training: wait for the first tfevents scalar to appear (anywhere from ~30 s @ 512 envs to ~20 min @ 4096 envs). Detect by `time_to_target(run_dir)` returning a non-None `t_train_start`.
2. Post-training: start counting against 5400 s.

---

## Job monitoring commands

```bash
# Is the job still on the queue (R / PD / CG)?
squeue -j "$JOBID" -h -o '%T'

# Final state after the job leaves the queue.
sacct -j "$JOBID" -X -o State,ExitCode -P -n

# Kill it.
scancel "$JOBID"

# Crash forensics.
tail -100 "$RUN_DIR/slurm.err"
```

A job is "crashed" iff `squeue -j` returns empty AND `sacct -j ... -o State -P -n -X` returns something other than `COMPLETED` (i.e. `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `CANCELLED`).

Resolve the `RUN_DIR` for an experiment from its sub file: the sub computes `HYDRA_RUN_DIR` exactly as
```
${REPO_ROOT}/train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_TAG}_${DATETIME}
```
The `DATETIME` is recorded into `slurm.log` on the first line of stdout; you can also recover it by `ls -dt train_dir/dagger_fixedinit_autoresearch/bc_only_det/${EXPERIMENT_TAG}_*` and taking the first match.

---

## `results.tsv` schema

Tab-separated, header row already written by Setup. One row per experiment, appended atomically (single `printf >> $TSV`). Columns:

| # | Column | Notes |
|---|--------|-------|
| 1 | `timestamp` | ISO 8601 — `date -Iseconds` |
| 2 | `commit` | 7-char hash of the experiment's commit on `autoresearch/dagger-hpopt` (`-------` for the upstream baseline row) |
| 3 | `experiment_tag` | exactly the `EXPERIMENT_TAG` you put in the sub |
| 4 | `num_envs` | |
| 5 | `minibatch_size` | |
| 6 | `mini_epochs` | |
| 7 | `learning_rate` | as written, e.g. `3e-4` |
| 8 | `expl_block_size` | for transparency since it scales with `num_envs` |
| 9 | `time_to_target_s` | float, or 0 for non-completions |
| 10 | `status` | `keep` \| `discard` \| `timeout` \| `crash` |
| 11 | `description` | one line, no tabs/newlines — what you changed and why |

A `keep` advances the "current best" pointer (lowest `time_to_target_s` across all `keep` rows). A `discard` / `timeout` / `crash` row stays in the TSV (for history) but the commit is `git revert`ed.

---

## Sanity checklist before EVERY sbatch

Run through every item. If any fails, scrap the variant and pick a different perturbation.

- [ ] `(NUM_ENVS * 16) % MINIBATCH_SIZE == 0`.
- [ ] `NUM_ENVS % EXPL_BLOCK == 0`.
- [ ] `#SBATCH --mem` is appropriate for `NUM_ENVS` (≥ ~120 GB per 512 envs).
- [ ] `agent.params.config.lr_schedule=identity` line is present.
- [ ] `USE_OBS_DELAY=True`, `USE_ACTION_DELAY=True`, `USE_CAMERA_DELAY=True` all set; `OBS_DELAY_MAX=3`, `ACTION_DELAY_MAX=3`, `CAMERA_DELAY_MAX=3`.
- [ ] `FIXED_START_POSE`, `HOLE_X_RANGE`, `HOLE_Y_RANGE` unchanged from baseline.
- [ ] All reset-noise vars (`RESET_POSITION_NOISE_*`, `RESET_DOF_POS_NOISE_*`, `RESET_DOF_VEL_NOISE`, `TABLE_RESET_Z_RANGE`) = 0.0.
- [ ] `LAMBDA_D_START=1.0`, `LAMBDA_D_FLOOR=1.0` (pure BC).
- [ ] `WANDB_PROJECT="dagger_fixedinit_autoresearch"`, `WANDB_GROUP="bc_only_det"`.
- [ ] `EXPERIMENT_TAG` is unique vs. every prior row in `results.tsv`.

---

## Autonomy clause

The human might be asleep. Once you start the loop:
- **Do not** ask for human approval on individual experiment ideas.
- **Do not** stop when you "run out of ideas" — think harder, look at the TSV for patterns, try a region of the search space you haven't probed.
- **Do not** stop on a crash — diagnose, fix the divisibility / memory issue, try again.
- **Do** stop only on an external interrupt (Ctrl+C, the user explicitly telling you to halt, or a hard infrastructure failure you can't recover from).

The loop runs until the human interrupts you, period.
