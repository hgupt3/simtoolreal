#!/bin/bash
# Drive offline_eval_robustness.py over 5 policies x 2 scale passes (off, +/- 15%).
# Each invocation: 1 env init (~60s) + 8 settings x ~40s rollout = ~6 min.
# Total wall time: ~60-70 min for 10 invocations.
#
# After each Python finishes (json written), we force-kill any orphan
# Isaac Sim shutdown so the GPU is free for the next invocation.

set -uo pipefail

REPO="/share/portal/kk837/depthbasedRL"
TEACHER="${REPO}/train_dir/isaacsimenvs/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_2026-05-11_16-50-01/0_lpeg_tol0p5mm_finetune_rgf0_2026-05-11_16-50-01/best/model.pth"
CKPT_DIR="${REPO}/hardware_rollouts/2026-05-13_camera_noise_checkpoints"
OUT="${REPO}/peg_in_hole_dynamic/offline_eval_outputs/robustness"
mkdir -p "$OUT"

POLICIES=(
    "no_delays_no_camnoise"
    "camrand_off_depthaug_off"
    "camrand_on_depthaug_off"
    "camrand_off_depthaug_on"
    "camrand_on_depthaug_on"
)

# (label, --table-scale-x, --table-scale-y, --table-scale-n)
SCALE_PASSES=(
    "scale_off|1.0,1.0|1.0,1.0|1"
    "scale_15pct|0.85,1.15|0.85,1.15|10"
)

run_one() {
    local policy="$1"
    local pass_label="$2" sx="$3" sy="$4" sn="$5"
    local tag="${policy}__${pass_label}"
    local logf="${OUT}/${tag}.log"
    local jsonf="${OUT}/${tag}.json"
    echo "=== START ${tag} ==="
    "${REPO}/.venv_isaacsim/bin/python" -u \
        "${REPO}/peg_in_hole_dynamic/offline_eval_robustness.py" \
        --teacher-checkpoint "$TEACHER" \
        --student-checkpoint "${CKPT_DIR}/${policy}/model.pth" \
        --policy-name "$policy" \
        --num-envs 10 --seed 42 --max-steps-per-episode 600 \
        --goal-mode finalGoalOnly \
        --table-scale-x "$sx" --table-scale-y "$sy" --table-scale-n "$sn" \
        --output-json "$jsonf" \
        > "$logf" 2>&1
    local code=$?
    echo "=== END ${tag} (exit=$code) ==="
    # Force-cleanup the python in case of slow Isaac Sim teardown.
    pkill -9 -f "offline_eval_robustness.py" 2>/dev/null || true
    sleep 4
}

for policy in "${POLICIES[@]}"; do
    for pass in "${SCALE_PASSES[@]}"; do
        IFS='|' read -r pass_label sx sy sn <<<"$pass"
        run_one "$policy" "$pass_label" "$sx" "$sy" "$sn"
    done
done

echo "=== ALL DONE ==="
ls -la "$OUT"/*.json
