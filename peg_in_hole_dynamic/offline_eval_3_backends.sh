#!/bin/bash
# Offline eval of one student checkpoint under a 3x2 grid:
#   {tiled, raycaster, foundation_stereo} x {cy=45, cy=47.13}.
# Each cell gets its own Isaac Sim process so camera_backend + offset
# changes are realised at env init.
#
# cy=45 (cy_match_real=0)        : image-center principal point. Use to
#                                  evaluate checkpoints trained BEFORE the
#                                  cy=47.13 fix landed.
# cy=47.13 (cy_match_real=1)     : matches real ZED HD1080 calibration.
#                                  Use for checkpoints trained AFTER the
#                                  fix, OR to test cy robustness.
#
# Usage:  ./offline_eval_3_backends.sh [output_dir]
set -euo pipefail

REPO=/share/portal/kk837/depthbasedRL
OUT_DIR=${1:-$REPO/peg_in_hole_dynamic/offline_eval_outputs/3_backends_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"

# Override via env vars to evaluate a different checkpoint without editing
# this script:  STUDENT_NAME=camrand_on_depthaug_on bash offline_eval_3_backends.sh
STUDENT_NAME=${STUDENT_NAME:-no_delays_no_camnoise}
STUDENT_CKPT=${STUDENT_CKPT:-$REPO/hardware_rollouts/2026-05-13_camera_noise_checkpoints/$STUDENT_NAME/model.pth}
TEACHER_CKPT=${TEACHER_CKPT:-$REPO/hardware_rollouts/2026-05-06_peg_in_hole_dynamic_checkpoints/peg_tol0p5mm/model.pth}
POLICY_NAME=${POLICY_NAME:-$STUDENT_NAME}

# num_envs=500 gives ~+/-0.02 standard error on the per-cell success rate,
# down from ~+/-0.15 at num_envs=10. Tiled / raycaster scale near-linearly
# with num_envs but stay fast at this scale (~2 min per cell). The Fast-FS
# backend's TRT engine was built with a fixed batch dim of 1, so the wrapper
# loops B times per env step -- a single FS cell at n=500 takes ~20 min vs
# ~2 min at n=10. Either accept the longer runtime, drop --num-envs back to
# something smaller for FS-heavy comparisons, or rebuild the FS engine with
# a dynamic batch dim. Override via env var:  NUM_ENVS=100 bash ...
NUM_ENVS=${NUM_ENVS:-500}
MAX_STEPS=${MAX_STEPS:-600}
SEED=${SEED:-42}
PROBLEM=${PROBLEM:-Lpeg.tol0p5mm}

PY=$REPO/.venv_isaacsim/bin/python
SCRIPT=$REPO/peg_in_hole_dynamic/offline_eval_one_backend.py

echo "==============================================================="
echo "  offline_eval_3_backends  (3 backends x 2 cy options = 6 cells)"
echo "  policy:   $POLICY_NAME"
echo "  problem:  $PROBLEM"
echo "  ckpt:     $STUDENT_CKPT"
echo "  num_envs: $NUM_ENVS   max_steps: $MAX_STEPS   seed: $SEED"
echo "  out_dir:  $OUT_DIR"
echo "==============================================================="

# Columns: backend  cy_match_real  cy_tag  variant_tag  fs_engine_subdir  fs_iters
# variant_tag/engine_subdir/iters only consumed for foundation_stereo cells.
FS_ROOT=$REPO/third_party/Fast-FoundationStereo/weights/23-36-37
declare -a CELLS=(
  "tiled             1   cy47   _      _                       _"
  "tiled             0   cy45   _      _                       _"
  "raycaster         1   cy47   _      _                       _"
  "raycaster         0   cy45   _      _                       _"
  "foundation_stereo 1   cy47   iters4 $FS_ROOT/onnx_384x224_iters4 4"
  "foundation_stereo 0   cy45   iters4 $FS_ROOT/onnx_384x224_iters4 4"
  "foundation_stereo 1   cy47   iters8 $FS_ROOT/onnx_384x224_iters8 8"
  "foundation_stereo 0   cy45   iters8 $FS_ROOT/onnx_384x224_iters8 8"
)

for cell in "${CELLS[@]}"; do
  read -r BACKEND CY_MATCH CY_TAG VARIANT_TAG FS_ENGINE_DIR FS_ITERS <<<"$cell"
  if [ "$BACKEND" = "foundation_stereo" ]; then
    CELL_TAG="${BACKEND}_${VARIANT_TAG}"
  else
    CELL_TAG="$BACKEND"
  fi
  JSON_OUT=$OUT_DIR/${POLICY_NAME}__${CELL_TAG}__${CY_TAG}.json
  LOG_OUT=$OUT_DIR/${POLICY_NAME}__${CELL_TAG}__${CY_TAG}.log
  VIDEO_OUT=$OUT_DIR/${POLICY_NAME}__${CELL_TAG}__${CY_TAG}__policy_depth.mp4
  echo
  echo "[$(date +%H:%M:%S)] === $CELL_TAG  $CY_TAG ==="
  if [ "$BACKEND" = "foundation_stereo" ]; then
    "$PY" "$SCRIPT" \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --student-checkpoint "$STUDENT_CKPT" \
      --policy-name "$POLICY_NAME" \
      --camera-backend "$BACKEND" \
      --cy-match-real "$CY_MATCH" \
      --problem "$PROBLEM" \
      --num-envs "$NUM_ENVS" \
      --max-steps-per-episode "$MAX_STEPS" \
      --seed "$SEED" \
      --fs-engine-dir "$FS_ENGINE_DIR" \
      --fs-valid-iters "$FS_ITERS" \
      --output-json "$JSON_OUT" \
      --video-path "$VIDEO_OUT" \
      --video-fps 60 \
      2>&1 | tee "$LOG_OUT" | grep -E '^=>|mean_succ|FAILED|Error' || true
  else
    "$PY" "$SCRIPT" \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --student-checkpoint "$STUDENT_CKPT" \
      --policy-name "$POLICY_NAME" \
      --camera-backend "$BACKEND" \
      --cy-match-real "$CY_MATCH" \
      --problem "$PROBLEM" \
      --num-envs "$NUM_ENVS" \
      --max-steps-per-episode "$MAX_STEPS" \
      --seed "$SEED" \
      --output-json "$JSON_OUT" \
      --video-path "$VIDEO_OUT" \
      --video-fps 60 \
      2>&1 | tee "$LOG_OUT" | grep -E '^=>|mean_succ|FAILED|Error' || true
  fi
done

echo
echo "==============================================================="
echo "  SUMMARY (mean_succ across $NUM_ENVS envs, $MAX_STEPS-step ep)"
echo "==============================================================="
printf "  %-20s %-10s %-10s %-12s\n" "backend" "cy" "mean_succ" "rollout"
echo "  ----------------------------------------------------------"
for cell in "${CELLS[@]}"; do
  read -r BACKEND CY_MATCH CY_TAG VARIANT_TAG _ _ <<<"$cell"
  if [ "$BACKEND" = "foundation_stereo" ]; then
    CELL_TAG="${BACKEND}_${VARIANT_TAG}"
  else
    CELL_TAG="$BACKEND"
  fi
  JSON_OUT=$OUT_DIR/${POLICY_NAME}__${CELL_TAG}__${CY_TAG}.json
  if [ -f "$JSON_OUT" ]; then
    MEAN=$("$PY" -c "import json; print(f'{json.load(open(\"$JSON_OUT\"))[\"mean_succ\"]:.4f}')" 2>/dev/null || echo 'N/A')
    SECS=$("$PY" -c "import json; print(f'{json.load(open(\"$JSON_OUT\"))[\"rollout_seconds\"]:.1f}s')" 2>/dev/null || echo 'N/A')
    printf "  %-22s %-6s %-10s %-12s\n" "$CELL_TAG" "$CY_TAG" "$MEAN" "$SECS"
  else
    printf "  %-22s %-6s (FAILED)\n" "$CELL_TAG" "$CY_TAG"
  fi
done
echo "==============================================================="
echo "Outputs under: $OUT_DIR"
