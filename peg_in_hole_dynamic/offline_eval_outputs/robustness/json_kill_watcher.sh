#!/bin/bash
# Watcher: for each active offline_eval_robustness.py process, when its
# --output-json path becomes a non-empty file, force-kill the python so the
# orchestrator can move on (Isaac Sim's teardown otherwise spins for ~5 min).
#
# Polls every 4s. Logs to robustness/json_kill_watcher.log.

OUT_DIR="/share/portal/kk837/depthbasedRL/peg_in_hole_dynamic/offline_eval_outputs/robustness"
SEEN_PIDS=()
already_killed() {
    local pid="$1"
    for p in "${SEEN_PIDS[@]}"; do [ "$p" = "$pid" ] && return 0; done
    return 1
}

while true; do
    while read -r pid cmdline; do
        [ -z "$pid" ] && continue
        already_killed "$pid" && continue
        # Extract --output-json path from the cmdline
        jsonpath=$(echo "$cmdline" | sed -nE 's/.*--output-json[ =]([^ ]+).*/\1/p')
        [ -z "$jsonpath" ] && continue
        # Has the python written its JSON yet?
        if [ -s "$jsonpath" ]; then
            echo "$(date +%H:%M:%S) PID=$pid wrote $(basename $jsonpath); SIGKILLing"
            kill -9 "$pid" 2>/dev/null
            SEEN_PIDS+=("$pid")
        fi
    done < <(pgrep -af "offline_eval_robustness.py" 2>/dev/null | awk '{pid=$1; $1=""; print pid, $0}')
    sleep 4
done
