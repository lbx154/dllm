#!/usr/bin/env bash
# =============================================================================
# launch_with_canary.sh — unified launcher: preflight + train + watcher + dashboard
#
# Usage:
#   bash scripts/launch_with_canary.sh scripts/launch_btgrpo_run20.sh
#
# What it does, in order:
#   1. Tier-0 preflight on the inner launch script. Aborts if blocking checks fail.
#   2. Starts the inner training script in the background, writing to .logs/.
#   3. Starts canary_watcher.py against the run log, with --target-pid pointing
#      at the training process (so SIGINT propagates on Tier-1 abort).
#   4. Starts dashboard.py against the same log (refresh 30s -> dashboard.png).
#   5. Traps Ctrl-C / EXIT and shuts everything down cleanly.
#
# Outputs:
#   .logs/<runtag>.log           — trainer stdout/stderr (the dict log)
#   .logs/<runtag>.canary.log    — watcher signature lights per step
#   .logs/<runtag>.dashboard.log — dashboard.py stdout
#   dashboard.png                — refreshed every 30s in repo root
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}"

INNER="${1:-}"
if [[ -z "${INNER}" ]] || [[ ! -f "${INNER}" ]]; then
    echo "usage: $0 <path/to/launch_btgrpo_runXX.sh>" >&2
    exit 64
fi

# Derive a runtag from the inner script filename (e.g. launch_btgrpo_run20.sh -> run20)
RUNTAG="$(basename "${INNER}" .sh | sed -E 's/^launch_btgrpo_//')"
LOG_DIR="${REPO_ROOT}/.logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/btgrpo-${RUNTAG}.log"
CANARY_LOG="${LOG_DIR}/btgrpo-${RUNTAG}.canary.log"
DASH_LOG="${LOG_DIR}/btgrpo-${RUNTAG}.dashboard.log"
DASH_PNG="${REPO_ROOT}/dashboard.png"

GREEN="\033[92m"; RED="\033[91m"; YEL="\033[93m"; RST="\033[0m"

# ----- 1. preflight ----------------------------------------------------------
echo -e "${GREEN}[1/4] Tier-0 preflight on ${INNER}${RST}"
if ! python "${REPO_ROOT}/scripts/canary_preflight.py" --launch-script "${INNER}"; then
    echo -e "${RED}preflight failed — aborting launch.${RST}" >&2
    exit 1
fi

# ----- 2. start training -----------------------------------------------------
echo -e "${GREEN}[2/4] starting training -> ${TRAIN_LOG}${RST}"
# Make sure the inner script writes its log to the file we expect. The existing
# run scripts do `>> "${LOG}"`, with LOG derived from the script name, so this
# matches up automatically as long as the inner script uses the same naming.
bash "${INNER}" &
TRAIN_PID=$!
echo "  trainer pid: ${TRAIN_PID}"

# Wait until the log file actually exists before starting the watcher, so the
# watcher doesn't block on a missing file forever.
for i in $(seq 1 60); do
    [[ -f "${TRAIN_LOG}" ]] && break
    sleep 1
done
if [[ ! -f "${TRAIN_LOG}" ]]; then
    echo -e "${RED}training log ${TRAIN_LOG} not appearing after 60s — check inner script's LOG path.${RST}" >&2
    kill -INT "${TRAIN_PID}" 2>/dev/null || true
    exit 1
fi

# ----- 3. start watcher ------------------------------------------------------
echo -e "${GREEN}[3/4] starting canary watcher -> ${CANARY_LOG}${RST}"
nohup python "${REPO_ROOT}/scripts/canary_watcher.py" \
    --log "${TRAIN_LOG}" \
    --target-pid "${TRAIN_PID}" \
    --refresh 2 \
    > "${CANARY_LOG}" 2>&1 &
WATCHER_PID=$!
echo "  watcher pid: ${WATCHER_PID}"

# ----- 4. start dashboard ----------------------------------------------------
echo -e "${GREEN}[4/4] starting dashboard -> ${DASH_PNG} (refresh 30s)${RST}"
nohup python "${REPO_ROOT}/dashboard.py" \
    --log "${TRAIN_LOG}" \
    --refresh 30 \
    > "${DASH_LOG}" 2>&1 &
DASH_PID=$!
echo "  dashboard pid: ${DASH_PID}"

# ----- cleanup ---------------------------------------------------------------
cleanup() {
    echo -e "\n${YEL}cleaning up watcher (${WATCHER_PID}) and dashboard (${DASH_PID})...${RST}"
    kill "${WATCHER_PID}" 2>/dev/null || true
    kill "${DASH_PID}"    2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ----- monitor ---------------------------------------------------------------
echo
echo -e "  train log:     ${TRAIN_LOG}"
echo -e "  watcher log:   ${CANARY_LOG}    (last lines = signature lights)"
echo -e "  dashboard:     ${DASH_PNG}      (open in viewer)"
echo
echo -e "${GREEN}All started. Tailing watcher output. Ctrl-C to stop everything.${RST}"
echo

# Print a one-liner whenever the watcher updates, plus die when training dies.
( tail -F "${CANARY_LOG}" 2>/dev/null ) &
TAIL_PID=$!
trap 'cleanup; kill ${TAIL_PID} 2>/dev/null || true' EXIT INT TERM

# Block on the trainer; when it exits (normally OR via canary SIGINT), we exit too.
wait "${TRAIN_PID}"
TRAIN_EXIT=$?
echo -e "\ntrainer exited with code ${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
