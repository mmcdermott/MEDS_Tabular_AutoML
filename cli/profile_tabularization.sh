#!/usr/bin/env bash

PROFILE_LOG_DIR="$1"

shift 1

SCRIPT_DIR=$(dirname "$0")
SCRIPT_NAME=$(basename "$0")

mkdir -p "$PROFILE_LOG_DIR"
{ time \
    mprof run --include-children --exit-code --output "${PROFILE_LOG_DIR}/mprofile.dat" \
    bash "${SCRIPT_DIR}/tabularization.sh" "$@" \
    2> "${PROFILE_LOG_DIR}/cmd.stderr"
} 2> "${PROFILE_LOG_DIR}/timings.txt"

cmd_exit_status=${PIPESTATUS[0]}
# Check the exit status of the second command in the pipeline (mprof run ...)
if [ -n "$cmd_exit_status" ] && [ "$cmd_exit_status" -ne 0 ]; then
    echo "${SCRIPT_NAME} failed with status $cmd_exit_status."
    echo "Stderr from ${SCRIPT_NAME} (see ${PROFILE_LOG_DIR}/cmd.stderr):"
    tail "${PROFILE_LOG_DIR}/cmd.stderr"
    exit "$cmd_exit_status"
fi
mprof plot -o "${PROFILE_LOG_DIR}/mprofile.png" "${PROFILE_LOG_DIR}/mprofile.dat"
mprof peak "${PROFILE_LOG_DIR}/mprofile.dat" > "${PROFILE_LOG_DIR}/peak_memory_usage.txt"
