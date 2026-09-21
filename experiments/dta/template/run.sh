#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$REPO_DIR/experiments/dta/template"
LOG_DIR="$EXP_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Fill this template from an existing experiment run.sh before running." | tee "$LOG_FILE"
exit 1
