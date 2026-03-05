#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/build/standalone_mj_croc_ocp"
CHECKER="$REPO_ROOT/standalone/check_and_plot_standalone.py"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/../.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
PROBLEM_YAML="$REPO_ROOT/../pc-dbCBS/deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_zerogoal.yaml"
XML_FILE="$REPO_ROOT/../pc-dbCBS/deps/dynoplan/dynobench/models/xml/2cfs_payload_tendons_empty.xml"
OUT_DIR="$REPO_ROOT/runs/standalone_ocp"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found: $BIN"
  echo "Build first with:"
  echo "  cmake --build $REPO_ROOT/build -j --target standalone_mj_croc_ocp"
  exit 1
fi

echo "[standalone] Case A: start -> goal hover"
set +e
"$BIN" \
  --problem-yaml "$PROBLEM_YAML" \
  --xml-file "$XML_FILE" \
  --horizon 220 \
  --max-iter 80 \
  --out-yaml "$OUT_DIR/zerogoal_start_to_goal.yaml"
RC_A=$?

echo "[standalone] Case B: hover hold (start from goal)"
"$BIN" \
  --problem-yaml "$PROBLEM_YAML" \
  --xml-file "$XML_FILE" \
  --horizon 220 \
  --max-iter 80 \
  --start-from-goal \
  --out-yaml "$OUT_DIR/zerogoal_hover_hold.yaml"
RC_B=$?
set -e

echo "[standalone] Checking + plotting Case A"
"$PYTHON_BIN" "$CHECKER" \
  --solution-yaml "$OUT_DIR/zerogoal_start_to_goal.yaml" \
  --problem-yaml "$PROBLEM_YAML" \
  --xml-file "$XML_FILE" \
  --out-dir "$OUT_DIR/start_to_goal_check"

echo "[standalone] Checking + plotting Case B"
"$PYTHON_BIN" "$CHECKER" \
  --solution-yaml "$OUT_DIR/zerogoal_hover_hold.yaml" \
  --problem-yaml "$PROBLEM_YAML" \
  --xml-file "$XML_FILE" \
  --out-dir "$OUT_DIR/hover_hold_check"

echo "[standalone] Outputs:"
ls -la "$OUT_DIR"
echo "[standalone] exit codes: caseA=$RC_A caseB=$RC_B (0 means converged by solver criteria)"
