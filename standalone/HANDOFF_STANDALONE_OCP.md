# Standalone Crocoddyl + MuJoCo OCP Handoff

## Scope and Intent
This branch adds a standalone OCP prototype for payload transport that does **not** use existing NMPC/controller/project model code paths. It uses:
- Crocoddyl shooting + FDDP
- MuJoCo XML dynamics via `mj_step`

Primary file:
- `standalone/standalone_mj_croc_ocp.cpp`

Runner + checker:
- `scripts/run_standalone_zerogoal_ocp.sh`
- `standalone/check_and_plot_standalone.py`

## What Was Implemented
1. Standalone Crocoddyl action model/data classes:
- `MujocoDiscreteActionModel : crocoddyl::ActionModelAbstract`
- `MujocoDiscreteActionData : crocoddyl::ActionDataAbstract`

2. OCP setup:
- Loads start/goal from `mujocoquadspayload_zerogoal.yaml`
- Builds running + terminal models
- Solves with `crocoddyl::SolverFDDP`

3. Quaternion handling:
- State convention in standalone vectors/YAML: `xyzw`
- Converted to MuJoCo `wxyz` before `mj_step`
- Converted back after stepping

4. Orientation cost:
- Quaternion manifold error vector `e_q` used in state cost
- Not raw component subtraction on quaternion entries

5. Initialization policy (explicitly requested):
- `us_init`: hover-like midpoint of actuator control range
- `xs_init`: repeated start state for all horizon nodes (no rollout prefill)

6. Added check/plot tooling:
- Replay consistency check against MuJoCo dynamics
- JSON report + PNG plots saved under `runs/standalone_ocp/*_check/`

## Build and Run
```bash
cmake --build build -j --target standalone_mj_croc_ocp
PYTHON_BIN=../.venv/bin/python scripts/run_standalone_zerogoal_ocp.sh
```

## Current Observed Behavior
- Solver still reports `solved=false` in both default test cases.
- Start->goal still shows meaningful improvement (lower final error than initial).
- Hover-hold case has good final-goal error in saved solution but checker reports large rollout mismatch; this must be debugged before trusting hover-hold conclusions.

## Known Gaps / Risks
1. Derivatives are finite-difference only (no analytical dynamics/cost derivatives).
2. `Lxu` is zeroed (not estimated).
3. Cost set is minimal (goal/terminal + control regularization).
4. No explicit constraints/costs yet for:
- control limits
- collision avoidance
- state bounds
- richer state regularization
- reference tracking from external trajectory

## Requested Next Work (NOT IMPLEMENTED YET)
1. Use pc-dbCBS reference trajectory as warm-start + tracking reference on same zerogoal example.
2. Add structured costs similar to current NMPC code:
- control limits
- collision avoidance
- state bounds
- state regularization
- reference tracking (trajectory, not fixed goal only)
3. Compare standalone OCP outcome vs current NMPC with same tuning.
4. Add visualization/video export and side-by-side plot comparison.

## Suggested Acceptance Gates for Next Phase
1. Deterministic run on zerogoal with fixed params.
2. Report table for both systems:
- final goal error
- payload position error
- solve status/iterations
- runtime stats
3. Matching visualization artifacts:
- trajectory plots
- MuJoCo videos
- checker reports
