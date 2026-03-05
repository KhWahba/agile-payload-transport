# Standalone OCP (Crocoddyl + MuJoCo)

This demo implements a payload transport OCP without using the project NMPC/controller code.
It uses only:
- MuJoCo XML dynamics (`mj_step`)
- Crocoddyl shooting + FDDP solver

## Build

```bash
cmake --build build -j --target standalone_mj_croc_ocp
```

## Run zerogoal test cases

```bash
scripts/run_standalone_zerogoal_ocp.sh
```

This runs:
1. `start -> goal hover` from `mujocoquadspayload_zerogoal.yaml`
2. `hover hold` (start from goal)

Outputs are written to `runs/standalone_ocp/*.yaml`.
Checker/plots are written under:
- `runs/standalone_ocp/start_to_goal_check/`
- `runs/standalone_ocp/hover_hold_check/`

Notes:
- State convention in this standalone tool is quaternion `xyzw` in YAML/state vectors.
- Before `mj_step`, it converts to MuJoCo `wxyz` in `qpos`, and converts back after stepping.
- Orientation state cost uses a quaternion manifold error vector (3D), not raw quaternion subtraction.
