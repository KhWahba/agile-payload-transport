# Prompt for Next Session

Continue from branch: `feature/standalone-crocoddyl-mujoco-ocp` in submodule `deps/agile-payload-transport`.

Context:
- A standalone Crocoddyl+MuJoCo OCP prototype exists and runs.
- It is intentionally independent from existing NMPC/controller code paths.
- Current files:
  - `standalone/standalone_mj_croc_ocp.cpp`
  - `scripts/run_standalone_zerogoal_ocp.sh`
  - `standalone/check_and_plot_standalone.py`
  - `standalone/HANDOFF_STANDALONE_OCP.md`

Goals for this session (implement):
1. Integrate pc-dbCBS trajectory reference into standalone OCP for `mujocoquadspayload_zerogoal.yaml`:
   - Use it as warm-start and as tracking reference.
   - Add clear CLI options for reference YAML path and mode switches.

2. Add structured costs comparable to NMPC formulation:
   - control limits / penalties
   - collision avoidance penalty
   - state bounds handling
   - state regularization
   - trajectory tracking (replace fixed-goal-only behavior)

3. Build a comparison harness between:
   - current NMPC implementation
   - standalone OCP implementation
   with same tuning knobs where possible.

4. Produce comparable artifacts:
   - numerical report table (JSON/CSV)
   - trajectory plots
   - MuJoCo videos
   - consistency checker outputs

Constraints:
- Do not regress standalone run reproducibility.
- Keep new logic organized (separate reference loader, cost config, comparison runner).
- Keep generated artifacts in `runs/` and avoid committing them.

Validation criteria:
- End-to-end command(s) reproducibly generate both systems' results.
- Comparison report includes final error, iterations/status, runtime.
- Visual outputs generated for both approaches.

Start by reading:
- `standalone/HANDOFF_STANDALONE_OCP.md`
- `standalone/standalone_mj_croc_ocp.cpp`
