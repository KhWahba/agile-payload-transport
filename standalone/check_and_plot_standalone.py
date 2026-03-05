#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
try:
    import mujoco  # type: ignore
except Exception:
    mujoco = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def has_payload_layout(nq: int, nv: int) -> bool:
    return nq > 0 and nv > 0 and nq % 7 == 0 and nv % 6 == 0 and (nq // 7) == (nv // 6)


def xyzw_to_mj_wxyz(state: np.ndarray, nq: int, nv: int) -> np.ndarray:
    s = state.copy()
    if has_payload_layout(nq, nv):
        n_bodies = nq // 7
        for b in range(n_bodies):
            qbase = 7 * b + 3
            qx, qy, qz, qw = s[qbase:qbase + 4]
            s[qbase:qbase + 4] = np.array([qw, qx, qy, qz], dtype=float)
    return s


def mj_wxyz_to_xyzw(state: np.ndarray, nq: int, nv: int) -> np.ndarray:
    s = state.copy()
    if has_payload_layout(nq, nv):
        n_bodies = nq // 7
        for b in range(n_bodies):
            qbase = 7 * b + 3
            qw, qx, qy, qz = s[qbase:qbase + 4]
            s[qbase:qbase + 4] = np.array([qx, qy, qz, qw], dtype=float)
    return s


def payload_pos(xs: np.ndarray) -> np.ndarray:
    return xs[:, 0:3]


def main() -> int:
    ap = argparse.ArgumentParser(description="Check and visualize standalone OCP solution")
    ap.add_argument("--solution-yaml", required=True)
    ap.add_argument("--problem-yaml", required=True)
    ap.add_argument("--xml-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sol = yaml.safe_load(Path(args.solution_yaml).read_text())
    prob = yaml.safe_load(Path(args.problem_yaml).read_text())
    start = np.array(prob["joint_robot"][0]["start"], dtype=float)
    goal = np.array(prob["joint_robot"][0]["goal"], dtype=float)
    xs = np.array(sol["xs"], dtype=float)
    us = np.array(sol["us"], dtype=float)

    step_mismatch = np.array([], dtype=float)
    if mujoco is not None:
        model = mujoco.MjModel.from_xml_path(str(Path(args.xml_file).resolve()))
        data = mujoco.MjData(model)
        nq, nv = model.nq, model.nv

        # Replay controls from x0 and compare to saved xs.
        replay_xs = [xs[0].copy()]
        for k in range(len(us)):
            xk = replay_xs[-1].copy()
            xk_mj = xyzw_to_mj_wxyz(xk, nq, nv)
            data.qpos[:] = xk_mj[:nq]
            data.qvel[:] = xk_mj[nq:nq + nv]
            data.ctrl[:] = us[k]
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            xn_mj = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            xn = mj_wxyz_to_xyzw(xn_mj, nq, nv)
            replay_xs.append(xn)
        replay_xs = np.asarray(replay_xs)

        if replay_xs.shape != xs.shape:
            raise RuntimeError(f"Replay shape mismatch: replay={replay_xs.shape}, sol={xs.shape}")

        step_mismatch = np.linalg.norm(replay_xs - xs, axis=1)
    p = payload_pos(xs)
    p_goal = goal[:3]
    p_err = np.linalg.norm(p - p_goal[None, :], axis=1)
    u_norm = np.linalg.norm(us, axis=1) if len(us) else np.array([])

    report = {
        "solved_flag": bool(sol.get("solved", False)),
        "iter": int(sol.get("iter", -1)),
        "cost": float(sol.get("cost", float("nan"))),
        "horizon_steps": int(len(us)),
        "initial_goal_error_l2": float(np.linalg.norm(xs[0] - goal)),
        "final_goal_error_l2": float(np.linalg.norm(xs[-1] - goal)),
        "initial_payload_pos_error": float(p_err[0]),
        "final_payload_pos_error": float(p_err[-1]),
        "mujoco_python_available": bool(mujoco is not None),
        "max_step_rollout_mismatch": float(np.max(step_mismatch)) if len(step_mismatch) else None,
        "mean_step_rollout_mismatch": float(np.mean(step_mismatch)) if len(step_mismatch) else None,
        "max_abs_u": float(np.max(np.abs(us))) if len(us) else 0.0,
        "mean_u_norm": float(np.mean(u_norm)) if len(us) else 0.0,
        "has_nan_x": bool(np.isnan(xs).any()),
        "has_nan_u": bool(np.isnan(us).any()),
    }
    (out_dir / "check_report.json").write_text(json.dumps(report, indent=2))

    t = np.arange(len(xs))
    fig, axs = plt.subplots(3, 1, figsize=(9, 10))
    axs[0].plot(t, p[:, 0], label="x")
    axs[0].plot(t, p[:, 1], label="y")
    axs[0].plot(t, p[:, 2], label="z")
    axs[0].axhline(p_goal[0], linestyle="--", linewidth=1, color="tab:blue")
    axs[0].axhline(p_goal[1], linestyle="--", linewidth=1, color="tab:orange")
    axs[0].axhline(p_goal[2], linestyle="--", linewidth=1, color="tab:green")
    axs[0].set_title("Payload position")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, p_err, label="payload goal distance")
    axs[1].set_title("Payload position error")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    if len(step_mismatch):
      axs[2].plot(np.arange(len(step_mismatch)), step_mismatch, label="rollout mismatch")
    if len(u_norm):
      axs[2].plot(np.arange(len(u_norm)), u_norm, label="||u||")
    axs[2].set_title("Consistency / control")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "solution_plots.png", dpi=140)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"Saved: {out_dir / 'check_report.json'}")
    print(f"Saved: {out_dir / 'solution_plots.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
try:
    import mujoco  # type: ignore
except Exception:
    mujoco = None
