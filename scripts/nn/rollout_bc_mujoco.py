#!/usr/bin/env python3
import os
import argparse
import yaml
import numpy as np
import torch

from base_simulator import Simulator  # uses MujocoModel internally


# -------------------------
# cfg + trajectory loading
# -------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_trajectory_from_yaml(path: str):
    with open(path, "r") as f:
        traj = yaml.safe_load(f)

    x = np.asarray(traj["states"], dtype=np.float32)   # (Tu+1, nx)
    u = np.asarray(traj["actions"], dtype=np.float32)  # (Tu, nu)

    assert x.ndim == 2 and u.ndim == 2
    assert x.shape[0] == u.shape[0] + 1, \
        f"Expected len(x)=len(u)+1, got {x.shape[0]} and {u.shape[0]}"
    return x, u


# -------------------------
# policy definition (same as training)
# -------------------------
class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]
            d = hidden
        layers += [torch.nn.Linear(d, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCTrajPolicy:
    """Predicts u[t] given (x[t], u_prev[t])."""
    def __init__(self, ckpt_path: str, device: str | None = None):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.nx = int(ckpt["nx"])
        self.nu = int(ckpt["nu"])

        self.x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
        self.x_std  = np.array(ckpt["x_std"], dtype=np.float32)
        self.up_mean = np.array(ckpt["up_mean"], dtype=np.float32)
        self.up_std  = np.array(ckpt["up_std"], dtype=np.float32)
        self.y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
        self.y_std  = np.array(ckpt["y_std"], dtype=np.float32)

        hidden = int(ckpt.get("hidden", 256))
        depth  = int(ckpt.get("depth", 3))

        self.model = MLP(in_dim=self.nx + self.nu, out_dim=self.nu, hidden=hidden, depth=depth)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict_u(self, x_t: np.ndarray, u_prev_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=np.float32).reshape(-1)
        u_prev_t = np.asarray(u_prev_t, dtype=np.float32).reshape(-1)
        assert x_t.shape[0] == self.nx
        assert u_prev_t.shape[0] == self.nu

        x_n  = (x_t - self.x_mean) / self.x_std
        up_n = (u_prev_t - self.up_mean) / self.up_std
        inp = np.concatenate([x_n, up_n], axis=0)

        xb = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        y_pred_n = self.model(xb).squeeze(0).cpu().numpy()
        y_pred = y_pred_n * self.y_std + self.y_mean
        return y_pred.astype(np.float32)


# -------------------------
# rollout
# -------------------------
def rollout(cfg: dict):
    # --- paths ---
    traj_path  = cfg["data"]["traj_path"]
    save_dir   = cfg["output"]["save_dir"]
    ckpt_path  = os.path.join(save_dir, "best.pt")
    env_path   = cfg["sim"]["env_path"]

    # --- sim options ---
    t_start = int(cfg["sim"].get("t_start", 0))
    H_req   = cfg["sim"].get("horizon", None)  # may be None
    log_sim = bool(cfg["sim"].get("log", False))
    rollout_ref = bool(cfg["sim"].get("rollout_ref", False))

    # optional video settings
    video_out_dir = cfg["sim"].get("video_out_dir", "rollout_videos")
    os.makedirs(video_out_dir, exist_ok=True)
    video_cfg = cfg.get("video_cfg", None)

    # action limits (optional)
    u_clip_min = cfg["sim"].get("u_clip_min", None)
    u_clip_max = cfg["sim"].get("u_clip_max", None)

    # --- load data ---
    x_ref_all, u_ref_all = load_trajectory_from_yaml(traj_path)
    Tu_all, nu = u_ref_all.shape
    Tx_all, nx = x_ref_all.shape
    assert Tx_all == Tu_all + 1

    if not (0 <= t_start < Tu_all):
        raise ValueError(f"t_start={t_start} out of range (0..{Tu_all-1})")

    # determine H
    if H_req is None:
        H = Tu_all - t_start
    else:
        H = int(H_req)
        H = max(1, H)
        H = min(H, Tu_all - t_start)  # cannot exceed remaining controls

    # slice window: x length must be H+1, u length H
    x_win = x_ref_all[t_start : t_start + H + 1].astype(np.float32)  # (H+1, nx)
    u_win = u_ref_all[t_start : t_start + H].astype(np.float32)      # (H, nu)

    # --- load policy ---
    policy = BCTrajPolicy(ckpt_path)

    # --- init state from window start ---
    init_state = x_win[0].copy()
    print("Window:", "t_start", t_start, "H", H)
    print("x_win", x_win.shape, "u_win", u_win.shape)
    print("Initial state:", init_state)

    # create simulator
    sim = Simulator(env_path, init_state, log=log_sim)

    # init u_prev
    init_u_prev_mode = str(cfg["sim"].get("init_u_prev", "zeros")).lower()
    if init_u_prev_mode == "zeros":
        u_prev = np.zeros((nu,), dtype=np.float32)
    elif init_u_prev_mode == "traj":
        # "previous control before the first predicted step":
        # safest is to use u_ref at (t_start-1) if available, else zeros
        if t_start > 0:
            u_prev = u_ref_all[t_start - 1].astype(np.float32).copy()
        else:
            u_prev = np.zeros((nu,), dtype=np.float32)
    else:
        raise ValueError("sim.init_u_prev must be 'zeros' or 'traj'")

    # rollout
    x_t = sim.robot.getState().astype(np.float32)

    for t in range(H):
        if rollout_ref:
            u_t = u_win[t]
        else:
            u_t = policy.predict_u(x_t, u_prev)

            if (u_clip_min is not None) or (u_clip_max is not None):
                lo = -np.inf if u_clip_min is None else float(u_clip_min)
                hi =  np.inf if u_clip_max is None else float(u_clip_max)
                u_t = np.clip(u_t, lo, hi).astype(np.float32)

        sim.simulate(u_t)
        x_t = sim.robot.getState().astype(np.float32)
        u_prev = u_t

    # plots / video
    if log_sim:
        sim.plot_sim_trajectories(x_win, u_win, payload=True,
                                  save_path=os.path.join(video_out_dir, "traj_plot.pdf"))
        if video_cfg is not None:
            os.makedirs(video_out_dir, exist_ok=True)
            sim.visualize(cam_cfg=video_cfg, video_basename=video_out_dir)
            print("Saved videos under:", video_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll out learned BC policy in MuJoCo")
    parser.add_argument("cfg", nargs="?", default="cfg.yaml", type=str, help="Path to cfg YAML")
    
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    rollout(cfg)
