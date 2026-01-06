import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_cfg(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class ActionFromPrev(Dataset):
    """
    Supervised BC dataset:
      input  = [ x[t] , u_prev[t] ]
      target = u[t]
    where u_prev[0] = 0, and u_prev[t] = u[t-1] for t>=1.
    """

    def __init__(self, x: np.ndarray, u: np.ndarray, u_prev: np.ndarray,
                 t_start: int, t_end: int,
                 x_mean, x_std, up_mean, up_std, y_mean, y_std):
        assert x.ndim == 2 and u.ndim == 2 and u_prev.ndim == 2
        assert x.shape[0] == u.shape[0] + 1, \
            f"Expected len(x)=len(u)+1, got {x.shape[0]} and {u.shape[0]}"
        assert u_prev.shape == u.shape, "u_prev must have same shape as u"

        self.x = x.astype(np.float32)
        self.u = u.astype(np.float32)
        self.u_prev = u_prev.astype(np.float32)

        self.t_start = int(t_start)
        self.t_end = int(t_end)

        self.x_mean, self.x_std = x_mean, x_std
        self.up_mean, self.up_std = up_mean, up_std
        self.y_mean, self.y_std = y_mean, y_std

    def __len__(self):
        return self.t_end - self.t_start

    def __getitem__(self, i):
        t = self.t_start + i  # t in [t_start, t_end)

        x_t = (self.x[t] - self.x_mean) / self.x_std
        up_t = (self.u_prev[t] - self.up_mean) / self.up_std
        y_t = (self.u[t] - self.y_mean) / self.y_std

        inp = np.concatenate([x_t, up_t], axis=0)  # (nx + nu,)
        return torch.from_numpy(inp), torch.from_numpy(y_t)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCPolicyWrapper(torch.nn.Module):
    """
    ONNX-friendly wrapper:
      forward(x, u_prev) -> u   (physical units)
    x: (B,nx) float32
    u_prev: (B,nu) float32
    """
    def __init__(self, ckpt: dict):
        super().__init__()
        nx = int(ckpt["nx"])
        nu = int(ckpt["nu"])
        hidden = int(ckpt.get("hidden", 256))
        depth  = int(ckpt.get("depth", 3))

        self.nx = nx
        self.nu = nu

        self.net = MLP(in_dim=nx + nu, out_dim=nu, hidden=hidden, depth=depth)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

        # buffers so they’re embedded in the exported graph
        self.register_buffer("x_mean", torch.tensor(ckpt["x_mean"], dtype=torch.float32))
        self.register_buffer("x_std",  torch.tensor(ckpt["x_std"],  dtype=torch.float32))
        self.register_buffer("up_mean", torch.tensor(ckpt["up_mean"], dtype=torch.float32))
        self.register_buffer("up_std",  torch.tensor(ckpt["up_std"],  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(ckpt["y_mean"], dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(ckpt["y_std"],  dtype=torch.float32))

    def forward(self, x: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        # expect (B,nx) and (B,nu)
        x_n  = (x - self.x_mean) / self.x_std
        up_n = (u_prev - self.up_mean) / self.up_std
        inp = torch.cat([x_n, up_n], dim=1)
        y_pred_n = self.net(inp)
        u = y_pred_n * self.y_std + self.y_mean
        return u


def export_onnx_from_ckpt(ckpt_path: str, onnx_path: str, opset: int = 17):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    nx = int(ckpt["nx"])
    nu = int(ckpt["nu"])

    model = BCPolicyWrapper(ckpt).eval()

    # batch=1 fixed (robust for control loops)
    x = torch.zeros(1, nx, dtype=torch.float32)
    u_prev = torch.zeros(1, nu, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (x, u_prev),
        onnx_path,
        input_names=["x", "u_prev"],
        output_names=["u"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print("Saved ONNX:", onnx_path)

def make_u_prev(u: np.ndarray, mode: str = "zeros") -> np.ndarray:
    """
    Create u_prev with same shape as u.
    - zeros: u_prev[0]=0, u_prev[t]=u[t-1]
    - repeat_first: u_prev[0]=u[0], u_prev[t]=u[t-1]
    """
    Tu, nu = u.shape
    u_prev = np.zeros_like(u, dtype=np.float32)
    if mode.lower() == "zeros":
        u_prev[0] = 0.0
    elif mode.lower() == "repeat_first":
        u_prev[0] = u[0]
    else:
        raise ValueError("u_prev_mode must be 'zeros' or 'repeat_first'")

    if Tu > 1:
        u_prev[1:] = u[:-1]
    return u_prev


def train_bc(x: np.ndarray, u: np.ndarray, cfg: dict):
    train_ratio = float(cfg["training"]["train_ratio"])
    epochs      = int(cfg["training"]["epochs"])
    batch_size  = int(cfg["training"]["batch_size"])
    lr          = float(cfg["training"]["learning_rate"])

    hidden = int(cfg["model"]["hidden"])
    depth  = int(cfg["model"]["depth"])

    save_dir = cfg["output"]["save_dir"]

    u_prev_mode = cfg.get("data", {}).get("u_prev_mode", "zeros")

    assert x.ndim == 2 and u.ndim == 2
    T, nx = x.shape
    Tu, nu = u.shape
    assert Tu + 1 == T
    assert Tu >= 2, "Need at least 2 actions to split train/test."

    # Build u_prev
    u_prev = make_u_prev(u, mode=u_prev_mode)

    # We have samples for t = 0..Tu-1
    n_samples = Tu
    n_train = int(train_ratio * n_samples)
    n_train = max(1, min(n_train, n_samples - 1))  # keep at least 1 test

    train_t0, train_t1 = 0, n_train
    test_t0,  test_t1  = n_train, n_samples

    # Normalization computed only on TRAIN window
    x_train  = x[train_t0:train_t1]         # x[t]
    up_train = u_prev[train_t0:train_t1]    # u_prev[t]
    y_train  = u[train_t0:train_t1]         # u[t]

    eps = 1e-6
    x_mean, x_std   = x_train.mean(axis=0),  np.maximum(x_train.std(axis=0), eps)
    up_mean, up_std = up_train.mean(axis=0), np.maximum(up_train.std(axis=0), eps)
    y_mean, y_std   = y_train.mean(axis=0),  np.maximum(y_train.std(axis=0), eps)

    lambda_reg = float(cfg["training"].get("lambda_reg", 0.0))


    train_ds = ActionFromPrev(x, u, u_prev, train_t0, train_t1,
                              x_mean, x_std, up_mean, up_std, y_mean, y_std)
    test_ds  = ActionFromPrev(x, u, u_prev, test_t0, test_t1,
                              x_mean, x_std, up_mean, up_std, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=nx + nu, out_dim=nu, hidden=hidden, depth=depth).to(device)

    # tensors for mapping u_prev -> y-normalized space
    up_mean_t = torch.tensor(up_mean, dtype=torch.float32, device=device)
    up_std_t  = torch.tensor(up_std,  dtype=torch.float32, device=device)
    y_mean_t  = torch.tensor(y_mean,  dtype=torch.float32, device=device)
    y_std_t   = torch.tensor(y_std,   dtype=torch.float32, device=device)


    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            # xb = [x_norm, u_prev_norm_up]  -> extract the u_prev part:
            u_prev_norm_up = xb[:, -nu:]  # normalized using up_mean/up_std

            # convert u_prev to y-normalized coordinates:
            u_prev_phys = u_prev_norm_up * up_std_t + up_mean_t
            u_prev_norm_y = (u_prev_phys - y_mean_t) / y_std_t

            loss = loss_fn(pred, yb)
            if lambda_reg > 0.0:
                loss = loss + lambda_reg * loss_fn(pred, u_prev_norm_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr.append(loss.item())

        model.eval()
        te = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                te.append(loss_fn(model(xb), yb).item())

        train_mse = float(np.mean(tr))
        test_mse  = float(np.mean(te)) if te else float("nan")
        train_losses.append(train_mse)
        test_losses.append(test_mse)
        print(f"epoch {ep:03d} | train_mse={train_mse:.6f} | test_mse={test_mse:.6f}")

    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "nx": nx, "nu": nu,
        "hidden": hidden,
        "depth": depth,
        "u_prev_mode": u_prev_mode,
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "up_mean": up_mean.tolist(), "up_std": up_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
    }
    torch.save(ckpt, os.path.join(save_dir, "best.pt"))
    print("Saved:", os.path.join(save_dir, "best.pt"))
    if bool(cfg["output"].get("generate_onnx", False)):
        onnx_path = cfg["output"].get("onnx_path", os.path.join(save_dir, "bc_policy.onnx"))
        export_onnx_from_ckpt(os.path.join(save_dir, "best.pt"), onnx_path)

    return train_losses, test_losses


class BCTrajPolicy:
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

        self.model = MLP(
            in_dim=self.nx + self.nu,
            out_dim=self.nu,
            hidden=int(ckpt.get("hidden", 256)),
            depth=int(ckpt.get("depth", 3))
        )
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


def load_trajectory_from_yaml(path: str, flip_quat: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        traj = yaml.safe_load(f)

    x = np.asarray(traj["states"], dtype=np.float32)
    u = np.asarray(traj["actions"], dtype=np.float32)

    assert x.ndim == 2 and u.ndim == 2
    assert x.shape[0] == u.shape[0] + 1, \
        f"Expected len(x)=len(u)+1, got len(x)={x.shape[0]}, len(u)={u.shape[0]}"

    if flip_quat:
        nx = x.shape[1]
        num_bodies = nx // 13
        for i in range(num_bodies):
            old_quat = x[:, 7*i+3:7*i+7].copy()
            x[:, 7*i+3]       = old_quat[:, 3]
            x[:, 7*i+4:7*i+7] = old_quat[:, 0:3]

    return x, u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC: predict u[t] from (x[t], u[t-1])")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    traj_path = cfg["data"]["traj_path"]
    x, u = load_trajectory_from_yaml(traj_path, flip_quat=False)

    print("Trajectory:")
    print(f"  len(x) = {x.shape[0]}   (T+1)")
    print(f"  len(u) = {u.shape[0]}   (T)")
    print(f"  nx     = {x.shape[1]}")
    print(f"  nu     = {u.shape[1]}")

    train_losses, test_losses = train_bc(x, u, cfg)

    # sanity plot: predict u[t] for all t
    save_dir = cfg["output"]["save_dir"]
    policy = BCTrajPolicy(os.path.join(save_dir, "best.pt"))

    Tu, nu = u.shape
    u_prev = make_u_prev(u, mode=cfg.get("data", {}).get("u_prev_mode", "zeros"))

    u_pred_all = np.zeros((Tu, nu), dtype=np.float32)
    for t in range(Tu):
        u_pred_all[t] = policy.predict_u(x[t], u_prev[t])

    save_dir = cfg["output"]["save_dir"]
    pdf_path = cfg["output"]["loss_pdf"]
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    # ---- Plot 1: u[t] vs pred u[t] (stacked nu subplots) ----
    fig_u, axes = plt.subplots(nu, 1, sharex=True, figsize=(10, 8))
    time = np.arange(Tu)
    if nu == 1:
        axes = [axes]

    for i in range(nu):
        axes[i].plot(time, u[:, i], label="true u[t]")
        axes[i].plot(time, u_pred_all[:, i], label="pred u[t]")
        axes[i].set_ylabel(f"u[{i}]")
        axes[i].grid(True)

    axes[-1].set_xlabel("t")
    axes[0].legend()
    fig_u.tight_layout()

    # ---- Plot 2: loss curves ----
    fig_loss = plt.figure(figsize=(8, 4))
    epochs_arr = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs_arr, train_losses, label="train")
    plt.plot(epochs_arr, test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("MSE (normalized)")
    plt.title("Training vs Test Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---- Save both into ONE PDF ----
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_u)
        pdf.savefig(fig_loss)

    plt.close(fig_u)
    plt.close(fig_loss)

    print("Saved training PDF:", pdf_path)
