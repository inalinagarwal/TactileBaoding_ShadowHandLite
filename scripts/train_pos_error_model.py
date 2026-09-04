"""Train an MLP to predict next-step joint_pos_error from a window of pos/vel/action history.

Pure PyTorch/numpy -- no Isaac Lab import, runs anywhere with the collected
`logs/pos_error_data/*.npz` files (see collect_pos_error_data.py).

Input (169-D): concat(pos4[52], vel4[52], act4[52], cur_action[13])
  - pos4/vel4/act4: 4 stacked frames (oldest -> newest) of normalised joint pos / vel /
    previous action, taken straight from the policy's obs stack.
  - cur_action: the action the policy just emitted (about to be applied).
Target (13-D): joint_pos_error read AFTER env.step(cur_action) -- i.e. the error the
  policy's encoder will see at the next timestep as a consequence of cur_action.

Usage:
    python train_pos_error_model.py --data_glob "../logs/pos_error_data/*.npz" \
        --output ../logs/pos_error_model/pos_error_mlp.pt
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from multimodal_rl.models.mlp import MLP

INPUT_DIM = 169
TARGET_DIM = 13


def load_dataset(data_glob: str):
    """Load and window all collected npz files into (X, y) arrays, filtering invalid rows."""
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched {data_glob!r}")

    X_parts, y_parts = [], []
    control_names = None
    obs_stack = None
    kept_total = 0
    raw_total = 0

    for path in paths:
        d = np.load(path, allow_pickle=True)
        if control_names is None:
            control_names = d["control_names"]
            obs_stack = int(d["obs_stack"])
        else:
            assert list(d["control_names"]) == list(control_names), f"{path}: control_names mismatch"

        pos4, vel4, act4 = d["pos4"], d["vel4"], d["act4"]
        cur_action, target = d["cur_action"], d["target"]
        done, steps_since_reset = d["done"], d["steps_since_reset"]

        num_steps, num_envs = done.shape
        raw_total += num_steps * num_envs

        # Row t is valid iff:
        #  (a) the 4-frame window (age steps_since_reset[t]) is fully within one episode
        #      (i.e. not still contaminated by pre-reset frames or hard-reset duplicate fill), and
        #  (b) this step's target doesn't belong to a freshly auto-reset episode (done[t]==False).
        valid = (steps_since_reset >= obs_stack) & (~done)

        X = np.concatenate(
            [pos4.reshape(num_steps, num_envs, -1),
             vel4.reshape(num_steps, num_envs, -1),
             act4.reshape(num_steps, num_envs, -1),
             cur_action.reshape(num_steps, num_envs, -1)],
            axis=-1,
        )
        y = target

        X_valid = X[valid]
        y_valid = y[valid]
        kept_total += X_valid.shape[0]

        X_parts.append(X_valid.astype(np.float32))
        y_parts.append(y_valid.astype(np.float32))

        print(f"[INFO] {os.path.basename(path)}: {num_steps}x{num_envs} rows, "
              f"{X_valid.shape[0]} valid ({100 * X_valid.shape[0] / (num_steps * num_envs):.1f}%)")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    print(f"[INFO] Total: {raw_total} raw rows -> {kept_total} valid rows "
          f"({100 * kept_total / raw_total:.1f}%)  X={X_all.shape} y={y_all.shape}")
    assert X_all.shape[1] == INPUT_DIM, f"expected input dim {INPUT_DIM}, got {X_all.shape[1]}"
    assert y_all.shape[1] == TARGET_DIM, f"expected target dim {TARGET_DIM}, got {y_all.shape[1]}"
    return X_all, y_all, list(control_names)


class PosErrorMLP(nn.Module):
    """MLP predictor built from the multimodal_rl MLP block, matching the encoder/policy style."""

    def __init__(self, input_dim=INPUT_DIM, hiddens=(256, 256, 128), target_dim=TARGET_DIM):
        super().__init__()
        activations = ["elu"] * len(hiddens)
        self.net = MLP(input_dim, list(hiddens), activations, layernorm=True)
        self.head = nn.Linear(hiddens[-1], target_dim)

    def forward(self, x):
        return self.head(self.net(x))


def main():
    parser = argparse.ArgumentParser(description="Train the pos-error prediction MLP.")
    parser.add_argument("--data_glob", type=str, default="../logs/pos_error_data/*.npz")
    parser.add_argument("--output", type=str, default="../logs/pos_error_model/pos_error_mlp.pt")
    parser.add_argument("--hiddens", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, control_names = load_dataset(args.data_glob)
    n = X.shape[0]

    # Split by contiguous shuffled blocks (rows already come from different env/step
    # combinations, so a random row-level split is fine here; there's no per-trajectory
    # leakage concern since input/target are single-transition, not sequence-level).
    perm = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
    y_mean, y_std = y[train_idx].mean(0), y[train_idx].std(0) + 1e-6

    device = torch.device(args.device)
    Xt = torch.from_numpy((X - x_mean) / x_std).to(device)
    yt = torch.from_numpy((y - y_mean) / y_std).to(device)
    y_raw = torch.from_numpy(y).to(device)

    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)

    model = PosErrorMLP(INPUT_DIM, tuple(args.hiddens), TARGET_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    y_std_t = torch.from_numpy(y_std).to(device)
    y_mean_t = torch.from_numpy(y_mean).to(device)

    def denorm(pred_norm):
        return pred_norm * y_std_t + y_mean_t

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    n_train = train_idx_t.shape[0]
    for epoch in range(args.epochs):
        model.train()
        shuffled = train_idx_t[torch.randperm(n_train, device=device)]
        total_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            batch = shuffled[i:i + args.batch_size]
            xb, yb = Xt[batch], yt[batch]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.shape[0]
        train_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(Xt[val_idx_t])
            val_pred = denorm(val_pred_norm)
            val_rmse = torch.sqrt(torch.mean((val_pred - y_raw[val_idx_t]) ** 2)).item()

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[epoch {epoch:3d}] train_norm_mse={train_loss:.5f}  val_rmse_rad={val_rmse:.5f}  best={best_val:.5f}")

        if epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()

    # --- evaluation vs baselines ------------------------------------------------
    with torch.no_grad():
        val_pred = denorm(model(Xt[val_idx_t]))
        y_val = y_raw[val_idx_t]

        rmse_model = torch.sqrt(torch.mean((val_pred - y_val) ** 2, dim=0))
        rmse_zero = torch.sqrt(torch.mean(y_val ** 2, dim=0))

        # predict-mean baseline (the pos_error slice was deliberately excluded from the
        # input features, so a predict-previous-error baseline isn't available post-filtering).
        y_train_mean = y[train_idx].mean(0)
        pred_mean = torch.from_numpy(y_train_mean).to(device).unsqueeze(0).expand_as(y_val)
        rmse_mean = torch.sqrt(torch.mean((pred_mean - y_val) ** 2, dim=0))

    print("\n[RESULTS] Per-joint validation RMSE (radians):")
    print(f"{'joint':12s} {'model':>10s} {'zero':>10s} {'mean':>10s}")
    for i, name in enumerate(control_names):
        print(f"{name:12s} {rmse_model[i].item():10.5f} {rmse_zero[i].item():10.5f} {rmse_mean[i].item():10.5f}")
    print(f"{'OVERALL':12s} {rmse_model.mean().item():10.5f} {rmse_zero.mean().item():10.5f} {rmse_mean.mean().item():10.5f}")

    # --- save --------------------------------------------------------------------
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "input_dim": INPUT_DIM, "target_dim": TARGET_DIM,
        "hiddens": list(args.hiddens),
        "control_names": control_names,
        "input_layout": (
            "X = concat(pos4[52], vel4[52], act4[52], cur_action[13]) -> 169-D; "
            "frame order oldest(t-3)->newest(t); target = joint_pos_error after env.step(cur_action)"
        ),
        "val_rmse_per_joint": rmse_model.cpu().numpy(),
        "val_rmse_overall": rmse_model.mean().item(),
    }, out_path)
    print(f"\n[INFO] Saved model -> {out_path}")


if __name__ == "__main__":
    main()
