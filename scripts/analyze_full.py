#!/usr/bin/env python3
"""Analyze a full-pipeline recording produced by record_full.py.

Pure numpy/matplotlib — no Isaac Lab import needed, so this can run outside
the s2r conda env / without a GPU.

Reports, per joint, the discrepancy at each pipeline stage:
  - action_scaled_13 (pre-coupling proxy) vs joint_pos_cmd[control cols]
    (post-coupling)  -> isolates what the coupling law changes.
  - joint_pos_cmd vs joint_pos (achieved)  -> PD tracking error.
  - a dedicated panel per coupled (J1, J2) pair showing the proxy, the
    post-coupling commands, and the achieved positions.
  - a best-effort reconstruction of the encoder's raw "prop" input from the
    logged joint_pos/joint_vel/joint_pos_error/action, to sanity-check that
    the observation the encoder saw matches the raw joint state (see the
    NOTE on frame-stack timing below).

NOTE on reset-settle: for a few steps right after every reset, RotoEnv's
settle_counter forces joint_pos_cmd to the default pose for ALL joints
regardless of coupling (see roto_env.py _pre_physics_step). This makes
action_scaled_13 vs joint_pos_cmd disagree on EVERY joint during that
window, not just the coupled ones. Use --skip to drop those steps (per
episode) before computing stats/plots; the obs-reconstruction sanity check
runs on the unskipped data since it needs strict step-to-step adjacency.

Usage:
    python analyze_full.py ../logs/full_pipeline_seed42.npz --out-dir ../logs/full_pipeline_plots
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


PER_STEP_KEYS = [
    "enc_in_prop", "enc_in_tactile", "enc_in_fused", "z",
    "action", "action_scaled_13",
    "joint_pos_cmd", "joint_pos", "joint_vel", "joint_pos_error", "torque",
    "tac",
]


def build_keep_mask(data: dict, skip: int) -> np.ndarray:
    """True to keep. Drops the first `skip` steps of every episode (reset-settle)."""
    T = data["joint_pos"].shape[0]
    ends = [int(e) for e in data["episode_ends"]]
    starts = [0] + [e + 1 for e in ends[:-1]]
    keep = np.ones(T, dtype=bool)
    for s in starts:
        keep[s:s + skip] = False
    return keep


def apply_mask(data: dict, keep: np.ndarray) -> dict:
    out = dict(data)
    for k in PER_STEP_KEYS:
        out[k] = data[k][keep]
    return out


def build_index_maps(data: dict):
    actuated_names = [str(n) for n in data["actuated_names"]]
    control_names = [str(n) for n in data["control_names"]]
    driver_names = [str(n) for n in data["coupled_driver_names"]]
    dependent_names = [str(n) for n in data["coupled_dependent_names"]]

    name_to_col = {n: j for j, n in enumerate(actuated_names)}
    control_cols = [name_to_col[n] for n in control_names]          # 13 -> col in the 16-wide arrays
    driver_cols_16 = [name_to_col[n] for n in driver_names]          # 3  -> col in the 16-wide arrays
    dependent_cols_16 = [name_to_col[n] for n in dependent_names]    # 3  -> col in the 16-wide arrays
    driver_cols_13 = [control_names.index(n) for n in driver_names]  # 3  -> col in the 13-wide (control) arrays

    return {
        "actuated_names": actuated_names,
        "control_names": control_names,
        "driver_names": driver_names,
        "dependent_names": dependent_names,
        "control_cols": control_cols,
        "driver_cols_16": driver_cols_16,
        "dependent_cols_16": dependent_cols_16,
        "driver_cols_13": driver_cols_13,
    }


def print_header(data: dict) -> None:
    print("=== Recording metadata ===")
    print(f"checkpoint: {data['checkpoint']}")
    print(f"seed: {int(data['seed'])}   rl_dt: {float(data['rl_dt']):.4f} s")
    print(f"Kp={float(data['Kp'])}  Kd={float(data['Kd'])}  coupling_theta={float(data['coupling_theta']):.3f} rad")
    print(f"lock_coupled_dependent_at_zero: {bool(data['lock_coupled_dependent_at_zero'])}")
    n_steps = data["joint_pos"].shape[0]
    print(f"steps recorded: {n_steps}   episodes: {len(data['episode_ends'])}")
    print()


def per_joint_table(data: dict, maps: dict, out_dir: str) -> None:
    names = maps["actuated_names"]
    control_cols = maps["control_cols"]
    cmd = data["joint_pos_cmd"]      # (T,16)
    pos = data["joint_pos"]          # (T,16)
    vel = data["joint_vel"]          # (T,16)
    torque = data["torque"]          # (T,16)

    err = cmd - pos
    rms_err_deg = np.degrees(np.sqrt((err ** 2).mean(axis=0)))

    # "Clamped" variant: roto_env's _pre_physics_step writes joint_pos_cmd via
    # scale(action, lower, upper) with NO clamp — since the policy's raw
    # deterministic action is unbounded (identity output layer, no tanh/clip;
    # see policy_value.py act()), joint_pos_cmd can overshoot the joint's
    # physical range by hundreds of degrees whenever |action| > 1. The 3
    # coupled driver/dependent joints are exempt (already torch.clamp'd inside
    # _handle_coupled_joints). Clamping cmd to [lower, upper] here mimics what
    # deploy_policy.py's hardware path does (np.clip before publishing,
    # deploy_policy.py:383) and gives a physically-meaningful error alongside
    # the raw (env-authentic) one.
    rms_err_clamped_deg = np.full(len(names), np.nan)
    cmd_ctrl_clamped = np.clip(cmd[:, control_cols], data["joint_lower"], data["joint_upper"])
    err_clamped = cmd_ctrl_clamped - pos[:, control_cols]
    rms_err_clamped_deg[control_cols] = np.degrees(np.sqrt((err_clamped ** 2).mean(axis=0)))

    mean_torque = torque.mean(axis=0)
    max_abs_torque = np.abs(torque).max(axis=0)
    mean_abs_vel = np.abs(vel).mean(axis=0)

    print("=== Per-joint tracking (16 actuated joints) ===")
    print("(RMS err = raw joint_pos_cmd-pos, matches roto_env's own joint_pos_error exactly;")
    print(" RMS err clamped = cmd clipped to [joint_lower,joint_upper] first, like hardware does)")
    header = (f"{'joint':8s} {'RMS err (deg)':>13s} {'RMS err clamp':>13s} "
              f"{'mean torque':>12s} {'max|torque|':>12s} {'mean|vel|':>10s}")
    print(header)
    rows = []
    for j, name in enumerate(names):
        clamp_str = f"{rms_err_clamped_deg[j]:.2f}" if not np.isnan(rms_err_clamped_deg[j]) else "n/a"
        row = [name, f"{rms_err_deg[j]:.2f}", clamp_str, f"{mean_torque[j]:.4f}", f"{max_abs_torque[j]:.4f}", f"{mean_abs_vel[j]:.4f}"]
        rows.append(row)
        print(f"{name:8s} {rms_err_deg[j]:13.2f} {clamp_str:>13s} {mean_torque[j]:12.4f} {max_abs_torque[j]:12.4f} {mean_abs_vel[j]:10.4f}")
    print()

    csv_path = os.path.join(out_dir, "per_joint_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "rms_err_deg", "rms_err_clamped_deg", "mean_torque", "max_abs_torque", "mean_abs_vel"])
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, rms_err_deg, w, label="raw (env-authentic)", color="#4c72b0")
    mask = ~np.isnan(rms_err_clamped_deg)
    ax.bar(x[mask] + w / 2, rms_err_clamped_deg[mask], w, label="clamped (hw-style)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("RMS tracking error (deg)")
    ax.set_title("Commanded vs achieved joint position (post-coupling)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_joint_tracking_error.png"), dpi=150)
    plt.close(fig)


def plot_joint_velocities(data: dict, maps: dict, out_dir: str) -> None:
    """4x4 grid: achieved joint velocity (deg/s) vs step, for all 16 actuated joints."""
    names = maps["actuated_names"]
    vel_deg = np.degrees(data["joint_vel"])  # (T,16)

    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(vel_deg[:, j], color="#4c72b0", lw=0.9)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(names[j], fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("deg/s")
    fig.suptitle("Achieved joint velocity (all 16 actuated joints)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "joint_velocities.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_cmd_vs_pos(data: dict, maps: dict, out_dir: str) -> None:
    """All 16 actuated joints: scaled+coupled commanded target (joint_pos_cmd,
    what's actually sent to the robot) vs achieved joint position, in degrees.

    For the 10 joints with a known physical range (the control joints, minus
    the 3 J2 drivers which are always clamped by the coupling law anyway),
    the y-axis is bounded to [lower-20%, upper+20%] of that joint's own range
    for readability — see the per-joint tracking-error note on why the raw
    (unclamped) command can otherwise run far outside the joint's physical
    range. The 3 coupled dependent (J1) joints have no saved limits (they are
    already clamped inside _handle_coupled_joints) so they autoscale.
    """
    names = maps["actuated_names"]
    control_cols = maps["control_cols"]
    cmd_deg = np.degrees(data["joint_pos_cmd"])   # (T,16) actual command (post scale + coupling)
    pos_deg = np.degrees(data["joint_pos"])       # (T,16) achieved

    col_to_limits = {}
    lower_deg = np.degrees(data["joint_lower"])
    upper_deg = np.degrees(data["joint_upper"])
    for k, col in enumerate(control_cols):
        col_to_limits[col] = (lower_deg[k], upper_deg[k])

    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
    axes_flat = axes.flat
    for j, name in enumerate(names):
        ax = axes_flat[j]
        cmd_j, pos_j = cmd_deg[:, j], pos_deg[:, j]
        if j in col_to_limits:
            lo, hi = col_to_limits[j]
            span = hi - lo
            ylo, yhi = lo - 0.2 * span, hi + 0.2 * span
            ax.plot(np.clip(cmd_j, ylo, yhi), color="k", ls="--", lw=1.0, alpha=0.8, label="cmd (scaled+coupled)")
            ax.axhline(lo, color="r", lw=0.6, ls=":", alpha=0.6)
            ax.axhline(hi, color="r", lw=0.6, ls=":", alpha=0.6)
            ax.set_ylim(ylo, yhi)
        else:
            ax.plot(cmd_j, color="k", ls="--", lw=1.0, alpha=0.8, label="cmd (scaled+coupled)")
        ax.plot(pos_j, color="C0", lw=1.2, label="q (achieved)")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
    for j in range(n, nrows * ncols):
        axes_flat[j].axis("off")
    axes_flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Commanded (joint_pos_cmd, clipped to plot range where limits known) vs achieved q — all 16 joints")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "cmd_vs_pos.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def coupling_effect(data: dict, maps: dict, out_dir: str) -> None:
    """Isolate what _handle_coupled_joints changes: proxy (pre-coupling) vs
    post-coupling commands, for the 3 driver (J2) joints only — the other 10
    control joints pass through _pre_physics_step's scale() unchanged."""
    control_cols = maps["control_cols"]
    action_scaled = data["action_scaled_13"]              # (T,13), pre-coupling
    cmd = data["joint_pos_cmd"]                             # (T,16), post-coupling

    cmd_control = cmd[:, control_cols]                       # (T,13), same order as action_scaled_13
    diff = cmd_control - action_scaled
    max_abs_diff = np.abs(diff).max(axis=0)

    print("=== Coupling effect (post-coupling minus pre-coupling, per control joint) ===")
    for j, name in enumerate(maps["control_names"]):
        tag = " <- J2 driver (expected to differ)" if j in maps["driver_cols_13"] else ""
        print(f"  {name:8s}  max|diff| = {max_abs_diff[j]:.4f} rad{tag}")
    print()


def coupled_pairs_panel(data: dict, maps: dict, out_dir: str) -> None:
    driver_names = maps["driver_names"]
    dependent_names = maps["dependent_names"]
    action_scaled = data["action_scaled_13"]     # (T,13) proxy (pre-coupling)
    cmd = data["joint_pos_cmd"]                   # (T,16) post-coupling
    pos = data["joint_pos"]                        # (T,16) achieved

    fig, axes = plt.subplots(len(driver_names), 1, figsize=(12, 3.2 * len(driver_names)), sharex=True)
    if len(driver_names) == 1:
        axes = [axes]

    for i, (drv, dep) in enumerate(zip(driver_names, dependent_names)):
        finger = drv.replace("rh_", "").replace("J2", "")
        col16_drv = maps["driver_cols_16"][i]
        col16_dep = maps["dependent_cols_16"][i]
        col13_drv = maps["driver_cols_13"][i]

        proxy = action_scaled[:, col13_drv]
        j2_cmd = cmd[:, col16_drv]
        j1_cmd = cmd[:, col16_dep]
        j2_q = pos[:, col16_drv]
        j1_q = pos[:, col16_dep]

        ax = axes[i]
        ax.plot(proxy, "k:", lw=1.0, alpha=0.7, label="J2 proxy (pre-coupling)")
        ax.plot(j2_cmd, color="C0", label=f"{drv} cmd (post-coupling)")
        ax.plot(j2_q, color="C0", ls="--", alpha=0.8, label=f"{drv} achieved")
        ax.plot(j1_cmd, color="C3", label=f"{dep} cmd (post-coupling)")
        ax.plot(j1_q, color="C3", ls="--", alpha=0.8, label=f"{dep} achieved")
        ax.set_title(f"{finger}: coupled J1<-J2")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

        frac_j1_gt_j2 = float((j1_q > j2_q).mean())
        print(f"  {finger}: achieved J1 > achieved J2 in {100 * frac_j1_gt_j2:.1f}% of steps "
              f"(J1 mean={np.degrees(j1_q.mean()):.1f} deg, J2 mean={np.degrees(j2_q.mean()):.1f} deg)")

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coupled_joints_panel.png"), dpi=150)
    plt.close(fig)
    print(f"[saved] {os.path.join(out_dir, 'coupled_joints_panel.png')}")
    print()


def obs_reconstruction_sanity(data: dict, maps: dict) -> None:
    """Best-effort check that the encoder's logged 'prop' input matches the
    raw joint state.

    NOTE on timing: `enc_in_prop` at row t is captured BEFORE env.step() at
    iteration t, so its newest (last 52) frame reflects iteration (t-1)'s
    POST-step values (pos/vel/err/action), which this script logs at row
    (t-1). We therefore compare row t's newest frame against row (t-1)'s raw
    values; row 0 has no valid predecessor (it's the reset-fill) and is
    skipped.
    """
    control_cols = maps["control_cols"]
    joint_lower = data["joint_lower"]
    joint_upper = data["joint_upper"]
    vel_limits = data["joint_vel_limits"]

    pos = data["joint_pos"][:, control_cols]
    vel = data["joint_vel"][:, control_cols]
    err = data["joint_pos_error"][:, control_cols]
    action = data["action"]
    enc_in_prop = data["enc_in_prop"]  # (T, 52*obs_stack)

    frame_len = 13 * 4  # pos13+vel13+err13+act13 per frame
    newest = enc_in_prop[:, -frame_len:]  # (T, 52), newest frame per row

    recon = np.concatenate(
        [unscale(pos, joint_lower, joint_upper), unscale(vel, -vel_limits, vel_limits), err, action],
        axis=1,
    )  # (T, 52), row t = raw values recorded AT row t

    # shift: compare newest[t] (reflects row t-1's raw values) against recon[t-1]
    diff = newest[1:] - recon[:-1]
    max_abs = np.abs(diff).max()
    mean_abs = np.abs(diff).mean()
    print("=== Observation reconstruction sanity check (encoder 'prop' input vs raw joint state) ===")
    print(f"  max|diff| = {max_abs:.5f}   mean|diff| = {mean_abs:.5f}  (expect ~0; large values indicate a "
          f"mismatch in obs assembly, normalization, or the frame-stack timing assumption above)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a record_full.py recording.")
    ap.add_argument("npz_path", type=str)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument(
        "--skip", type=int, default=15,
        help="Drop this many steps from the start of every episode (reset-settle transient) "
             "before computing per-joint / coupling stats and plots (default: 15).",
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (os.path.splitext(args.npz_path)[0] + "_plots")
    os.makedirs(out_dir, exist_ok=True)

    data = load(args.npz_path)
    maps = build_index_maps(data)

    print_header(data)
    # Run the obs sanity check on the UNMASKED data: it needs strict step-to-step
    # adjacency (row t vs row t-1), which --skip would break across episode boundaries.
    obs_reconstruction_sanity(data, maps)

    keep = build_keep_mask(data, args.skip)
    n_dropped = int((~keep).sum())
    print(f"[INFO] Dropping {n_dropped}/{keep.size} steps as reset-settle transient (--skip {args.skip})\n")
    data_ss = apply_mask(data, keep)

    per_joint_table(data_ss, maps, out_dir)
    plot_joint_velocities(data_ss, maps, out_dir)
    plot_cmd_vs_pos(data_ss, maps, out_dir)
    coupling_effect(data_ss, maps, out_dir)
    coupled_pairs_panel(data_ss, maps, out_dir)

    print(f"Plots + CSV saved to {out_dir}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
