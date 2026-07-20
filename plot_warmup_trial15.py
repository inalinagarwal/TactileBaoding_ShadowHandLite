#!/usr/bin/env python3
"""Plot empty-motion warmup sensors (12 FSR + 4 BioTac) and compare to sim tac ON%.

Usage (on analysis machine after copying warmup npz back):
  python plot_warmup_trial15.py
  python plot_warmup_trial15.py --warmup hw_warmup_trial15_fsr.npz \\
      --sim sim_policy_log_trial15_seed42.npz --out-dir warmup_trial15_plots
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

TAC_NAMES = (
    "world forearm palm ffknuck mfknuck rfknuckle thbase "
    "ffprox mfprox rfprox thprox ffmid mfmid rfmid thhub "
    "ffdist mfdist rfdist thmid fftip mftip rftip thdist thtip"
).split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", default="hw_warmup_trial15_fsr.npz")
    ap.add_argument("--sim", default="sim_policy_log_trial15_seed42.npz")
    ap.add_argument("--out-dir", default="warmup_trial15_plots")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    warm_path = args.warmup if os.path.isabs(args.warmup) else os.path.join(root, args.warmup)
    sim_path = args.sim if os.path.isabs(args.sim) else os.path.join(root, args.sim)
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out, exist_ok=True)

    w = np.load(warm_path, allow_pickle=True)
    fsr = w["fsr"].astype(float)
    bt = w["biotac_pdc"].astype(float)
    t = w["t"].astype(float)
    t0 = t - t[0]
    fsr_names = [str(x) for x in w["fsr_names"]]
    bt_names = [str(x) for x in w["biotac_names"]]
    fsr_ch = w["fsr_channels"].astype(int)
    bt_ch = w["biotac_channels"].astype(int)
    bt_idx = w["biotac_idx"].astype(int)

    # --- 12 FSR raw time series ---
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t0, fsr[:, i], lw=0.8, color="C0")
        if "fsr_hi" in w.files:
            ax.axhline(float(w["fsr_hi"][i]), color="C3", ls="--", lw=0.8, label="hi")
            ax.axhline(float(w["fsr_lo"][i]), color="C2", ls="--", lw=0.8, label="lo")
            ax.axhline(float(w["fsr_baseline"][i]), color="k", ls=":", lw=0.8, label="med")
        ax.set_title(fsr_names[i])
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Empty-motion warmup: raw FSR (+ fitted thresholds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_fsr_raw_warmup.png"), dpi=150)
    plt.close(fig)

    # --- 4 BioTac PDC ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for ax, k, name in zip(axes.flat, bt_idx, bt_names):
        ax.plot(t0, bt[:, k], lw=0.8, color="C1")
        if "bt_hi" in w.files:
            j = list(bt_idx).index(k)
            ax.axhline(float(w["bt_hi"][j]), color="C3", ls="--", lw=0.8)
            ax.axhline(float(w["bt_lo"][j]), color="C2", ls="--", lw=0.8)
            ax.axhline(float(w["bt_baseline"][j]), color="k", ls=":", lw=0.8)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Empty-motion warmup: BioTac PDC (+ fitted thresholds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_biotac_raw_warmup.png"), dpi=150)
    plt.close(fig)

    # --- 16-sensor summary bars (range / std) ---
    labels = fsr_names + bt_names
    ranges = list(fsr.max(0) - fsr.min(0))
    stds = list(fsr.std(0))
    for k in bt_idx:
        col = bt[:, k]
        valid = col[np.isfinite(col)]
        ranges.append(float(valid.max() - valid.min()) if valid.size else 0.0)
        stds.append(float(valid.std()) if valid.size else 0.0)

    x = np.arange(16)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax0.bar(x, ranges, color="#4c72b0")
    ax0.set_ylabel("raw range")
    ax0.set_title("Warmup empty-motion: sensor dynamic range (12 FSR + 4 BioTac)")
    ax0.grid(True, axis="y", alpha=0.3)
    ax1.bar(x, stds, color="#dd8452")
    ax1.set_ylabel("raw std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03_sensor_range_std.png"), dpi=150)
    plt.close(fig)

    # --- binary ON% vs sim (if available) ---
    if "tac_bin" in w.files and os.path.isfile(sim_path):
        htac = w["tac_bin"].astype(float)
        stac = np.load(sim_path)["tac"].astype(float)
        plot_ch = list(fsr_ch) + list(bt_ch)
        plot_lab = [TAC_NAMES[c] for c in plot_ch]
        hv = [100.0 * htac[:, c].mean() for c in plot_ch]
        sv = [100.0 * stac[:, c].mean() for c in plot_ch]
        xx = np.arange(len(plot_ch))
        ww = 0.38
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(xx - ww / 2, sv, ww, label="sim play ON% (with balls)", color="#4c72b0")
        ax.bar(xx + ww / 2, hv, ww, label="HW empty warmup ON% (fitted thresh)", color="#dd8452")
        ax.set_xticks(xx)
        ax.set_xticklabels(plot_lab, rotation=45, ha="right")
        ax.set_ylabel("ON %")
        ax.set_title("Sim contact rhythm vs empty-motion false positives")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "04_onrate_vs_sim.png"), dpi=150)
        plt.close(fig)

        with open(os.path.join(out, "summary.txt"), "w") as f:
            f.write(f"warmup={warm_path}\nsim={sim_path}\nsteps={len(fsr)}\n\n")
            f.write(f"{'ch':>3} {'name':12} {'simON%':>8} {'emptyON%':>9} {'fsr_range':>10}\n")
            for i, c in enumerate(plot_ch):
                rng = ranges[i] if i < 12 else ranges[i]
                f.write(f"{c:3d} {plot_lab[i]:12} {sv[i]:8.1f} {hv[i]:9.1f} {rng:10.1f}\n")

    print("Saved plots ->", out)
    print("files:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
