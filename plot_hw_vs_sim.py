#!/usr/bin/env python3
"""Plot sim vs hardware policy rollouts (roto_2, 13-DOF / deploy_policy_new).

Hardware logs store q (13 control joints) and cmd (16 published joints incl. J1 mimics).
Sim logs store full 16-joint q and cmd. Skip the first --sim-skip sim steps (reset settle).

Presets:
  link   -> scripts/sim_policy_log_seed42.npz + hw_policy_log_link.npz
  padtac -> scripts/sim_policy_log_padtac_seed42.npz + hw_policy_log_padtac_new.npz
            (regenerate padtac sim with play.py + shadowlite_padtac if missing)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

JOINTS = [
    "FFJ4", "MFJ4", "RFJ4", "THJ5", "FFJ3", "MFJ3", "RFJ3", "THJ4",
    "FFJ2", "MFJ2", "RFJ2", "FFJ1", "MFJ1", "RFJ1", "THJ2", "THJ1",
]

TAC_NAMES = (
    "world forearm palm ffknuck mfknuck rfknuckle thbase "
    "ffprox mfprox rfprox thprox ffmid mfmid rfmid thhub "
    "ffdist mfdist rfdist thmid fftip mftip rftip thdist thtip"
).split()

# 13-d POLICY_JOINTS index -> 16-d PUBLISH_JOINTS slot (None = no /joint_states feedback)
Q13_TO_PUB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None, None, 11, 12]

# HW mux C0..C11 -> sim tactile ch (C5=rfmid, C6=palm, C11=ffmid after wire check)
FSR_CH = [10, 7, 4, 9, 5, 13, 2, 3, 8, 18, 12, 11]
BIOTAC_CH = [15, 16, 17, 22]
PLOT_CH = FSR_CH + BIOTAC_CH
BIOTAC_NAMES = {"ffdist", "mfdist", "rfdist", "thdist"}

PRESETS = {
    "link": {
        "sim": "scripts/sim_policy_log_seed42.npz",
        "hw": "hw_policy_log_link.npz",
        "out": "link_plots",
        "title": "Link policy (24-link tactile, shadow_touchlab_col)",
    },
    "padtac": {
        "sim": "scripts/sim_policy_log_padtac_seed42.npz",
        "hw": "hw_policy_log_padtac_best.npz",
        "out": "padtac_plots_best",
        "title": "PadTac policy (12 FSR pads)",
    },
    "padtac_bt_ft": {
        "sim": "scripts/sim_policy_log_padtac_bt_ft_seed42.npz",
        "hw": "hw_policy_log_padtacbt.npz",
        "out": "padtacbt_plots_0p3",
        "title": "PadTac+BT FT (HW speed 0.3)",
    },
    "padtac_bt_ft_20": {
        "sim": "scripts/sim_policy_log_padtac_bt_ft_seed42.npz",
        "hw": "hw_policy_log_padtacbt_20.npz",
        "out": "padtacbt_plots_0p2",
        "title": "PadTac+BT FT (HW speed 0.2)",
    },
    "padtac_bt_ft_100": {
        "sim": "scripts/sim_policy_log_padtac_bt_ft_seed42.npz",
        "hw": "hw_policy_log_padtacbt_100.npz",
        "out": "padtacbt_plots_1p0",
        "title": "PadTac+BT FT (HW full speed / no governor)",
    },
    "trial15_simtac": {
        "sim": "sim_policy_log_trial15_seed42.npz",
        "hw": "hw_policy_log_trial15_simtac.npz",
        "out": "trial15_simtac_plots",
        "title": "Trial15 Gate C (sim tactile + real prop, 60 Hz)",
    },
}


def expand_hw_q(q13: np.ndarray) -> np.ndarray:
    """Map 13-d measured joints to 16-d publish order; J1 slots are NaN (not subscribed)."""
    out = np.full((q13.shape[0], 16), np.nan, dtype=np.float32)
    for pub_i, q_i in enumerate(Q13_TO_PUB):
        if q_i is not None:
            out[:, pub_i] = q13[:, q_i]
    return out


def load_pair(sim_path: str, hw_path: str, sim_skip: int) -> dict:
    if not os.path.isfile(sim_path):
        raise FileNotFoundError(
            f"Sim log not found: {sim_path}\n"
            "Generate with play.py from roto_2/scripts/ (see plot_hw_vs_sim.py header)."
        )
    if not os.path.isfile(hw_path):
        raise FileNotFoundError(f"Hardware log not found: {hw_path}")

    sim = np.load(sim_path, allow_pickle=True)
    hw = np.load(hw_path, allow_pickle=True)

    sq = sim["q"][sim_skip:]
    scmd = sim["cmd"][sim_skip:]
    stac = sim["tac"][sim_skip:]
    sact = sim["act"][sim_skip:] if "act" in sim.files else None

    n = min(len(sq), len(hw["q"]))
    hcmd = hw["cmd"][:n]
    hq = expand_hw_q(hw["q"][:n])
    htac = hw["tac"][:n]
    hact = hw["act"][:n] if "act" in hw.files else None
    t_hw = hw["t"][:n] - hw["t"][0] if "t" in hw.files else np.arange(n) * 0.05

    return {
        "n": n,
        "sq": sq[:n],
        "scmd": scmd[:n],
        "stac": stac[:n],
        "sact": sact[:n] if sact is not None else None,
        "hq": hq,
        "hcmd": hcmd,
        "htac": htac,
        "hact": hact,
        "t_hw": t_hw,
        "hw": hw,
    }


def plot_all(data: dict, out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = data["n"]
    sq, scmd = data["sq"], data["scmd"]
    hq, hcmd = data["hq"], data["hcmd"]
    stac, htac = data["stac"], data["htac"]
    t = data["t_hw"]

    # --- joint trajectories ---
    fig, ax = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for j, a in enumerate(ax.flat):
        a.plot(t, scmd[:, j], "k--", lw=0.8, alpha=0.7, label="sim cmd")
        a.plot(t, sq[:, j], color="C0", alpha=0.8, label="sim q")
        a.plot(t, hcmd[:, j], color="C3", ls="--", lw=0.8, alpha=0.7, label="hw cmd")
        if not np.all(np.isnan(hq[:, j])):
            a.plot(t, hq[:, j], color="C1", alpha=0.8, label="hw q")
        a.set_title(JOINTS[j])
        a.grid(True, alpha=0.3)
    ax.flat[0].legend(fontsize=7)
    fig.suptitle(f"Joint trajectories: sim vs hardware — {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hw_vs_sim_joints.png"), dpi=150)
    plt.close(fig)

    # --- tracking error (only joints with HW feedback) ---
    sim_err = np.sqrt(((scmd - sq) ** 2).mean(0))
    hw_err = np.full(16, np.nan)
    for j in range(16):
        if not np.all(np.isnan(hq[:, j])):
            hw_err[j] = np.sqrt(((hcmd[:, j] - hq[:, j]) ** 2).mean())

    x = np.arange(16)
    w = 0.35
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(x - w / 2, np.degrees(sim_err), w, label="sim", color="#4c72b0")
    mask = ~np.isnan(hw_err)
    ax2.bar(x[mask] + w / 2, np.degrees(hw_err[mask]), w, label="hw", color="#dd8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels(JOINTS, rotation=45, ha="right")
    ax2.set_ylabel("RMS tracking error (deg)")
    ax2.set_title("Command vs achieved mismatch (hw: 13 joints with feedback)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "hw_vs_sim_tracking_err.png"), dpi=150)
    plt.close(fig2)

    # --- tactile bars (16 HW channels: 12 FSR + 4 BioTac) ---
    sim_freq = 100 * stac.mean(0)
    hw_freq = 100 * htac.mean(0)
    labels = [TAC_NAMES[c] for c in PLOT_CH]
    sv = [sim_freq[c] for c in PLOT_CH]
    hv = [hw_freq[c] for c in PLOT_CH]
    xch = np.arange(len(PLOT_CH))
    fig3, ax3 = plt.subplots(figsize=(18, 5))
    ax3.bar(xch - w / 2, sv, w, label="sim", color="#4c72b0")
    ax3.bar(xch + w / 2, hv, w, label="hw (FSR+BioTac)", color="#dd8452")
    for i, ch in enumerate(PLOT_CH):
        if ch in BIOTAC_CH:
            ax3.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="green")
    ax3.set_xticks(xch)
    ax3.set_xticklabels(labels, rotation=35, ha="right")
    ax3.set_ylabel("Contact frequency (%)")
    ax3.set_title("Tactile activation: sim vs hardware (12 FSR + 4 BioTac distals)")
    for i, (a, b) in enumerate(zip(sv, hv)):
        ax3.text(i - w / 2, a + 1, f"{a:.0f}%", ha="center", fontsize=7)
        ax3.text(i + w / 2, b + 1, f"{b:.0f}%", ha="center", fontsize=7)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "hw_vs_sim_tactile_bars.png"), dpi=150)
    plt.close(fig3)

    # --- tactile raster (all channels that fire in either) ---
    active = sorted(
        set(i for i in range(24) if stac[:, i].any())
        | set(i for i in range(24) if htac[:, i].any())
    )
    fig4, (a4s, a4h) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    a4s.imshow(stac[:, active].T, aspect="auto", cmap="Greys", interpolation="nearest")
    a4s.set_yticks(range(len(active)))
    a4s.set_yticklabels([TAC_NAMES[i] for i in active])
    a4s.set_title("Sim tactile raster")
    a4h.imshow(htac[:, active].T, aspect="auto", cmap="Greys", interpolation="nearest")
    a4h.set_yticks(range(len(active)))
    a4h.set_yticklabels([TAC_NAMES[i] for i in active])
    a4h.set_xlabel("timestep")
    a4h.set_title("Hardware tactile raster (FSR + BioTac)")
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "hw_vs_sim_tactile_raster.png"), dpi=150)
    plt.close(fig4)

    # --- policy action magnitude ---
    if data["sact"] is not None and data["hact"] is not None:
        fig6, ax6 = plt.subplots(figsize=(12, 3))
        ax6.plot(np.abs(data["sact"]).mean(1), label="sim |act| mean", alpha=0.9)
        ax6.plot(np.abs(data["hact"]).mean(1), label="hw |act| mean", alpha=0.9)
        ax6.set_xlabel("step")
        ax6.set_ylabel("|action| (per step, mean over 13 joints)")
        ax6.set_title("Policy action magnitude")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        fig6.savefig(os.path.join(out_dir, "hw_vs_sim_actions.png"), dpi=150)
        plt.close(fig6)

    # --- raw BioTac pdc ---
    hw = data["hw"]
    if "biotac_pdc" in hw.files:
        bp = hw["biotac_pdc"][:n]
        fig5, ax5 = plt.subplots(figsize=(14, 3))
        for i, name in zip([0, 1, 2, 4], ["ff", "mf", "rf", "th"]):
            ax5.plot(t, bp[:, i], label=name, alpha=0.8)
        ax5.set_ylabel("pdc")
        ax5.set_xlabel("time (s)")
        ax5.set_title("Raw BioTac pdc")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        fig5.savefig(os.path.join(out_dir, "hw_biotac_pdc.png"), dpi=150)
        plt.close(fig5)

    # --- console summary ---
    print(f"\n=== {title} ===")
    print(f"Compared {n} steps ({t[-1]:.1f}s HW)")
    print("\nRMS tracking error (deg):")
    for j, name in enumerate(JOINTS):
        hw_s = f"{np.degrees(hw_err[j]):5.1f}" if not np.isnan(hw_err[j]) else "  n/a"
        print(f"  {name:5s}  sim {np.degrees(sim_err[j]):5.1f}   hw {hw_s}")
    print("\nTactile activation (12 FSR + 4 BioTac):")
    for name, s, h in zip(labels, sv, hv):
        tag = " [BT]" if name in BIOTAC_NAMES else ""
        print(f"  {name:10s}  sim {s:5.1f}%   hw {h:5.1f}%{tag}")
    print(f"\nSaved plots -> {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sim vs hardware policy logs.")
    parser.add_argument(
        "--preset",
        choices=[
            "link", "padtac", "padtac_bt_ft", "padtac_bt_ft_20", "padtac_bt_ft_100",
            "trial15_simtac", "both", "all_bt_ft",
        ],
        default="link",
        help="Which policy pair to plot (default: link)",
    )
    parser.add_argument("--sim", type=str, default=None, help="Override sim .npz path")
    parser.add_argument("--hw", type=str, default=None, help="Override hw .npz path")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for PNGs")
    parser.add_argument(
        "--sim-skip",
        type=int,
        default=16,
        help="Skip first N sim steps (reset/settle hold; default 16)",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    if args.preset == "both":
        presets = ["link", "padtac"]
    elif args.preset == "all_bt_ft":
        presets = ["padtac_bt_ft", "padtac_bt_ft_20", "padtac_bt_ft_100"]
    else:
        presets = [args.preset]
    errors = []

    for name in presets:
        cfg = PRESETS[name]
        sim_path = os.path.join(root, args.sim or cfg["sim"])
        hw_path = os.path.join(root, args.hw or cfg["hw"])
        out_dir = os.path.join(root, args.out_dir or cfg["out"])

        if args.preset in ("both", "all_bt_ft") and args.sim:
            sim_path = os.path.join(root, cfg["sim"])
        if args.preset in ("both", "all_bt_ft") and args.hw:
            hw_path = os.path.join(root, cfg["hw"])
        if args.preset in ("both", "all_bt_ft") and args.out_dir:
            out_dir = os.path.join(root, cfg["out"])

        try:
            data = load_pair(sim_path, hw_path, args.sim_skip)
            plot_all(data, out_dir, cfg["title"])
        except FileNotFoundError as e:
            errors.append(str(e))
            print(f"[skip {name}] {e}", file=sys.stderr)

    if errors and len(errors) == len(presets):
        sys.exit(1)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
