#!/usr/bin/env python
"""Live FSR visualizer for Shadow Hand Lite pads (Arduino CSV on serial).

Compatible with the control laptop's older Python / matplotlib (same era as
PyTorch 1.4 legacy checkpoints). Python 3.5+; no f-strings, no 3.9 typing.

Shows per-pad (C0..C11):
  - raw ADC trace (amber)
  - resting baseline = p50 / median of empty capture (cyan dashed)
  - ON threshold = p99.5(empty) + MARGIN_ABS (red dotted)
  - binary lamp with hysteresis (ON above hi, OFF below lo = median + MARGIN_LO)

Matches Trial 15 warmup deploy threshing (not the old K*sigma path).

Controls
  B  -- re-capture empty envelope (~3 s; hand empty, cup pose preferred;
        move fingers gently if you want a motion-aware p99.5)
  [ / ] -- decrease / increase MARGIN_ABS (default 5.0 ADC)
  Q / Esc -- quit

Example
  python fsr_visualizer.py
  python fsr_visualizer.py --port /dev/ttyACM0 --margin 5.0
"""

from __future__ import print_function

import argparse
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch

try:
    import serial
except ImportError:
    raise SystemExit("Need pyserial: pip install pyserial")

from fsr_pad_map import FSR_PAD_ENTRIES, N_FSR, print_mapping_table

# (mux_label, site_name, sim_ch, usd_link, parent_link)
PAD_META = [
    (
        "C%d" % e["mux"],
        e["name"],
        e["sim_ch"],
        "rh_fsr_pad_%s" % e["usd"],
        e["parent"],
    )
    for e in FSR_PAD_ENTRIES
]
HISTORY = 250

# Same defaults as deploy_warmup_trial15.py
P_HI = 99.5
P_LO = 50.0
DEFAULT_MARGIN_ABS = 5.0
DEFAULT_MARGIN_LO = 2.0

# Visual theme
BG = "#12141a"
PANEL = "#1a1e28"
AX_BG = "#0e1016"
RAW = "#f0c14a"
BASE = "#5ec8ff"
THRESH = "#ff5c5c"
ON_COL = "#3dd68c"
OFF_COL = "#3a4050"
TEXT = "#e8eaef"
MUTED = "#8b93a7"


class FSRStream(object):
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()
        self.latest = np.zeros(N_FSR, dtype=np.float64)
        self.ok = False
        self.err = None
        self._stop = False
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
            # pyserial API changed: flushInput (old) vs reset_input_buffer (new)
            if hasattr(ser, "reset_input_buffer"):
                ser.reset_input_buffer()
            else:
                ser.flushInput()
            self.ok = True
        except Exception as e:
            self.err = str(e)
            return
        while not self._stop:
            line = ser.readline()
            if isinstance(line, bytes):
                line = line.decode("utf-8", "ignore")
            line = line.strip()
            if not line:
                continue
            try:
                vals = np.array([float(x) for x in line.split(",")], dtype=np.float64)
            except ValueError:
                continue
            if vals.shape[0] == N_FSR:
                with self.lock:
                    self.latest = vals

    def read(self):
        with self.lock:
            return self.latest.copy()


class Visualizer(object):
    def __init__(self, stream, margin_abs=5.0, margin_lo=2.0, baseline_s=3.0):
        self.stream = stream
        self.margin_abs = float(margin_abs)
        self.margin_lo = float(margin_lo)
        self.baseline_s = float(baseline_s)

        self.hist = [deque([0.0] * HISTORY, maxlen=HISTORY) for _ in range(N_FSR)]
        self.t0 = time.time()
        self.n_samples = 0

        self.baseline = np.zeros(N_FSR)   # p50 / median of empty capture
        self.p_hi = np.zeros(N_FSR)       # p99.5 of empty capture
        self.fsr_hi = np.zeros(N_FSR)     # p99.5 + margin_abs
        self.fsr_lo = np.zeros(N_FSR)     # median + margin_lo
        self.binary = np.zeros(N_FSR, dtype=bool)
        self.has_baseline = False
        self.status = (
            "Waiting for serial... press B after hand is empty to set p99.5 thresholds"
        )

        self._baseline_buf = []
        self._capturing_baseline = False
        self._baseline_until = 0.0

        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": "#2a3140",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 9,
        })

        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("FSR Live - C0..C11 (p99.5)")
        except Exception:
            pass

        # GridSpec(figure=...) is newer; fall back for old matplotlib
        try:
            gs = gridspec.GridSpec(
                4, 3, figure=self.fig,
                hspace=0.45, wspace=0.28,
                top=0.90, bottom=0.06, left=0.05, right=0.98,
            )
        except TypeError:
            gs = gridspec.GridSpec(4, 3, hspace=0.45, wspace=0.28)
            self.fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.98)

        self.axes = []
        self.raw_lines = []
        self.base_lines = []
        self.thr_lines = []
        self.lamps = []
        self.value_texts = []

        for i, (cid, name, ch, usd_link, parent) in enumerate(PAD_META):
            ax = self.fig.add_subplot(gs[i // 3, i % 3])
            ax.set_facecolor(AX_BG)
            for spine in ax.spines.values():
                spine.set_color("#2a3140")
            (raw_ln,) = ax.plot([], [], color=RAW, lw=1.8, label="raw", zorder=3)
            (base_ln,) = ax.plot([], [], color=BASE, lw=1.4, ls="--", label="median", zorder=2)
            (thr_ln,) = ax.plot([], [], color=THRESH, lw=1.4, ls=":", label="p99.5+margin", zorder=2)
            ax.set_xlim(0, HISTORY)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            ax.set_xticks([])

            lamp = FancyBboxPatch(
                (0.86, 0.72), 0.11, 0.22,
                transform=ax.transAxes,
                boxstyle="round,pad=0.02",
                facecolor=OFF_COL,
                edgecolor="#555c6e",
                linewidth=1.2,
                zorder=5,
                clip_on=False,
            )
            ax.add_patch(lamp)
            lamp_lbl = ax.text(
                0.915, 0.83, "OFF",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=TEXT, zorder=6,
            )

            ax.set_title(
                "%s | %s | ch %d | %s @ %s" % (cid, name, ch, usd_link, parent),
                loc="left", fontsize=8, color=TEXT, pad=6,
            )
            vtxt = ax.text(
                0.02, 0.95, "",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8, color=MUTED,
                family="monospace", zorder=6,
            )

            self.axes.append(ax)
            self.raw_lines.append(raw_ln)
            self.base_lines.append(base_ln)
            self.thr_lines.append(thr_ln)
            self.lamps.append((lamp, lamp_lbl))
            self.value_texts.append(vtxt)

        # labelcolor= is matplotlib >= 3.3; skip on older
        try:
            self.axes[0].legend(
                loc="upper left", fontsize=7, framealpha=0.25,
                facecolor=PANEL, edgecolor="#2a3140", labelcolor=TEXT,
                bbox_to_anchor=(0.0, -0.08), ncol=3,
            )
        except TypeError:
            self.axes[0].legend(
                loc="upper left", fontsize=7, framealpha=0.25,
                facecolor=PANEL, edgecolor="#2a3140",
                bbox_to_anchor=(0.0, -0.08), ncol=3,
            )

        self.header = self.fig.suptitle("", fontsize=13, color=TEXT, y=0.97, fontweight="bold")
        self.footer = self.fig.text(0.5, 0.015, "", ha="center", fontsize=9, color=MUTED)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # cache_frame_data is matplotlib >= 3.4
        try:
            self.ani = FuncAnimation(
                self.fig, self._update, interval=40, blit=False, cache_frame_data=False,
            )
        except TypeError:
            self.ani = FuncAnimation(
                self.fig, self._update, interval=40, blit=False,
            )

    def _refit_thresholds(self):
        """Recompute hi/lo from locked p50/p99.5 using current margins."""
        self.fsr_hi = self.p_hi + self.margin_abs
        self.fsr_lo = self.baseline + self.margin_lo
        # Keep hysteresis valid: lo must be strictly below hi.
        self.fsr_lo = np.minimum(self.fsr_lo, self.fsr_hi - 1.0)

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
            self.stream.stop()
        elif event.key in ("b", "B"):
            self._start_baseline_capture()
        elif event.key == "[":
            self.margin_abs = max(0.0, self.margin_abs - 0.5)
            if self.has_baseline:
                self._refit_thresholds()
            self.status = (
                "MARGIN_ABS = %.1f (lower = more sensitive)" % self.margin_abs
            )
        elif event.key == "]":
            self.margin_abs = min(50.0, self.margin_abs + 0.5)
            if self.has_baseline:
                self._refit_thresholds()
            self.status = (
                "MARGIN_ABS = %.1f (higher = less sensitive)" % self.margin_abs
            )

    def _start_baseline_capture(self):
        self._baseline_buf = []
        self._capturing_baseline = True
        self._baseline_until = time.time() + self.baseline_s
        self.status = (
            "Capturing empty envelope for %.1fs -- keep hand EMPTY "
            "(cup pose; gentle motion OK for motion-aware p99.5)..."
            % self.baseline_s
        )

    def _finish_baseline(self):
        self._capturing_baseline = False
        if len(self._baseline_buf) < 10:
            self.status = "Baseline capture failed (too few samples). Check serial."
            return
        s = np.array(self._baseline_buf)
        self.baseline = np.percentile(s, P_LO, axis=0)
        self.p_hi = np.percentile(s, P_HI, axis=0)
        self._refit_thresholds()
        self.has_baseline = True
        self.binary[:] = False
        self.status = (
            "Envelope locked | n=%d | hi = p%.1f + %.1f | lo = p%.1f + %.1f"
            " | tap pads to see spikes / binary ON"
            % (len(self._baseline_buf), P_HI, self.margin_abs, P_LO, self.margin_lo)
        )

    def _schmitt(self, raw):
        if not self.has_baseline:
            return
        for i in range(N_FSR):
            if raw[i] > self.fsr_hi[i]:
                self.binary[i] = True
            elif raw[i] < self.fsr_lo[i]:
                self.binary[i] = False

    def _update(self, _frame):
        if self.stream.err:
            self.header.set_text("Serial error: %s" % self.stream.err)
            self.header.set_color(THRESH)
            return []

        raw = self.stream.read()
        self.n_samples += 1

        if self._capturing_baseline:
            self._baseline_buf.append(raw)
            if time.time() >= self._baseline_until:
                self._finish_baseline()

        self._schmitt(raw)

        for i in range(N_FSR):
            self.hist[i].append(float(raw[i]))

        x = np.arange(HISTORY)
        for i in range(N_FSR):
            y = np.array(self.hist[i])
            self.raw_lines[i].set_data(x, y)

            if self.has_baseline:
                b = float(self.baseline[i])
                thr = float(self.fsr_hi[i])
            else:
                # live preview from rolling history before B locks
                b = float(np.percentile(y, P_LO)) if len(y) else 0.0
                thr = float(np.percentile(y, P_HI) + self.margin_abs) if len(y) else 1.0

            self.base_lines[i].set_data([0, HISTORY], [b, b])
            self.thr_lines[i].set_data([0, HISTORY], [thr, thr])

            ymin = min(y.min(), b, thr) - 0.05 * (abs(thr - b) + y.std() + 1)
            ymax = max(y.max(), b, thr) + 0.15 * (abs(thr - b) + y.std() + 1)
            if ymax - ymin < 10:
                mid = 0.5 * (ymin + ymax)
                ymin, ymax = mid - 5, mid + 5
            self.axes[i].set_ylim(ymin, ymax)

            spike = float(raw[i] - b) if self.has_baseline else 0.0
            on = bool(self.binary[i]) if self.has_baseline else False
            lamp, lamp_lbl = self.lamps[i]
            if on:
                lamp.set_facecolor(ON_COL)
                lamp.set_edgecolor("#1a8f5a")
                lamp_lbl.set_text("ON")
                lamp_lbl.set_color("#062816")
                self.axes[i].title.set_color(ON_COL)
            else:
                lamp.set_facecolor(OFF_COL)
                lamp.set_edgecolor("#555c6e")
                lamp_lbl.set_text("OFF")
                lamp_lbl.set_color(TEXT)
                self.axes[i].title.set_color(TEXT)

            self.value_texts[i].set_text(
                "raw %7.1f   med %7.1f   hi %7.1f   spike %+7.1f"
                % (raw[i], b, thr, spike)
            )
            self.value_texts[i].set_color(ON_COL if on else MUTED)

        n_on = int(self.binary.sum()) if self.has_baseline else 0
        self.header.set_text(
            "FSR Live  |  hi=p%.1f+%.1f  lo=p%.1f+%.1f  |  binary ON: %d/12  |  samples: %d"
            % (P_HI, self.margin_abs, P_LO, self.margin_lo, n_on, self.n_samples)
        )
        self.header.set_color(TEXT)
        self.footer.set_text(
            "%s    |    keys:  B = capture envelope   [ ] = MARGIN   Q = quit"
            % self.status
        )
        return []

    def show(self):
        def _auto_b():
            time.sleep(1.0)
            if self.stream.ok and (not self.has_baseline) and (not self._capturing_baseline):
                self._start_baseline_capture()

        t = threading.Thread(target=_auto_b)
        t.daemon = True
        t.start()
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Live FSR visualizer (C0-C11), p99.5 + margin thresholds"
    )
    ap.add_argument("--port", default="/dev/ttyACM0", help="Arduino serial port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN_ABS,
        help="ON margin: hi = p99.5(empty) + MARGIN (default %.1f)" % DEFAULT_MARGIN_ABS,
    )
    ap.add_argument(
        "--margin-lo", type=float, default=DEFAULT_MARGIN_LO,
        help="OFF margin: lo = median(empty) + MARGIN_LO (default %.1f)" % DEFAULT_MARGIN_LO,
    )
    ap.add_argument(
        "--baseline-s", type=float, default=3.0,
        help="Seconds of empty capture for p50/p99.5 envelope",
    )
    ap.add_argument(
        "--print-map", action="store_true",
        help="Print mux -> sim ch -> USD link table and exit (wire-check reference)",
    )
    # Keep old flags as aliases so existing laptop commands still work
    ap.add_argument("--k-hi", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--k-lo", type=float, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.print_map:
        print_mapping_table()
        return

    margin_abs = args.margin if args.k_hi is None else float(args.k_hi)
    margin_lo = args.margin_lo if args.k_lo is None else float(args.k_lo)

    stream = FSRStream(args.port, args.baud)
    stream.start()
    time.sleep(0.3)
    if stream.err:
        raise SystemExit("Cannot open %s: %s" % (args.port, stream.err))

    print("FSR visualizer running (p99.5 + margin).")
    print("Pad map (mux CSV -> sim ch -> USD link):")
    print_mapping_table()
    print("")
    print("  Amber  = raw")
    print("  Cyan   = median (p50) of empty capture")
    print("  Red    = ON threshold (p99.5 + %.1f)" % margin_abs)
    print("  Green lamp = binary ON (hysteresis; OFF below median + %.1f)" % margin_lo)
    print("Keys: B capture envelope | [ ] MARGIN | Q quit")
    viz = Visualizer(
        stream,
        margin_abs=margin_abs,
        margin_lo=margin_lo,
        baseline_s=args.baseline_s,
    )
    try:
        viz.show()
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
