#!/usr/bin/env python
"""Live FSR visualizer for Shadow Hand Lite pads (Arduino CSV on serial).

Compatible with the control laptop's older Python / matplotlib (same era as
PyTorch 1.4 legacy checkpoints). Python 3.5+; no f-strings, no 3.9 typing.

Shows per-pad (C0..C11):
  - raw ADC trace (amber)
  - resting baseline (cyan dashed)
  - ON threshold = baseline + K_HI * noise (red dotted)
  - binary lamp that lights the moment Schmitt trips ON

Same 12-channel order / names as deploy_policy_new.py.

Controls
  B  -- re-capture baseline+noise (~1.5 s; hand empty, cup pose preferred)
  [ / ] -- decrease / increase K_HI (deploy default 5.0)
  Q / Esc -- quit

Example
  python fsr_visualizer.py
  python fsr_visualizer.py --port /dev/ttyACM0 --k-hi 5.0
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

# ---- pad map (must match deploy_policy_new.py) ----
# Wire-checked mux order (C5=rfmid, C6=palm, C11=ffmid); matches deploy_policy_new.py
PAD_META = [
    ("C0", "thprox", 10),
    ("C1", "ffprox", 7),
    ("C2", "mfknuckle", 4),
    ("C3", "rfprox", 9),
    ("C4", "rfknuckle", 5),
    ("C5", "rfmid", 13),
    ("C6", "palm", 2),
    ("C7", "ffknuckle", 3),
    ("C8", "mfprox", 8),
    ("C9", "thmiddle", 18),
    ("C10", "mfmid", 12),
    ("C11", "ffmid", 11),
]
N_FSR = 12
HISTORY = 250

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
    def __init__(self, stream, k_hi=5.0, k_lo=2.0, baseline_s=1.5):
        self.stream = stream
        self.k_hi = float(k_hi)
        self.k_lo = float(k_lo)
        self.baseline_s = float(baseline_s)

        self.hist = [deque([0.0] * HISTORY, maxlen=HISTORY) for _ in range(N_FSR)]
        self.t0 = time.time()
        self.n_samples = 0

        self.baseline = np.zeros(N_FSR)
        self.noise = np.ones(N_FSR)
        self.binary = np.zeros(N_FSR, dtype=bool)
        self.has_baseline = False
        self.status = "Waiting for serial... press B after hand is empty to set baseline"

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
            self.fig.canvas.manager.set_window_title("FSR Live - C0..C11")
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

        for i, (cid, name, ch) in enumerate(PAD_META):
            ax = self.fig.add_subplot(gs[i // 3, i % 3])
            ax.set_facecolor(AX_BG)
            for spine in ax.spines.values():
                spine.set_color("#2a3140")
            (raw_ln,) = ax.plot([], [], color=RAW, lw=1.8, label="raw", zorder=3)
            (base_ln,) = ax.plot([], [], color=BASE, lw=1.4, ls="--", label="baseline", zorder=2)
            (thr_ln,) = ax.plot([], [], color=THRESH, lw=1.4, ls=":", label="ON threshold", zorder=2)
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
                "%s  |  %s  |  sim ch %d" % (cid, name, ch),
                loc="left", fontsize=10, color=TEXT, pad=6,
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

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
            self.stream.stop()
        elif event.key in ("b", "B"):
            self._start_baseline_capture()
        elif event.key == "[":
            self.k_hi = max(1.0, self.k_hi - 0.5)
            self.status = "K_HI = %.1f (lower = more sensitive)" % self.k_hi
        elif event.key == "]":
            self.k_hi = min(20.0, self.k_hi + 0.5)
            self.status = "K_HI = %.1f (higher = less sensitive)" % self.k_hi

    def _start_baseline_capture(self):
        self._baseline_buf = []
        self._capturing_baseline = True
        self._baseline_until = time.time() + self.baseline_s
        self.status = (
            "Capturing baseline for %.1fs -- keep hand EMPTY (cup pose preferred)..."
            % self.baseline_s
        )

    def _finish_baseline(self):
        self._capturing_baseline = False
        if len(self._baseline_buf) < 10:
            self.status = "Baseline capture failed (too few samples). Check serial."
            return
        s = np.array(self._baseline_buf)
        self.baseline = s.mean(0)
        self.noise = s.std(0) + 1e-6
        self.has_baseline = True
        self.binary[:] = False
        self.status = (
            "Baseline locked | mean noise=%.2f | threshold = baseline + %.1f*sigma"
            " | tap pads to see spikes / binary ON"
            % (self.noise.mean(), self.k_hi)
        )

    def _schmitt(self, raw):
        if not self.has_baseline:
            return
        hi = self.baseline + self.k_hi * self.noise
        lo = self.baseline + self.k_lo * self.noise
        for i in range(N_FSR):
            if raw[i] > hi[i]:
                self.binary[i] = True
            elif raw[i] < lo[i]:
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
                n = float(self.noise[i])
            else:
                b = float(y.mean()) if len(y) else 0.0
                n = max(float(y.std()), 1.0)
            thr = b + self.k_hi * n

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
                "raw %7.1f   base %7.1f   thr %7.1f   spike %+7.1f"
                % (raw[i], b, thr, spike)
            )
            self.value_texts[i].set_color(ON_COL if on else MUTED)

        n_on = int(self.binary.sum()) if self.has_baseline else 0
        self.header.set_text(
            "FSR Live Visualizer  |  K_HI=%.1f  K_LO=%.1f  |  binary ON: %d/12  |  samples: %d"
            % (self.k_hi, self.k_lo, n_on, self.n_samples)
        )
        self.header.set_color(TEXT)
        self.footer.set_text(
            "%s    |    keys:  B = set baseline   [ ] = K_HI   Q = quit" % self.status
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
    ap = argparse.ArgumentParser(description="Live FSR visualizer (C0-C11)")
    ap.add_argument("--port", default="/dev/ttyACM0", help="Arduino serial port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--k-hi", type=float, default=5.0,
                    help="ON threshold = base + K_HI * sigma (deploy default 5)")
    ap.add_argument("--k-lo", type=float, default=2.0,
                    help="OFF threshold = base + K_LO * sigma")
    ap.add_argument("--baseline-s", type=float, default=1.5,
                    help="Seconds of empty capture for baseline")
    args = ap.parse_args()

    stream = FSRStream(args.port, args.baud)
    stream.start()
    time.sleep(0.3)
    if stream.err:
        raise SystemExit("Cannot open %s: %s" % (args.port, stream.err))

    print("FSR visualizer running.")
    print("  Amber  = raw")
    print("  Cyan   = baseline (rest)")
    print("  Red    = ON threshold (baseline + K_HI*sigma)")
    print("  Green lamp = binary ON (same Schmitt idea as deploy)")
    print("Keys: B baseline | [ ] K_HI | Q quit")
    viz = Visualizer(stream, k_hi=args.k_hi, k_lo=args.k_lo, baseline_s=args.baseline_s)
    try:
        viz.show()
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
