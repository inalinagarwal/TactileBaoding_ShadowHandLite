#!/usr/bin/env python
"""Trial 15 deploy — J2 90↔100 remap + SPEED_FRAC ramp (NEW FILE ONLY).

This is a standalone experiment script. It does not edit deploy_warmup_trial15.py
or any other deploy code.

================================================================================
WHAT THIS FILE CHANGES vs plain trial15 warmup deploy
================================================================================
1) USE_J2_90_100_REMAP (flag) — linear J2 sim 100° ↔ HW 90°
   - Policy / sim think FFJ2/MFJ2/RFJ2 live in [0, J2_UPPER_SIM] ≈ 100° (1.745 rad).
   - Shadow Hand Lite plant only accepts [0, J2_UPPER_HW] ≈ 90° (1.571 rad).
   - True:  PUBLISH j2_hw = j2_sim * (90/100);
            OBS/ENC j2_sim = j2_hw * (100/90) so pos / last_command / pos_err
            share sim-radian space; gate uses remapped measured J2.
   - False: no compress/expand (legacy trial15 path for J2 radians).

2) CLIP_TO_PUB_LIMITS (flag) — extra plant clip after pack/(optional remap)
   - True  (default): np.clip(pub, PUB_LOWER, PUB_UPPER) before PD (legacy HW).
   - False: skip that second clip; affined+coupled(+optional J2-compress) cmd
     goes to PD more like sim (sim has no extra plant clip after scale/coupling).
   - With remap on, J2 is already ≤~90° after compress, so False mainly drops
     a safety net; ROS may still reject out-of-range targets — log cmd vs q.

3) SPEED_FRAC governor (flag) — pick mode at top of file
   Not a queue: every 60 Hz tick the policy still runs fresh; we only limit
   how far pub may step toward the new desire from prev_pub.
   - USE_SPEED_FRAC_RAMP = False → constant SPEED_FRAC (float or None).
     Examples: 0.5 (half rated speed), 1.0 (full rated), None (bang-bang).
   - USE_SPEED_FRAC_RAMP = True  → time schedule:
       Stage 0  (steps 0 .. RAMP_START-1):     SPEED_FRAC = 0.2  (slow start)
       Stage 1  (RAMP_START .. RAMP_END-1):     linear ramp 0.2 → 1.0
       Stage 2  (RAMP_END .. UNLIMITED_AT-1):   SPEED_FRAC = 1.0  (rated-speed cap)
       Stage 3  (step >= UNLIMITED_AT):          SPEED_FRAC = None (true bang-bang)

Note on tactile: empty-motion thresholds are frozen after warmup. If a pad's ADC
baseline drifts up after contact, hysteresis can stick ON (raw never falls below
lo). That is a separate sim↔HW gap from the joint flags above.

================================================================================
Flow (HAND EMPTY until Phase C) — same Gate-B warmup as trial15
================================================================================
  A) Replay Trial-15 sim achieved-q  WARMUP_REPEATS times @ 60 Hz.
     Log raw FSR (12) + BioTac PDC (5) every step.
     (Warmup open-loop still clips q to PUB_UPPER; no J2 expand on replay.)
  B) Fit per-channel thresholds from empty-motion envelope:
        hi = p99.5(raw) + 5 ADC
        lo = median + 2 (narrow pads) or hi - HYST_BAND_WIDE (wide envelopes)
     Hysteresis ON/OFF. mfprox live (FSR_MUTE_MUX empty after pad replace).
     Save WARMUP_NPZ for offline plots / sim comparison.
  C) Prompt to place balls in the cup.
  D) Closed-loop policy with REAL tactile + optional J2 remap / CLIP /
     SPEED_FRAC (ramp or fixed). Save POLICY_NPZ (includes per-step speed_frac).

Laptop files needed next to this script (or absolute paths below):
  sim_policy_log_trial15_seed42.npz
  best_agent_legacy_padtac_bt_scratch_trial15.pt  (or under /home/user/experiments/)
"""

from __future__ import print_function

import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll

import serial
import threading

# ============================================================
# PATHS / PROTOCOL
# ============================================================
REPLAY_Q_FILE = "sim_policy_log_trial15_seed42.npz"  # must have q (T,16) + joints
CHECKPOINT = "/home/user/experiments/best_agent_legacy_padtac_bt_scratch_trial15.pt"

WARMUP_REPEATS = 5
WARMUP_NPZ = "hw_warmup_trial15_j2remap_ramp_fsr.npz"
POLICY_NPZ = "hw_policy_log_trial15_j2remap_ramp.npz"

CONTROL_HZ = 60

# ----- Experiment flags (policy phase) -----
# J2 90↔100 linear remap (publish compress + obs/gate expand).
USE_J2_90_100_REMAP = True

# After affine + coupling + optional J2 compress, optionally clip 16-d pub to PUB_*.
# True  = legacy HW second clip (safety net).
# False = skip it — closer to sim PD path (no extra clip after scale/coupling).
CLIP_TO_PUB_LIMITS = True

# SPEED_FRAC: fraction of PUB_VEL_LIMIT as max |Δcmd| per tick.
# None = no governor (pub may jump fully to desire after optional clip).
#   USE_SPEED_FRAC_RAMP = True  → use the ramp schedule below
#   USE_SPEED_FRAC_RAMP = False → use fixed SPEED_FRAC for the whole run
USE_SPEED_FRAC_RAMP = True
SPEED_FRAC = 0.5   # used only when USE_SPEED_FRAC_RAMP is False
                   # try: 0.5  |  1.0  |  None

# Ramp schedule (used only when USE_SPEED_FRAC_RAMP is True)
# Units: control steps @ CONTROL_HZ (60 Hz → 60 steps = 1 s).
SPEED_FRAC_START = 0.2          # Stage 0: cautious start
SPEED_FRAC_PEAK = 1.0           # Stage 1→2: ramp up to rated-speed cap
RAMP_START_STEP = 5 * CONTROL_HZ   # 5 s at START, then begin ramp
RAMP_END_STEP = 20 * CONTROL_HZ    # finish ramp by 20 s (15 s linear ramp)
UNLIMITED_AT_STEP = 40 * CONTROL_HZ  # after 40 s: SPEED_FRAC = None

# Empty-motion envelope thresholds (not K*std — std blows up on spike pads):
#   hi = percentile(warmup, P_HI) + MARGIN_ABS
#   lo = percentile(warmup, P_LO) + MARGIN_LO   (narrow pads)
#   lo = hi - HYST_BAND_WIDE                    (wide empty envelopes)
P_HI = 99.5
P_LO = 50.0
MARGIN_ABS = 5.0   # ADC counts above empty p99.5 -> ON
MARGIN_LO = 2.0    # ADC counts above empty median -> release (narrow pads)
WIDE_ENVELOPE_ADC = 15.0
HYST_BAND_WIDE = 3.0
STD_FLOOR_BT = 1.0  # BioTac still uses a small noise floor for PDC
# Legacy K*std (unused for FSR now; kept for optional BT fallback / docs):
# K_HI, K_LO = 5.0, 2.0

# ============================================================
# MODEL
# ============================================================
OBS_DIM = 304
PROP_DIM = 52
NUM_J = 13
NUM_TACTILE = 24
OBS_STACK = 4


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 1024), nn.LayerNorm(1024), nn.ELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, NUM_J),
        )

    def forward(self, z):
        return self.policy_net(z)


# ============================================================
# JOINTS / COUPLING (same as deploy_policy_new)
# ============================================================
POLICY_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]
JOINT_LOWER = np.array(
    [-0.3491, -0.3491, -0.3491, -1.0472,
     -0.2618, -0.2618, -0.2618, 0.0,
     0.0, 0.0, 0.0, -0.6981, -0.2618],
    dtype=np.float32,
)
JOINT_UPPER = np.array(
    [0.3491, 0.3491, 0.3491, 1.0472,
     1.5708, 1.5708, 1.5708, 1.2217,
     1.7450, 1.7450, 1.7450, 0.6981, 1.5708],
    dtype=np.float32,
)
JOINT_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 2.0, 4.0],
    dtype=np.float32,
)

COUPLING_THETA = 0.785
J2_UPPER_SIM = 1.7450
J1_UPPER_SIM = 1.3960
GATE_J2_TOL = 0.035
CURL_J2_IDX = [8, 9, 10]

PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]
CTRL_NONCOUPLED = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
PUB_NONCOUPLED = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]
PUB_J2_SLOTS = [8, 9, 10]
PUB_J1_SLOTS = [11, 12, 13]

PUB_LOWER = np.array(
    [-0.3491, -0.3491, -0.3491, -1.0472,
     -0.2618, -0.2618, -0.2618, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     -0.6981, -0.2618],
    dtype=np.float32,
)
PUB_UPPER = np.array(
    [0.3491, 0.3491, 0.3491, 1.0472,
     1.5708, 1.5708, 1.5708, 1.2217,
     1.5708, 1.5708, 1.5708,
     1.5708, 1.5708, 1.5708,
     0.6981, 1.5708],
    dtype=np.float32,
)
PUB_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0,
     2.0, 2.0, 2.0,
     2.0, 4.0],
    dtype=np.float32,
)

# ----- J2 sim↔HW linear remap (FFJ2 / MFJ2 / RFJ2 only) -----
# Sim / policy J2 upper (JOINT_UPPER[8:11], coupling math): ~100°
# Hardware publish upper (PUB_UPPER[8:10]):                ~90°
J2_UPPER_HW = 1.5708  # 90°  — plant / PUB clip
# J2_UPPER_SIM already defined above (= 1.7450 ≈ 100°)
J2_SIM_TO_HW = J2_UPPER_HW / J2_UPPER_SIM   # compress before publish (~0.9)
J2_HW_TO_SIM = J2_UPPER_SIM / J2_UPPER_HW   # expand measured J2 for obs (~1.111)

# Mux C0..C11 -> 24-d channel (wire-checked map)
FSR_CHANNELS = [10, 7, 4, 9, 5, 13, 2, 3, 8, 18, 12, 11]
FSR_NAMES = [
    "C0_thprox", "C1_ffprox", "C2_mfknuckle", "C3_rfprox",
    "C4_rfknuckle", "C5_rfmid", "C6_palm", "C7_ffknuckle",
    "C8_mfprox", "C9_thmiddle", "C10_mfmid", "C11_ffmid",
]
N_FSR = 12
# Force these mux indices silent in the policy tactile vector (always 0).
# mfprox replaced — leave empty unless a channel is clearly broken again.
FSR_MUTE_MUX = []
BIOTAC_IDX = [0, 1, 2, 4]
BIOTAC_CH = [15, 16, 17, 22]
BIOTAC_NAMES = ["BT_ffdist", "BT_mfdist", "BT_rfdist", "BT_thdist"]
N_BIOTAC = len(BIOTAC_IDX)
USE_BIOTAC = True

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200


def unscale(x, lo, hi):
    return (2.0 * x - hi - lo) / (hi - lo)


def scale(a, lo, hi):
    return 0.5 * (a + 1.0) * (hi - lo) + lo


def j2_sim_to_hw(j2_sim):
    """Compress sim-space J2 (0..100°) → HW publish space (0..90°) if remap on."""
    x = np.asarray(j2_sim, dtype=np.float32)
    if not USE_J2_90_100_REMAP:
        return x
    return x * J2_SIM_TO_HW


def j2_hw_to_sim(j2_hw):
    """Expand HW-measured J2 (0..90°) → sim / policy space (0..100°) if remap on."""
    x = np.asarray(j2_hw, dtype=np.float32)
    if not USE_J2_90_100_REMAP:
        return x
    return x * J2_HW_TO_SIM


def joint_pos_sim():
    """13-d policy joint positions; J2 remapped into sim radians when flag on."""
    pos = current_joint_pos.copy()
    if USE_J2_90_100_REMAP:
        pos[CURL_J2_IDX] = j2_hw_to_sim(pos[CURL_J2_IDX])
    return pos


def speed_frac_at_step(step):
    """Return SPEED_FRAC for this policy tick (float or None).

    If USE_SPEED_FRAC_RAMP is False: always return fixed SPEED_FRAC.
    If True:
      Stage 0: hold SPEED_FRAC_START
      Stage 1: linear ramp START → PEAK over [RAMP_START_STEP, RAMP_END_STEP)
      Stage 2: hold SPEED_FRAC_PEAK
      Stage 3: None (unlimited) once step >= UNLIMITED_AT_STEP
    """
    if not USE_SPEED_FRAC_RAMP:
        return None if SPEED_FRAC is None else float(SPEED_FRAC)

    if step >= UNLIMITED_AT_STEP:
        return None  # Stage 3 — true no governor
    if step < RAMP_START_STEP:
        return float(SPEED_FRAC_START)  # Stage 0
    if step < RAMP_END_STEP:
        # Stage 1 — linear interpolate
        alpha = float(step - RAMP_START_STEP) / float(RAMP_END_STEP - RAMP_START_STEP)
        return float(SPEED_FRAC_START + alpha * (SPEED_FRAC_PEAK - SPEED_FRAC_START))
    return float(SPEED_FRAC_PEAK)  # Stage 2


def action_to_publish(action, meas_j2_sim, prev_pub, speed_frac):
    """Policy action → 16-d HW trajectory target.

    Stages inside this call:
      1) Unscale action → raw 13-d cmd in sim joint limits (incl. J2 up to 100°).
      2) Coupling: proxy J2 → (j2_cmd, j1_cmd) in sim space; gate with meas_j2_sim.
      3) Pack 16-d pub; optional COMPRESS J2 slots sim→HW (USE_J2_90_100_REMAP).
      4) Optional CLIP_TO_PUB_LIMITS: np.clip to PUB_LOWER/UPPER (extra plant clip).
      5) Optional slew governor: step toward desire by at most
         PUB_VEL_LIMIT * speed_frac / CONTROL_HZ  (skipped if speed_frac is None).
    Returns:
      pub     — what we send to the robot (HW radians)
      raw_cmd — 13-d sim-space desire (for last_command non-J2)
      j2_cmd  — 3-d sim-space J2 desire (for last_command J2; pre-compress)
    """
    # --- Stage 1: action → sim joint cmd ---
    raw_cmd = scale(action, JOINT_LOWER, JOINT_UPPER)
    proxy = raw_cmd[CURL_J2_IDX]
    j2_cmd = np.clip(proxy * (J2_UPPER_SIM / COUPLING_THETA), 0.0, J2_UPPER_SIM)
    j1_cmd = np.clip(
        (proxy - COUPLING_THETA) / (J2_UPPER_SIM - COUPLING_THETA) * J1_UPPER_SIM,
        0.0, J1_UPPER_SIM,
    )
    # --- Stage 2: underactuated J1 gate (meas_j2 in same space as opens_at) ---
    opens_at = J2_UPPER_SIM - GATE_J2_TOL
    gate = np.clip((np.asarray(meas_j2_sim, dtype=np.float32) - opens_at) / GATE_J2_TOL, 0.0, 1.0)
    j1_cmd = j1_cmd * gate

    # --- Stage 3: pack (+ optional J2 compress sim→HW) ---
    pub = np.empty(16, dtype=np.float32)
    pub[PUB_NONCOUPLED] = raw_cmd[CTRL_NONCOUPLED]
    pub[PUB_J2_SLOTS] = j2_sim_to_hw(j2_cmd)  # identity if remap off
    pub[PUB_J1_SLOTS] = j1_cmd

    # --- Stage 4: optional extra plant clip (sim has no equivalent) ---
    if CLIP_TO_PUB_LIMITS:
        pub = np.clip(pub, PUB_LOWER, PUB_UPPER)

    # --- Stage 5: slew governor on published cmd ---
    if prev_pub is not None and speed_frac is not None:
        max_delta = PUB_VEL_LIMIT * float(speed_frac) / CONTROL_HZ
        pub = prev_pub + np.clip(pub - prev_pub, -max_delta, max_delta)

    return pub.astype(np.float32), raw_cmd, j2_cmd


# ============================================================
# GLOBAL STATE
# ============================================================
current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
current_joint_pos16 = np.zeros(16, dtype=np.float32)
current_joint_vel16 = np.zeros(16, dtype=np.float32)
last_action = np.zeros(NUM_J, dtype=np.float32)
last_command = np.zeros(NUM_J, dtype=np.float32)
joint_ready = False
biotac_ready = False

prop_buffer = deque(maxlen=OBS_STACK)
tactile_buffer = deque(maxlen=OBS_STACK)

latest_fsr = np.zeros(N_FSR, dtype=np.float32)
fsr_lock = threading.Lock()
latest_biotac_pdc = np.full(5, np.nan, dtype=np.float32)
bt_lock = threading.Lock()

fsr_baseline = np.zeros(N_FSR, dtype=np.float32)
fsr_noise = np.ones(N_FSR, dtype=np.float32)
fsr_hi = np.zeros(N_FSR, dtype=np.float32)
fsr_lo = np.zeros(N_FSR, dtype=np.float32)
bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
bt_noise = np.ones(N_BIOTAC, dtype=np.float32)
bt_hi = np.zeros(N_BIOTAC, dtype=np.float32)
bt_lo = np.zeros(N_BIOTAC, dtype=np.float32)

fsr_state = np.zeros(N_FSR, dtype=bool)
bt_state = np.zeros(N_BIOTAC, dtype=bool)


def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    global current_joint_pos16, current_joint_vel16
    idx = {n: i for i, n in enumerate(msg.name)}
    try:
        for i, j in enumerate(POLICY_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        for i, j in enumerate(PUBLISH_JOINTS):
            current_joint_pos16[i] = msg.position[idx[j]]
            current_joint_vel16[i] = msg.velocity[idx[j]]
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


def serial_reader():
    global latest_fsr
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1.0)
    ser.reset_input_buffer()
    while not rospy.is_shutdown():
        line = ser.readline().decode(errors="ignore").strip()
        try:
            vals = np.array([float(x) for x in line.split(",")], dtype=np.float32)
        except ValueError:
            continue
        if vals.shape[0] == N_FSR:
            with fsr_lock:
                latest_fsr = vals


def biotac_cb(msg):
    global latest_biotac_pdc, biotac_ready
    with bt_lock:
        for i, t in enumerate(msg.tactiles):
            latest_biotac_pdc[i] = float(t.pdc)
    biotac_ready = True


def publish_target(pub, target, duration):
    msg = JointTrajectory()
    msg.joint_names = PUBLISH_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = target.tolist()
    pt.time_from_start = rospy.Duration(duration)
    msg.points.append(pt)
    pub.publish(msg)


def _lo_from_envelope(baseline, phi, hi):
    """Release threshold: tight band under hi for wide empty envelopes."""
    envelope = (phi - baseline).astype(np.float32)
    lo_narrow = (baseline + MARGIN_LO).astype(np.float32)
    lo_wide = (hi - HYST_BAND_WIDE).astype(np.float32)
    wide = envelope > WIDE_ENVELOPE_ADC
    lo = np.where(wide, lo_wide, lo_narrow).astype(np.float32)
    lo = np.minimum(lo, hi - 1.0).astype(np.float32)
    return lo, wide, envelope


def fit_thresholds_from_warmup(fsr_arr, bt_arr):
    """fsr_arr (N,12), bt_arr (N,5) with NaNs possible on BT.

    FSR: hi = p99.5(empty) + MARGIN_ABS.
    lo = median + MARGIN_LO on narrow pads; hi - HYST_BAND_WIDE on wide pads.
    BioTac PDC: same idea on valid samples.
    """
    global fsr_baseline, fsr_noise, fsr_hi, fsr_lo
    global bt_baseline, bt_noise, bt_hi, bt_lo

    fsr_baseline = np.percentile(fsr_arr, P_LO, axis=0).astype(np.float32)
    fsr_phi = np.percentile(fsr_arr, P_HI, axis=0).astype(np.float32)
    fsr_noise = np.maximum(fsr_arr.std(axis=0), 1e-6).astype(np.float32)  # logged only
    fsr_hi = (fsr_phi + MARGIN_ABS).astype(np.float32)
    fsr_lo, fsr_wide, fsr_env = _lo_from_envelope(fsr_baseline, fsr_phi, fsr_hi)

    bt_sel = bt_arr[:, BIOTAC_IDX]
    bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
    bt_noise = np.ones(N_BIOTAC, dtype=np.float32)
    bt_hi = np.zeros(N_BIOTAC, dtype=np.float32)
    bt_lo = np.zeros(N_BIOTAC, dtype=np.float32)
    for k in range(N_BIOTAC):
        col = bt_sel[:, k]
        valid = col[np.isfinite(col) & (col > 0)]
        if valid.size < 10:
            rospy.logwarn("BioTac tip idx %d: few valid warmup samples (%d)", BIOTAC_IDX[k], valid.size)
            bt_baseline[k] = 0.0
            bt_noise[k] = STD_FLOOR_BT
            bt_hi[k] = MARGIN_ABS
            bt_lo[k] = 0.0
        else:
            bt_baseline[k] = float(np.percentile(valid, P_LO))
            bt_noise[k] = max(float(valid.std()), STD_FLOOR_BT)
            phi = float(np.percentile(valid, P_HI))
            bt_hi[k] = float(phi + MARGIN_ABS)
            lo_arr, _, _ = _lo_from_envelope(
                np.array([bt_baseline[k]], dtype=np.float32),
                np.array([phi], dtype=np.float32),
                np.array([bt_hi[k]], dtype=np.float32),
            )
            bt_lo[k] = float(lo_arr[0])

    rospy.loginfo(
        "FSR thresh mode=p%.1f+%.1f / lo=med+%.1f or hi-%.1f if env>%.1f",
        P_HI, MARGIN_ABS, MARGIN_LO, HYST_BAND_WIDE, WIDE_ENVELOPE_ADC,
    )
    rospy.loginfo("FSR baseline(med)=%s", fsr_baseline)
    rospy.loginfo("FSR empty_envelope(p99.5-med)=%s", fsr_env)
    rospy.loginfo("FSR wide_pads=%s (%s)", np.where(fsr_wide)[0].tolist(),
                  [FSR_NAMES[i] for i in np.where(fsr_wide)[0]])
    rospy.loginfo("FSR hi=%s", fsr_hi)
    rospy.loginfo("FSR lo=%s", fsr_lo)
    rospy.loginfo("FSR_MUTE_MUX=%s (%s)", FSR_MUTE_MUX,
                  [FSR_NAMES[i] for i in FSR_MUTE_MUX])
    rospy.loginfo("BioTac baseline=%s hi=%s", bt_baseline, bt_hi)


def apply_fsr_mute(fsr_on):
    """Zero muted mux channels (e.g. broken C8/mfprox)."""
    out = fsr_on.astype(np.float32).copy()
    for mux_i in FSR_MUTE_MUX:
        out[mux_i] = 0.0
    return out


def read_tactile():
    global fsr_state, bt_state
    with fsr_lock:
        fsr_vals = latest_fsr.copy()
    fsr_state = np.where(
        fsr_vals > fsr_hi,
        True,
        np.where(fsr_vals < fsr_lo, False, fsr_state),
    )
    with bt_lock:
        bt_vals = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
    for k in range(N_BIOTAC):
        v = bt_vals[k]
        if not np.isfinite(v) or v < 0:
            continue
        if v > bt_hi[k]:
            bt_state[k] = True
        elif v < bt_lo[k]:
            bt_state[k] = False
    t = np.zeros(NUM_TACTILE, dtype=np.float32)
    fsr_on = apply_fsr_mute(fsr_state.astype(np.float32))
    t[FSR_CHANNELS] = np.maximum(t[FSR_CHANNELS], fsr_on)
    if USE_BIOTAC:
        for k, ch in enumerate(BIOTAC_CH):
            t[ch] = max(t[ch], float(bt_state[k]))
    return t


def build_prop():
    """52-d proprio; J2 HW→sim only when USE_J2_90_100_REMAP."""
    pos_sim = joint_pos_sim()
    pos_norm = unscale(pos_sim, JOINT_LOWER, JOINT_UPPER)
    vel = current_joint_vel.copy()
    if USE_J2_90_100_REMAP:
        vel[CURL_J2_IDX] = vel[CURL_J2_IDX] * J2_HW_TO_SIM
    vel_norm = vel / JOINT_VEL_LIMIT
    error = last_command - pos_sim
    return np.concatenate([pos_norm, vel_norm, error, last_action]).astype(np.float32)


def run_q_replay(pub, rec_q, episode_id, log):
    """One open-loop replay of sim achieved q. Appends to log dicts."""
    rate = rospy.Rate(CONTROL_HZ)
    for q_sim in rec_q:
        if rospy.is_shutdown():
            break
        target = np.clip(q_sim, PUB_LOWER, PUB_UPPER)
        publish_target(pub, target, 1.0 / CONTROL_HZ)

        log["t"].append(rospy.get_time())
        log["episode"].append(episode_id)
        log["cmd"].append(target.copy())
        log["q"].append(current_joint_pos16.copy())
        with fsr_lock:
            log["fsr"].append(latest_fsr.copy())
        with bt_lock:
            log["biotac_pdc"].append(latest_biotac_pdc.copy())
        rate.sleep()


def main():
    global last_action, last_command, fsr_state, bt_state

    rospy.init_node("deploy_warmup_trial15_j2remap_ramp")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
    rospy.Subscriber("/joint_states", JointState, joint_callback)
    rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states OK.")

    rospy.loginfo("Waiting for /rh/tactile ...")
    while not biotac_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("BioTac OK.")

    th = threading.Thread(target=serial_reader)
    th.daemon = True
    th.start()
    rospy.sleep(0.5)

    # ----- load Trial-15 q replay -----
    data = np.load(REPLAY_Q_FILE, allow_pickle=True)
    rec_q = data["q"].astype(np.float32)
    assert rec_q.ndim == 2 and rec_q.shape[1] == 16, rec_q.shape
    if "joints" in data.files:
        assert list(data["joints"]) == PUBLISH_JOINTS, (list(data["joints"]), PUBLISH_JOINTS)
    rospy.loginfo("Loaded %s  T=%d for warmup x%d", REPLAY_Q_FILE, len(rec_q), WARMUP_REPEATS)

    input(
        "HAND EMPTY — cup pose ready. Press Enter to start %d x q-replay warmup..."
        % WARMUP_REPEATS
    )

    # Move to first frame, settle
    first = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    publish_target(pub, first, 2.0)
    rospy.sleep(3.0)

    warmup = {
        "t": [], "episode": [], "cmd": [], "q": [],
        "fsr": [], "biotac_pdc": [],
    }
    for ep in range(WARMUP_REPEATS):
        rospy.loginfo("Warmup replay %d/%d", ep + 1, WARMUP_REPEATS)
        run_q_replay(pub, rec_q, ep, warmup)
        # brief pause between repeats (hold last pose)
        rospy.sleep(0.3)

    fsr_arr = np.asarray(warmup["fsr"], dtype=np.float32)
    bt_arr = np.asarray(warmup["biotac_pdc"], dtype=np.float32)
    fit_thresholds_from_warmup(fsr_arr, bt_arr)

    # Approximate binary ON rates under fitted thresholds (causal hysteresis replay)
    fsr_state[:] = False
    bt_state[:] = False
    tac_warmup = []
    for i in range(len(fsr_arr)):
        # feed hysteresis as if streaming
        fsr_vals = fsr_arr[i]
        fsr_state[:] = np.where(
            fsr_vals > fsr_hi, True, np.where(fsr_vals < fsr_lo, False, fsr_state)
        )
        bt_row = bt_arr[i, BIOTAC_IDX]
        for k in range(N_BIOTAC):
            v = bt_row[k]
            if not np.isfinite(v) or v < 0:
                continue
            if v > bt_hi[k]:
                bt_state[k] = True
            elif v < bt_lo[k]:
                bt_state[k] = False
        tvec = np.zeros(NUM_TACTILE, dtype=np.float32)
        fsr_on = apply_fsr_mute(fsr_state.astype(np.float32))
        tvec[FSR_CHANNELS] = fsr_on
        if USE_BIOTAC:
            for k, ch in enumerate(BIOTAC_CH):
                tvec[ch] = float(bt_state[k])
        tac_warmup.append(tvec.copy())
    tac_warmup = np.asarray(tac_warmup, dtype=np.float32)

    np.savez(
        WARMUP_NPZ,
        t=np.asarray(warmup["t"]),
        episode=np.asarray(warmup["episode"], dtype=np.int32),
        cmd=np.asarray(warmup["cmd"], dtype=np.float32),
        q=np.asarray(warmup["q"], dtype=np.float32),
        fsr=fsr_arr,
        biotac_pdc=bt_arr,
        tac_bin=tac_warmup,
        fsr_names=np.array(FSR_NAMES),
        biotac_names=np.array(BIOTAC_NAMES),
        fsr_channels=np.array(FSR_CHANNELS, dtype=np.int32),
        biotac_channels=np.array(BIOTAC_CH, dtype=np.int32),
        biotac_idx=np.array(BIOTAC_IDX, dtype=np.int32),
        fsr_baseline=fsr_baseline,
        fsr_noise=fsr_noise,
        fsr_hi=fsr_hi,
        fsr_lo=fsr_lo,
        bt_baseline=bt_baseline,
        bt_noise=bt_noise,
        bt_hi=bt_hi,
        bt_lo=bt_lo,
        P_HI=np.float32(P_HI),
        P_LO=np.float32(P_LO),
        MARGIN_ABS=np.float32(MARGIN_ABS),
        MARGIN_LO=np.float32(MARGIN_LO),
        WIDE_ENVELOPE_ADC=np.float32(WIDE_ENVELOPE_ADC),
        HYST_BAND_WIDE=np.float32(HYST_BAND_WIDE),
        thresh_mode=np.array("p99_5_plus_margin_hysteresis_wide_tight"),
        fsr_mute_mux=np.array(FSR_MUTE_MUX, dtype=np.int32),
        joints=PUBLISH_JOINTS,
        replay_file=REPLAY_Q_FILE,
        warmup_repeats=np.int32(WARMUP_REPEATS),
        speed_frac=np.float32(
            -1.0 if USE_SPEED_FRAC_RAMP else (-1.0 if SPEED_FRAC is None else SPEED_FRAC)
        ),
        use_speed_frac_ramp=np.bool_(USE_SPEED_FRAC_RAMP),
        clip_to_pub_limits=np.bool_(CLIP_TO_PUB_LIMITS),
        use_j2_90_100_remap=np.bool_(USE_J2_90_100_REMAP),
        j2_upper_sim=np.float32(J2_UPPER_SIM),
        j2_upper_hw=np.float32(J2_UPPER_HW),
        note=np.array(
            "warmup open-loop unchanged; flags: J2 remap, CLIP_TO_PUB_LIMITS, SPEED_FRAC"
        ),
    )
    rospy.loginfo("Saved warmup sensors -> %s  (%d steps)", WARMUP_NPZ, len(fsr_arr))
    rospy.loginfo("Empty-motion binary ON%% (should be low):")
    for i, name in enumerate(FSR_NAMES):
        rospy.loginfo("  %s  ON=%.1f%%  med=%.1f hi=%.1f",
                      name, 100.0 * tac_warmup[:, FSR_CHANNELS[i]].mean(),
                      fsr_baseline[i], fsr_hi[i])
    for k, name in enumerate(BIOTAC_NAMES):
        rospy.loginfo("  %s  ON=%.1f%%  med=%.1f hi=%.1f",
                      name, 100.0 * tac_warmup[:, BIOTAC_CH[k]].mean(),
                      bt_baseline[k], bt_hi[k])

    # Hold last warmup pose while user places balls
    if warmup["cmd"]:
        publish_target(pub, np.asarray(warmup["cmd"][-1], dtype=np.float32), 1.0)

    input("Warmup done. PLACE BALLS in cup, then press Enter to start POLICY...")

    # ----- load policy -----
    rospy.loginfo("Loading checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    encoder, policy = Encoder(), Policy()
    e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    rospy.loginfo("encoder missing=%s unexpected=%s", e_res.missing_keys, e_res.unexpected_keys)
    policy.load_state_dict(
        {k: v for k, v in ckpt["policy"].items() if k != "log_std_parameter"},
        strict=True,
    )
    encoder.eval()
    policy.eval()

    fsr_state[:] = False
    bt_state[:] = False
    # Seed last_command in sim space (J2 expanded) so first pos_err is ~0.
    last_command[:] = joint_pos_sim()
    last_action[:] = 0.0
    p0, t0 = build_prop(), read_tactile()
    prop_buffer.clear()
    tactile_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())
        tactile_buffer.append(t0.copy())

    rospy.sleep(1.0)
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = None
    step = 0
    rec = {
        "t": [], "q": [], "q_sim": [], "cmd": [], "tac": [],
        "fsr": [], "biotac_pdc": [], "act": [], "speed_frac": [],
    }
    if USE_SPEED_FRAC_RAMP:
        rospy.loginfo(
            "Policy @ %d Hz | J2_REMAP=%s | CLIP_TO_PUB=%s | SPEED_FRAC RAMP: "
            "%.2f for %ds → ramp to %.2f by %ds → hold → None after %ds",
            CONTROL_HZ, USE_J2_90_100_REMAP, CLIP_TO_PUB_LIMITS,
            SPEED_FRAC_START, RAMP_START_STEP // CONTROL_HZ,
            SPEED_FRAC_PEAK, RAMP_END_STEP // CONTROL_HZ,
            UNLIMITED_AT_STEP // CONTROL_HZ,
        )
    else:
        rospy.loginfo(
            "Policy @ %d Hz | J2_REMAP=%s | CLIP_TO_PUB=%s | SPEED_FRAC FIXED: %s",
            CONTROL_HZ, USE_J2_90_100_REMAP, CLIP_TO_PUB_LIMITS,
            "None (unlimited)" if SPEED_FRAC is None else ("%.3f" % float(SPEED_FRAC)),
        )

    try:
        while not rospy.is_shutdown():
            # ----- Stage A: build obs (prop uses sim-space J2) -----
            prop_buffer.append(build_prop())
            tactile_buffer.append(read_tactile())
            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            # ----- Stage B: policy forward -----
            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)

            # ----- Stage C: SPEED_FRAC for this tick (ramp schedule or fixed) -----
            sf = speed_frac_at_step(step)
            if USE_SPEED_FRAC_RAMP and step in (
                0, RAMP_START_STEP, RAMP_END_STEP, UNLIMITED_AT_STEP
            ):
                rospy.loginfo(
                    "SPEED_FRAC stage change @ step=%d (t=%.1fs): %s",
                    step, step / float(CONTROL_HZ),
                    "None (unlimited)" if sf is None else ("%.3f" % sf),
                )

            # ----- Stage D: action → pub (J2 compress + optional slew) -----
            meas_j2_sim = j2_hw_to_sim(current_joint_pos[CURL_J2_IDX])
            pub_target, raw_cmd, j2_cmd = action_to_publish(
                action, meas_j2_sim, prev_pub, sf
            )
            prev_pub = pub_target
            publish_target(pub, pub_target, 1.0 / CONTROL_HZ)

            # ----- Stage E: log + update encoder memory in sim space -----
            rec["t"].append(rospy.get_time())
            rec["q"].append(current_joint_pos.copy())          # raw HW 13-d
            rec["q_sim"].append(joint_pos_sim())               # remapped for debug
            rec["cmd"].append(pub_target.copy())               # what plant got
            rec["tac"].append(tactile_buffer[-1].copy())
            with fsr_lock:
                rec["fsr"].append(latest_fsr.copy())
            with bt_lock:
                rec["biotac_pdc"].append(latest_biotac_pdc.copy())
            rec["act"].append(action.copy())
            rec["speed_frac"].append(-1.0 if sf is None else float(sf))

            last_action[:] = action
            last_command[:] = raw_cmd
            last_command[CURL_J2_IDX] = j2_cmd  # sim-space J2 (pre-compress if remap on)
            step += 1
            rate.sleep()
    finally:
        if rec["t"]:
            np.savez(
                POLICY_NPZ,
                **{k: np.array(v) for k, v in rec.items()},
                fsr_baseline=fsr_baseline,
                fsr_hi=fsr_hi,
                fsr_lo=fsr_lo,
                bt_baseline=bt_baseline,
                bt_hi=bt_hi,
                bt_lo=bt_lo,
                warmup_file=WARMUP_NPZ,
                j2_upper_sim=np.float32(J2_UPPER_SIM),
                j2_upper_hw=np.float32(J2_UPPER_HW),
                j2_sim_to_hw=np.float32(J2_SIM_TO_HW),
                use_j2_90_100_remap=np.bool_(USE_J2_90_100_REMAP),
                use_speed_frac_ramp=np.bool_(USE_SPEED_FRAC_RAMP),
                clip_to_pub_limits=np.bool_(CLIP_TO_PUB_LIMITS),
                speed_frac_fixed=np.float32(
                    -1.0 if SPEED_FRAC is None else float(SPEED_FRAC)
                ),
                speed_frac_start=np.float32(SPEED_FRAC_START),
                speed_frac_peak=np.float32(SPEED_FRAC_PEAK),
                ramp_start_step=np.int32(RAMP_START_STEP),
                ramp_end_step=np.int32(RAMP_END_STEP),
                unlimited_at_step=np.int32(UNLIMITED_AT_STEP),
            )
            rospy.loginfo("saved %s: %d steps", POLICY_NPZ, len(rec["t"]))


if __name__ == "__main__":
    main()
