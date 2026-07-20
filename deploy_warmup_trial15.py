#!/usr/bin/env python
"""Trial 15 deploy with empty-motion tactile warmup (Gate B style).

Flow (HAND EMPTY until Phase C):
  A) Replay Trial-15 sim achieved-q  WARMUP_REPEATS times @ 60 Hz.
     Log raw FSR (12) + BioTac PDC (5) every step.
  B) Fit per-channel thresholds from empty-motion envelope:
        hi = p99.5(raw) + 5 ADC, lo = median + 2 (hysteresis).
     C8/mfprox is muted (always 0) until you replace the pad — see FSR_MUTE_MUX.
     Save hw_warmup_trial15_fsr.npz for offline plots / sim comparison.
  C) Prompt to place balls in the cup.
  D) Closed-loop policy with REAL tactile using those thresholds.
     Save hw_policy_log_trial15_warmupcal.npz.

Does NOT modify deploy_policy_new.py.

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
WARMUP_NPZ = "hw_warmup_trial15_fsr.npz"
POLICY_NPZ = "hw_policy_log_trial15_warmupcal.npz"

CONTROL_HZ = 60
SPEED_FRAC = None  # governor off

# Empty-motion envelope thresholds (not K*std — std blows up on spike pads):
#   hi = percentile(warmup, P_HI) + MARGIN_ABS
#   lo = percentile(warmup, P_LO) + MARGIN_LO   (with lo clipped below hi)
# Contact when raw rises above the empty-motion ceiling (+ small ADC margin).
P_HI = 99.5
P_LO = 50.0
MARGIN_ABS = 5.0   # ADC counts above empty p99.5 -> ON
MARGIN_LO = 2.0    # ADC counts above empty median -> release (hysteresis)
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

# Mux C0..C11 -> 24-d channel (wire-checked map)
FSR_CHANNELS = [10, 7, 4, 9, 5, 13, 2, 3, 8, 18, 12, 11]
FSR_NAMES = [
    "C0_thprox", "C1_ffprox", "C2_mfknuckle", "C3_rfprox",
    "C4_rfknuckle", "C5_rfmid", "C6_palm", "C7_ffknuckle",
    "C8_mfprox", "C9_thmiddle", "C10_mfmid", "C11_ffmid",
]
N_FSR = 12
# Force these mux indices silent in the policy tactile vector (always 0).
# C8/mfprox: empty warmup already spans ~0–968; replace pad later, then
# remove 8 from this list (or empty the list) to re-enable.
FSR_MUTE_MUX = [8]  # C8_mfprox -> ch 8
# FSR_MUTE_MUX = []  # <-- uncomment this (and comment the line above) after replacing C8
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


def action_to_publish(action, meas_j2, prev_pub):
    raw_cmd = scale(action, JOINT_LOWER, JOINT_UPPER)
    proxy = raw_cmd[CURL_J2_IDX]
    j2_cmd = np.clip(proxy * (J2_UPPER_SIM / COUPLING_THETA), 0.0, J2_UPPER_SIM)
    j1_cmd = np.clip(
        (proxy - COUPLING_THETA) / (J2_UPPER_SIM - COUPLING_THETA) * J1_UPPER_SIM,
        0.0, J1_UPPER_SIM,
    )
    opens_at = J2_UPPER_SIM - GATE_J2_TOL
    gate = np.clip((np.asarray(meas_j2, dtype=np.float32) - opens_at) / GATE_J2_TOL, 0.0, 1.0)
    j1_cmd = j1_cmd * gate

    pub = np.empty(16, dtype=np.float32)
    pub[PUB_NONCOUPLED] = raw_cmd[CTRL_NONCOUPLED]
    pub[PUB_J2_SLOTS] = j2_cmd
    pub[PUB_J1_SLOTS] = j1_cmd
    pub = np.clip(pub, PUB_LOWER, PUB_UPPER)

    if prev_pub is not None and SPEED_FRAC is not None:
        max_delta = PUB_VEL_LIMIT * SPEED_FRAC / CONTROL_HZ
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


def fit_thresholds_from_warmup(fsr_arr, bt_arr):
    """fsr_arr (N,12), bt_arr (N,5) with NaNs possible on BT.

    FSR: hi = p99.5(empty) + MARGIN_ABS, lo = median(empty) + MARGIN_LO.
    BioTac PDC: same percentile+margin idea on valid samples.
    """
    global fsr_baseline, fsr_noise, fsr_hi, fsr_lo
    global bt_baseline, bt_noise, bt_hi, bt_lo

    fsr_baseline = np.percentile(fsr_arr, P_LO, axis=0).astype(np.float32)
    fsr_phi = np.percentile(fsr_arr, P_HI, axis=0).astype(np.float32)
    fsr_noise = np.maximum(fsr_arr.std(axis=0), 1e-6).astype(np.float32)  # logged only
    fsr_hi = (fsr_phi + MARGIN_ABS).astype(np.float32)
    fsr_lo = (fsr_baseline + MARGIN_LO).astype(np.float32)
    # Keep hysteresis valid: lo must be strictly below hi.
    fsr_lo = np.minimum(fsr_lo, fsr_hi - 1.0).astype(np.float32)

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
            bt_hi[k] = float(np.percentile(valid, P_HI) + MARGIN_ABS)
            bt_lo[k] = float(bt_baseline[k] + MARGIN_LO)
            if bt_lo[k] >= bt_hi[k]:
                bt_lo[k] = bt_hi[k] - 1.0

    rospy.loginfo("FSR thresh mode=p%.1f+%.1f / p%.1f+%.1f", P_HI, MARGIN_ABS, P_LO, MARGIN_LO)
    rospy.loginfo("FSR baseline(med)=%s", fsr_baseline)
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
    pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
    vel_norm = current_joint_vel / JOINT_VEL_LIMIT
    error = last_command - current_joint_pos
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

    rospy.init_node("deploy_warmup_trial15")
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
        fsr_mute_mux=np.array(FSR_MUTE_MUX, dtype=np.int32),
        joints=PUBLISH_JOINTS,
        replay_file=REPLAY_Q_FILE,
        warmup_repeats=np.int32(WARMUP_REPEATS),
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
    last_command[:] = current_joint_pos
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
    rec = {"t": [], "q": [], "cmd": [], "tac": [], "fsr": [], "biotac_pdc": [], "act": []}
    rospy.loginfo("Policy loop @ %d Hz | motion-calibrated tactile | governor=%s",
                  CONTROL_HZ, SPEED_FRAC)

    try:
        while not rospy.is_shutdown():
            prop_buffer.append(build_prop())
            tactile_buffer.append(read_tactile())
            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)

            pub_target, raw_cmd, j2_cmd = action_to_publish(
                action, current_joint_pos[CURL_J2_IDX], prev_pub
            )
            prev_pub = pub_target
            publish_target(pub, pub_target, 1.0 / CONTROL_HZ)

            rec["t"].append(rospy.get_time())
            rec["q"].append(current_joint_pos.copy())
            rec["cmd"].append(pub_target.copy())
            rec["tac"].append(tactile_buffer[-1].copy())
            with fsr_lock:
                rec["fsr"].append(latest_fsr.copy())
            with bt_lock:
                rec["biotac_pdc"].append(latest_biotac_pdc.copy())
            rec["act"].append(action.copy())

            last_action[:] = action
            last_command[:] = raw_cmd
            last_command[CURL_J2_IDX] = j2_cmd
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
            )
            rospy.loginfo("saved %s: %d steps", POLICY_NPZ, len(rec["t"]))


if __name__ == "__main__":
    main()
