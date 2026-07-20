#!/usr/bin/env python
"""Gate C deploy: SIM tactile + REAL proprioception (Trial 15 PadTac+BT).

Same contract as deploy_policy_new.py (13-DOF / 304-d, coupling, 60 Hz), but the
24-d tactile slots in the observation are fed from a recorded sim rollout
(sim_policy_log_trial15_seed42.npz['tac']) instead of live FSR/BioTac.

Use this to test whether idealized tactile restores closed-loop rotation.
Does NOT modify deploy_policy_new.py.

Copy to the control laptop with:
  best_agent_legacy_padtac_bt_scratch_trial15.pt
  sim_policy_log_trial15_seed42.npz  (or set SIM_TAC_FILE to its path)
"""

import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll

REPLAY_FILE = None  # closed-loop only (no q-replay in this script)

# Gate C tactile source: "sim" | "zero" | "real"
TACTILE_MODE = "sim"
# Trial 15 play log (must contain tac shaped (T, 24)). On the laptop, put the npz
# next to this script or set an absolute path.
SIM_TAC_FILE = "sim_policy_log_trial15_seed42.npz"
LOG_NPZ = "hw_policy_log_trial15_simtac.npz"

# ============================================================
# MODEL (must match roto_2 rl_only_pt trained architecture)
# ============================================================
OBS_DIM = 304
PROP_DIM = 52   # 13 * 4 per stacked frame
NUM_J = 13
NUM_TACTILE = 24
OBS_STACK = 4
CONTROL_HZ = 60


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 1024), nn.LayerNorm(1024), nn.ELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, NUM_J),
        )

    def forward(self, z):
        return self.policy_net(z)


# ============================================================
# JOINT ORDER + LIMITS  (matches sim control_joint_names / control_dof_indices)
# ============================================================
# JOINT_LOWER/UPPER are the SIM limits (USD soft_joint_pos_limits of
# shadow_touchlab_col.usd), verified via pxr. They MUST match sim exactly because
# the policy's obs uses pos_norm = unscale(q, lo, hi) and actions are scaled with
# scale(a, lo, hi). NOTE: FFJ2/MFJ2/RFJ2 upper is 1.745 (99.98 deg) in the USD,
# NOT 1.5708 (90 deg). The 90 deg value corrupted the obs for the 3 curl joints.
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
     1.7450, 1.7450, 1.7450, 0.6981, 1.5708],   # FFJ2/MFJ2/RFJ2 = 1.745 (USD), not 1.5708
    dtype=np.float32,
)
JOINT_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 2.0, 4.0],
    dtype=np.float32,
)

# ============================================================
# FINGER-CURL COUPLING
# Must replicate roto_env._handle_coupled_joints + shadowlite.py cfg. The policy's
# FFJ2/MFJ2/RFJ2 output is a COMBINED-CURL PROXY, not a direct J2 target. Sim maps
#   j2_cmd = clip(proxy * (J2_UPPER_SIM / theta), 0, J2_UPPER_SIM)      (~2.22x gain)
#   j1_cmd = clip((proxy - theta)/(J2_UPPER_SIM - theta) * J1_UPPER_SIM, 0, J1_UPPER_SIM)
#   j1_cmd *= gate   where gate ramps 0->1 as MEASURED j2 nears its limit.
# Sending the raw scaled proxy straight to FFJ2 (as before) under-curls to ~40% and
# makes the fingers move wrongly.
# ============================================================
COUPLING_THETA = 0.785     # rad, shadowlite.coupling_theta (split point)
J2_UPPER_SIM   = 1.7450    # rad, USD FFJ2/MFJ2/RFJ2 upper (used for gain + clamp, matches sim)
J1_UPPER_SIM   = 1.3960    # rad, USD FFJ1/MFJ1/RFJ1 upper (79.98 deg)
GATE_J2_TOL    = 0.035     # rad, couple_gate_j2_tol (frac=1.0 -> J1 opens within tol of J2 limit)
CURL_J2_IDX    = [8, 9, 10]   # FFJ2, MFJ2, RFJ2 positions in the 13-d control vector

# ============================================================
# PUBLISHED TRAJECTORY: 16 joints (control 13 + FFJ1/MFJ1/RFJ1 mimics), matching
# sim actuated_joint_names and the (validated) old deploy_policy.py joint set.
# ============================================================
PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]
# Map the 10 non-coupled control joints into their publish slots.
CTRL_NONCOUPLED = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
PUB_NONCOUPLED  = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]
PUB_J2_SLOTS    = [8, 9, 10]     # FFJ2/MFJ2/RFJ2 in PUBLISH_JOINTS
PUB_J1_SLOTS    = [11, 12, 13]   # FFJ1/MFJ1/RFJ1 in PUBLISH_JOINTS

# Hardware-SAFE clamp on the PUBLISHED targets (real Shadow Hand Lite limits: J2/J1
# max 90 deg = 1.5708, unlike the sim USD which models J2 to 100 deg). The firmware
# clamps too, but we clamp here so logs reflect what we actually asked for.
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
# Per-joint rated velocities (rad/s) for the velocity governor below.
PUB_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0,
     2.0, 2.0, 2.0,
     2.0, 4.0],
    dtype=np.float32,
)
# Velocity governor: cap per-step change so the hand can't snap violently (safety).
# max delta per control tick = PUB_VEL_LIMIT * SPEED_FRAC / CONTROL_HZ.
#   SPEED_FRAC = 1.0  -> full rated speed (pure safety cap, closest to sim)
#   SPEED_FRAC = 0.25 -> very slow, for first bring-up so you can watch each move
# Set to None to disable governing entirely.
SPEED_FRAC = None

# Trial 15 scratch PadTac+BT (legacy serialization for the HW laptop torch).
CHECKPOINT = "/home/user/experiments/best_agent_legacy_padtac_bt_scratch_trial15.pt"


def unscale(x, lo, hi):
    """Raw radians -> [-1, 1] (matches roto_env.unscale)."""
    return (2.0 * x - hi - lo) / (hi - lo)


def scale(a, lo, hi):
    """[-1, 1] -> raw radians (matches roto_env.scale)."""
    return 0.5 * (a + 1.0) * (hi - lo) + lo


def action_to_publish(action, meas_j2, prev_pub):
    """Turn a 13-d policy action into a 16-d hardware trajectory command.

    Replicates roto_env: scale(action) -> _handle_coupled_joints (J2 gain + gated
    J1) -> hardware clamp -> velocity governor. Returns:
      pub      : (16,) target radians in PUBLISH_JOINTS order (published to hand)
      raw_cmd  : (13,) unclamped scale(action) == sim joint_pos_cmd for non-coupled
      j2_cmd   : (3,)  post-coupling J2 command == sim joint_pos_cmd for FFJ2/MFJ2/RFJ2
    raw_cmd (with j2_cmd substituted) is what feeds the obs cmd_error term.
    """
    raw_cmd = scale(action, JOINT_LOWER, JOINT_UPPER)          # 13, unclamped (matches sim)
    proxy = raw_cmd[CURL_J2_IDX]                               # combined-curl proxy

    j2_cmd = np.clip(proxy * (J2_UPPER_SIM / COUPLING_THETA), 0.0, J2_UPPER_SIM)
    j1_cmd = np.clip(
        (proxy - COUPLING_THETA) / (J2_UPPER_SIM - COUPLING_THETA) * J1_UPPER_SIM,
        0.0, J1_UPPER_SIM,
    )
    # Gate J1 on MEASURED J2 (couple_gate_j1_on_measured, frac=1.0). On real hardware
    # J2 caps at 90 deg (< opens_at ~1.71), so this gate keeps J1 ~= 0, matching the
    # observed sim rollout where FFJ1/MFJ1/RFJ1 stayed 0.
    opens_at = J2_UPPER_SIM - GATE_J2_TOL
    gate = np.clip((np.asarray(meas_j2, dtype=np.float32) - opens_at) / GATE_J2_TOL, 0.0, 1.0)
    j1_cmd = j1_cmd * gate

    pub = np.empty(16, dtype=np.float32)
    pub[PUB_NONCOUPLED] = raw_cmd[CTRL_NONCOUPLED]
    pub[PUB_J2_SLOTS]   = j2_cmd
    pub[PUB_J1_SLOTS]   = j1_cmd
    pub = np.clip(pub, PUB_LOWER, PUB_UPPER)

    # Velocity governor (safety): limit change per control tick.
    if prev_pub is not None and SPEED_FRAC is not None:
        max_delta = PUB_VEL_LIMIT * SPEED_FRAC / CONTROL_HZ
        pub = prev_pub + np.clip(pub - prev_pub, -max_delta, max_delta)

    return pub.astype(np.float32), raw_cmd, j2_cmd


# ============================================================
# GLOBAL STATE
# ============================================================
current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
# Normalized policy output from the previous control step (fed into proprio as last_action).
last_action = np.zeros(NUM_J, dtype=np.float32)
# Scaled position command in radians for the 13 control joints (fed into cmd_error).
last_command = np.zeros(NUM_J, dtype=np.float32)
joint_ready = False
biotac_ready = False

# --- REPLAY-ONLY 16-DOF LOGGING (comment out this block to disable) ---
# Extra buffers for FFJ1/MFJ1/RFJ1. Policy path still uses current_joint_pos (13).
current_joint_pos16 = np.zeros(16, dtype=np.float32)
current_joint_vel16 = np.zeros(16, dtype=np.float32)
# --- end REPLAY-ONLY 16-DOF LOGGING ---

prop_buffer = deque(maxlen=OBS_STACK)
tactile_buffer = deque(maxlen=OBS_STACK)


# ============================================================
# ROS CALLBACK
# ============================================================
def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    global current_joint_pos16, current_joint_vel16  # REPLAY-ONLY 16-DOF LOGGING
    idx = {n: i for i, n in enumerate(msg.name)}
    try:
        for i, j in enumerate(POLICY_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        # --- REPLAY-ONLY 16-DOF LOGGING (comment out to disable) ---
        for i, j in enumerate(PUBLISH_JOINTS):
            current_joint_pos16[i] = msg.position[idx[j]]
            current_joint_vel16[i] = msg.velocity[idx[j]]
        # --- end REPLAY-ONLY 16-DOF LOGGING ---
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


# ============================================================
# TACTILE — FSR + BioTac -> 24-d binary vector
# ============================================================
FSR_CHANNELS = [
    10,  # C0  thumb proximal   -> thprox
    7,   # C1  first proximal   -> ffprox
    4,   # C2  middle knuckle   -> mfknuckle
    9,   # C3  ring proximal    -> rfprox
    5,   # C4  ring knuckle     -> rfknuckle
    13,   # C5  rfmid (was palm)
    2,  # C6  palm (was ffmid)
    3,   # C7  first knuckle    -> ffknuckle
    8,   # C8  middle proximal  -> mfprox
    18,  # C9  thumb middle     -> thmiddle
    12,  # C10 middle middle    -> mfmid
    11,  # C11 ffmid (was rfmid)
]
N_FSR = 12

BIOTAC_IDX = [0, 1, 2, 4]
BIOTAC_CH = [15, 16, 17, 22]
N_BIOTAC = len(BIOTAC_IDX)
USE_BIOTAC = True

import serial
import threading

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

latest_fsr = np.zeros(N_FSR, dtype=np.float32)
fsr_lock = threading.Lock()


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


latest_biotac_pdc = np.full(5, np.nan, dtype=np.float32)
bt_lock = threading.Lock()


def biotac_cb(msg):
    global latest_biotac_pdc, biotac_ready
    with bt_lock:
        for i, t in enumerate(msg.tactiles):
            latest_biotac_pdc[i] = float(t.pdc)
    biotac_ready = True


fsr_baseline = np.zeros(N_FSR, dtype=np.float32)
fsr_noise = np.ones(N_FSR, dtype=np.float32)
bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
bt_noise = np.ones(N_BIOTAC, dtype=np.float32)


def calibrate_fsr(seconds=2.0):
    global fsr_baseline, fsr_noise
    samples = []
    t_end = rospy.get_time() + seconds
    while rospy.get_time() < t_end and not rospy.is_shutdown():
        with fsr_lock:
            samples.append(latest_fsr.copy())
        rospy.sleep(0.01)
    s = np.array(samples)
    fsr_baseline = s.mean(0)
    fsr_noise = s.std(0) + 1e-6
    rospy.loginfo("FSR baseline=%s noise=%s", fsr_baseline, fsr_noise)


def calibrate_biotac(seconds=2.0):
    global bt_baseline, bt_noise
    samples = []
    t_end = rospy.get_time() + seconds
    while rospy.get_time() < t_end and not rospy.is_shutdown():
        with bt_lock:
            row = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
        if np.all(np.isfinite(row)) and np.all(row > 0):
            samples.append(row)
        rospy.sleep(0.01)
    if not samples:
        rospy.logwarn("BioTac calibrate: no valid samples; using defaults")
        return
    s = np.array(samples)
    bt_baseline = s.mean(0)
    bt_noise = s.std(0) + 1e-6
    rospy.loginfo("BioTac baseline=%s noise=%s", bt_baseline, bt_noise)


K_HI, K_LO = 5.0, 2.0
bt_state = np.zeros(N_BIOTAC, dtype=bool)
fsr_state = np.zeros(N_FSR, dtype=bool)


def read_tactile():
    global fsr_state, bt_state
    with fsr_lock:
        fsr_vals = latest_fsr.copy()
    fsr_hi = fsr_baseline + K_HI * fsr_noise
    fsr_lo = fsr_baseline + K_LO * fsr_noise
    fsr_state = np.where(
        fsr_vals > fsr_hi,
        True,
        np.where(fsr_vals < fsr_lo, False, fsr_state),
    )
    with bt_lock:
        bt_vals = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
    bt_hi = bt_baseline + K_HI * bt_noise
    bt_lo = bt_baseline + K_LO * bt_noise
    for k in range(N_BIOTAC):
        v = bt_vals[k]
        if not np.isfinite(v) or v < 0:
            continue
        if v > bt_hi[k]:
            bt_state[k] = True
        elif v < bt_lo[k]:
            bt_state[k] = False
    t = np.zeros(NUM_TACTILE, dtype=np.float32)
    t[FSR_CHANNELS] = np.maximum(t[FSR_CHANNELS], fsr_state.astype(np.float32))
    if USE_BIOTAC:
        for k, ch in enumerate(BIOTAC_CH):
            t[ch] = max(t[ch], float(bt_state[k]))
    return t


# Sim-tactile playback state (loaded in main).
sim_tac = None
sim_tac_i = 0


def get_tactile():
    """Obs tactile: sim playback, all-zero, or live FSR/BioTac."""
    global sim_tac_i
    if TACTILE_MODE == "zero":
        return np.zeros(NUM_TACTILE, dtype=np.float32)
    if TACTILE_MODE == "sim":
        assert sim_tac is not None
        t = sim_tac[min(sim_tac_i, len(sim_tac) - 1)].copy()
        sim_tac_i += 1
        return t
    return read_tactile()


# ============================================================
# OBS
# ============================================================
def build_prop():
    pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
    vel_norm = current_joint_vel / JOINT_VEL_LIMIT
    error = last_command - current_joint_pos
    return np.concatenate([pos_norm, vel_norm, error, last_action]).astype(np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    global last_action, last_command, sim_tac, sim_tac_i

    rospy.init_node("deploy_policy_simtactile")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
    rospy.Subscriber("/joint_states", JointState, joint_callback)

    rospy.loginfo("Loading checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    encoder, policy = Encoder(), Policy()
    e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    rospy.loginfo("encoder load missing=%s unexpected=%s", e_res.missing_keys, e_res.unexpected_keys)
    policy.load_state_dict(
        {k: v for k, v in ckpt["policy"].items() if k != "log_std_parameter"},
        strict=True,
    )
    encoder.eval()
    policy.eval()

    if TACTILE_MODE == "sim":
        data = np.load(SIM_TAC_FILE, allow_pickle=True)
        sim_tac = data["tac"].astype(np.float32)
        assert sim_tac.ndim == 2 and sim_tac.shape[1] == NUM_TACTILE, sim_tac.shape
        sim_tac_i = 0
        rospy.loginfo("SIM TACTILE from %s  T=%d", SIM_TAC_FILE, len(sim_tac))
    else:
        rospy.loginfo("TACTILE_MODE=%s", TACTILE_MODE)

    rec = {"t": [], "q": [], "cmd": [], "tac": [], "fsr": [], "biotac_pdc": [], "act": []}
    LOG_EVERY = 1
    step_i = 0

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states received.")

    # Optional live sensors for logging only (policy uses get_tactile).
    rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)
    t = threading.Thread(target=serial_reader)
    t.daemon = True
    t.start()
    if TACTILE_MODE == "real":
        rospy.loginfo("Waiting for /rh/tactile ...")
        while not biotac_ready and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Calibrating FSR + BioTac (hand empty)...")
        rospy.sleep(0.5)
        calibrate_fsr(2.0)
        calibrate_biotac(2.0)
    else:
        rospy.loginfo("Skipping FSR/BioTac calibrate (TACTILE_MODE=%s); still logging raw if present",
                      TACTILE_MODE)
        rospy.sleep(0.5)

    input("Place the balls in the cup, then press Enter to start the policy...")

    # Seed stacks: first frame uses zero last_action (matches sim reset).
    last_command[:] = current_joint_pos
    last_action[:] = 0.0
    p0, t0 = build_prop(), get_tactile()
    prop_buffer.clear()
    tactile_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())
        tactile_buffer.append(t0.copy())

    rospy.sleep(2.0)
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = None
    rospy.loginfo("Closed-loop @ %d Hz | TACTILE_MODE=%s | governor=%s",
                  CONTROL_HZ, TACTILE_MODE, SPEED_FRAC)

    try:
        while not rospy.is_shutdown():
            prop_buffer.append(build_prop())
            tactile_buffer.append(get_tactile())
            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)
            assert action.shape == (NUM_J,), action.shape

            pub_target, raw_cmd, j2_cmd = action_to_publish(
                action, current_joint_pos[CURL_J2_IDX], prev_pub
            )
            prev_pub = pub_target

            msg = JointTrajectory()
            msg.joint_names = PUBLISH_JOINTS
            pt = JointTrajectoryPoint()
            pt.positions = pub_target.tolist()
            pt.time_from_start = rospy.Duration(1.0 / CONTROL_HZ)
            msg.points.append(pt)
            pub.publish(msg)

            if step_i % LOG_EVERY == 0:
                rec["t"].append(rospy.get_time())
                rec["q"].append(current_joint_pos.copy())
                rec["cmd"].append(pub_target.copy())
                rec["tac"].append(tactile_buffer[-1].copy())  # fed obs tactile (sim)
                with fsr_lock:
                    rec["fsr"].append(latest_fsr.copy())
                with bt_lock:
                    rec["biotac_pdc"].append(latest_biotac_pdc.copy())
                rec["act"].append(action.copy())
            step_i += 1

            last_action[:] = action
            last_command[:] = raw_cmd
            last_command[CURL_J2_IDX] = j2_cmd
            rate.sleep()
    finally:
        if rec["t"]:
            np.savez(LOG_NPZ, **{k: np.array(v) for k, v in rec.items()})
            rospy.loginfo("saved %s: %d steps", LOG_NPZ, len(rec["t"]))


if __name__ == "__main__":
    main()
