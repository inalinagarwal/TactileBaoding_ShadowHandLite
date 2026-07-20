#!/usr/bin/env python
"""Deploy RoTO Shadow Lite blind Baoding policy on the real Shadow Dexterous Hand Lite (ROS).

Loads encoder+policy from best_agent.pt, builds the 352-d observation (prop 256 + tactile 96)
at 60 Hz, and commands joint position targets on /rh_trajectory_controller/command (radians).


See SHADOWLITE_DEPLOY.md for the full obs/action contract, joint order, coupling and tactile map.
"""

import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll

REPLAY_FILE = None
# ============================================================
# MODEL (must match trained architecture exactly)
# ============================================================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(352, 1024), nn.LayerNorm(1024), nn.ELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64),  nn.ELU(),
            nn.Linear(64, 16),
        )

    def forward(self, z):
        return self.policy_net(z)


# ============================================================
# JOINT ORDER + LIMITS  (authoritative: sim actuated_dof_indices dump)
# Order is breadth-first by joint level, NOT per-finger.
# ============================================================
POLICY_JOINTS = [
    'rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_THJ5',
    'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_THJ4',
    'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2',
    'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1',
    'rh_THJ2', 'rh_THJ1',
]
JOINT_LOWER = np.array([-0.3491, -0.3491, -0.3491, -1.0472, -0.2618, -0.2618, -0.2618,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6981, -0.2618], dtype=np.float32)
JOINT_UPPER = np.array([0.3491, 0.3491, 0.3491, 1.0472, 1.5708, 1.5708, 1.5708,
                        1.2217, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 0.6981, 1.5708], dtype=np.float32)
JOINT_VEL_LIMIT = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 4.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0], dtype=np.float32)

# Hardware coupling J1 <= J2 (FF/MF/RF): (distal_idx, middle_idx)
COUPLED_PAIRS = [(11, 8), (12, 9), (13, 10)]

# Update to the location of best_agent.pt on the control laptop.
# (Trained on the GPU box at:
#  scripts/logs/shadowlite_baoding/rl_only_pt/<timestamp>/checkpoints/best_agent.pt)
CHECKPOINT = "/home/user/experiments/best_agent_legacy.pt"
CONTROL_HZ = 60
OBS_STACK = 4
NUM_J = 16
NUM_TACTILE = 24


def unscale(x, lo, hi):
    """Raw radians -> [-1, 1] (matches roto_env.unscale)."""
    return (2.0 * x - hi - lo) / (hi - lo)


def scale(a, lo, hi):
    """[-1, 1] -> raw radians (matches roto_env.scale)."""
    return 0.5 * (a + 1.0) * (hi - lo) + lo


# ============================================================
# GLOBAL STATE
# ============================================================
current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
last_action = np.zeros(NUM_J, dtype=np.float32)
last_command = np.zeros(NUM_J, dtype=np.float32)
joint_ready = False
biotac_ready = False

prop_buffer = deque(maxlen=OBS_STACK)
tactile_buffer = deque(maxlen=OBS_STACK)


# ============================================================
# ROS CALLBACK
# ============================================================
def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    idx = {n: i for i, n in enumerate(msg.name)}
    try:
        for i, j in enumerate(POLICY_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


# ============================================================
# TACTILE -- FAKE (zeros) for now. Replace with FSR/Arduino source.
#
# 24 sim-body channels (order):
#   0 world         6 rh_thbase     12 rh_mfmiddle   18 rh_thmiddle
#   1 rh_forearm    7 rh_ffproximal 13 rh_rfmiddle   19 rh_fftip*
#   2 rh_palm       8 rh_mfproximal 14 rh_thhub      20 rh_mftip*
#   3 rh_ffknuckle  9 rh_rfproximal 15 rh_ffdistal   21 rh_rftip*
#   4 rh_mfknuckle 10 rh_thproximal 16 rh_mfdistal   22 rh_thdistal
#   5 rh_rfknuckle 11 rh_ffmiddle   17 rh_rfdistal   23 rh_thtip*
# (* world/forearm/knuckles/hub/tips were ~always 0 in sim -> leave 0)
#
# FSR -> channel plan:
#   tips      -> 15, 16, 17, 22
#   middles   -> 11, 12, 13, 18
#   proximals -> 7, 8, 9, 10
#   palm      -> 2   (OR multiple pads into this single channel)
#   thumb base-> 6
# Resolution is per-link: OR all pads on one link into that one channel.
# Binarize each pad with a per-pad threshold above its noise floor.
# i-th Arduino value -> sim tactile channel index (wire the Arduino CSV in THIS order)
FSR_CHANNELS = [
    10,  # C0  thumb proximal   -> thprox
    7,   # C1  first proximal   -> ffprox
    4,   # C2  middle knuckle   -> mfknuckle
    9,   # C3  ring proximal    -> rfprox
    5,   # C4  ring knuckle     -> rfknuckle
    13,  # C5  ring middle      -> rfmid   (wire-checked; was palm)
    2,   # C6  palm             -> palm   (wire-checked; was ffmid)
    3,   # C7  first knuckle    -> ffknuckle
    8,   # C8  middle proximal  -> mfprox
    18,  # C9  thumb middle     -> thmiddle
    12,  # C10 middle middle    -> mfmid
    11,  # C11 first middle     -> ffmid  (wire-checked; was rfmid)
]
N_FSR = 12

# BioTac tactiles[i]: 0=FF, 1=MF, 2=RF, 3=LF(unused), 4=TH
BIOTAC_IDX = [0, 1, 2, 4]
BIOTAC_CH  = [15, 16, 17, 22]   # ffdist, mfdist, rfdist, thdistal
N_BIOTAC   = len(BIOTAC_IDX)


import serial, threading

SERIAL_PORT = "/dev/ttyACM0"   # check `ls /dev/ttyACM* /dev/ttyUSB*`
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
fsr_noise    = np.ones(N_FSR, dtype=np.float32)
bt_baseline  = np.zeros(N_BIOTAC, dtype=np.float32)
bt_noise     = np.ones(N_BIOTAC, dtype=np.float32)

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

K_HI, K_LO = 5.0, 2.0          # tune per setup; hi > lo
bt_state  = np.zeros(N_BIOTAC, dtype=bool)
fsr_state = np.zeros(N_FSR, dtype=bool)

def read_tactile():
    global fsr_state, bt_state
    with fsr_lock:
        fsr_vals = latest_fsr.copy()
    fsr_hi = fsr_baseline + K_HI * fsr_noise
    fsr_lo = fsr_baseline + K_LO * fsr_noise
    fsr_state = np.where(fsr_vals > fsr_hi, True,
                 np.where(fsr_vals < fsr_lo, False, fsr_state))
    with bt_lock:
        bt_vals = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
    bt_hi = bt_baseline + K_HI * bt_noise
    bt_lo = bt_baseline + K_LO * bt_noise
    for k in range(N_BIOTAC):
        v = bt_vals[k]
        if not np.isfinite(v) or v < 0:   # disconnected sentinel e.g. pdc=-9784
            continue
        if v > bt_hi[k]:
            bt_state[k] = True
        elif v < bt_lo[k]:
            bt_state[k] = False
    t = np.zeros(NUM_TACTILE, dtype=np.float32)
    t[FSR_CHANNELS] = np.maximum(t[FSR_CHANNELS], fsr_state.astype(np.float32))
    for k, ch in enumerate(BIOTAC_CH):
        t[ch] = max(t[ch], float(bt_state[k]))
    return t


# ============================================================
# OBS
# ============================================================
def build_prop():
    pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)   # [-1, 1]
    vel_norm = current_joint_vel / JOINT_VEL_LIMIT                    # per-joint limit, NOT /3.0
    error = last_command - current_joint_pos                         # raw radians (unnormalized in sim)
    return np.concatenate([pos_norm, vel_norm, error, last_action]).astype(np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    global last_action, last_command

    rospy.init_node("deploy_policy")
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

    rec = {'t': [], 'q': [], 'cmd': [], 'tac': [], 'fsr': [], 'biotac_pdc': [], 'act': []}
    LOG_EVERY = 1
    step_i = 0

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states received.")
    

    rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)
    
    rospy.loginfo("Waiting for /rh/tactile ...")
    while not biotac_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("BioTac states received.")


    t = threading.Thread(target=serial_reader)
    t.daemon=True
    t.start()

    rospy.loginfo("Calibrating FSR + BioTac (hand flat, empty)...")
    rospy.sleep(0.5)
    calibrate_fsr(2.0)
    calibrate_biotac(2.0)

    raw_input('Place the balls in the cup, then press Enter to start the policy...')

    if REPLAY_FILE:
        data = np.load(REPLAY_FILE, allow_pickle=True)
        rec_act = data["act"]                      # (T, 16) normalized actions from sim
        log = {"t": [], "cmd": [], "q": [], "qd": []}

        # move to the first target and let it settle BEFORE logging
        first = np.clip(scale(rec_act[0], JOINT_LOWER, JOINT_UPPER), JOINT_LOWER, JOINT_UPPER)
        m = JointTrajectory(); m.joint_names = POLICY_JOINTS
        p = JointTrajectoryPoint(); p.positions = first.tolist()
        p.time_from_start = rospy.Duration(2.0); m.points.append(p)
        pub.publish(m); rospy.sleep(3.0)

        rate = rospy.Rate(CONTROL_HZ)
        for a in rec_act:
            target = np.clip(scale(a, JOINT_LOWER, JOINT_UPPER), JOINT_LOWER, JOINT_UPPER)
            # NOTE: deliberately NO J1<=J2 clamp here -> identical commands to sim,
            #       so the hardware itself enforces the coupling and we can SEE the gap.
            msg = JointTrajectory(); msg.joint_names = POLICY_JOINTS
            pt = JointTrajectoryPoint(); pt.positions = target.tolist()
            pt.time_from_start = rospy.Duration(1.0 / CONTROL_HZ); msg.points.append(pt)
            pub.publish(msg)

            log["t"].append(rospy.get_time())
            log["cmd"].append(target.copy())
            log["q"].append(current_joint_pos.copy())     # achieved, from /joint_states
            log["qd"].append(current_joint_vel.copy())
            rate.sleep()

        np.savez("hw_replay_log.npz", **{k: np.array(v) for k, v in log.items()})
        rospy.loginfo("saved hw_replay_log.npz: %d steps", len(log["t"]))
        return

    # Init command to current pose (first error ~0) and fill stacks with the first REAL frame.
    last_command[:] = current_joint_pos
    p0, t0 = build_prop(), read_tactile()
    prop_buffer.clear()
    tactile_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())
        tactile_buffer.append(t0.copy())

    rospy.sleep(2.0)
    rate = rospy.Rate(CONTROL_HZ)

    try:
        while not rospy.is_shutdown():
            # 1) observation (oldest -> newest, prop then tactile)
            prop_buffer.append(build_prop())
            tactile_buffer.append(read_tactile())
            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))  # 4*64 + 4*24 = 352
            assert obs.shape[0] == 352, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            # 2) policy (deterministic = mean)
            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0]
            #action = np.clip(action, -1.0, 1.0)

            # 3) action -> absolute joint targets (rad), clamp + coupling
            raw_cmd = scale(action, JOINT_LOWER, JOINT_UPPER)
            target = np.clip(raw_cmd, JOINT_LOWER, JOINT_UPPER)
            for j1, j2 in COUPLED_PAIRS:
                target[j1] = min(target[j1], target[j2])   # enforce J1 <= J2
            

            # 4) publish
            msg = JointTrajectory()
            msg.joint_names = POLICY_JOINTS
            pt = JointTrajectoryPoint()
            pt.positions = target.tolist()
            pt.time_from_start = rospy.Duration(1.0 / CONTROL_HZ)
            msg.points.append(pt)
            pub.publish(msg)

            if step_i % LOG_EVERY == 0:
                rec["t"].append(rospy.get_time())
                rec["q"].append(current_joint_pos.copy())
                rec["cmd"].append(target.copy())
                rec["tac"].append(tactile_buffer[-1].copy())   # 24-d binary (policy input)
                with fsr_lock:
                    rec["fsr"].append(latest_fsr.copy())       # 7 raw ADC
                with bt_lock:
                    rec["biotac_pdc"].append(latest_biotac_pdc.copy()) # 5 raw ADC
                rec["act"].append(action.copy())               # normalized policy output
            step_i += 1

            # 5) bookkeeping for next obs
            last_action[:] = action
            last_command[:] = raw_cmd
            rate.sleep()
    finally:
        if rec["t"]:
            np.savez("hw_policy_log_padtac.npz", **{k: np.array(v) for k, v in rec.items()})
            rospy.loginfo("saved hw_policy_log_padtac.npz: %d steps", len(rec["t"]))


if __name__ == "__main__":
    main()
