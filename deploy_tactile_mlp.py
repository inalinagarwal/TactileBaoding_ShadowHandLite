#!/usr/bin/env python
"""Trial-15 deploy: real tactile + SPEED_FRAC slew + MLP-synthesised pos_err.

Same warmup / Gate-B tactile calibration as deploy_warmup_trial15.py, but the
proprioceptive command-error channel is NOT taken from raw/slewed last_command.
Instead a frozen MLP predicts soft-sim joint_pos_error from:

  X = concat(pos4, vel4, act4, cur_action)   # 169-D
  y = joint_pos_error after env.step(cur_action)

Plant path: policy action -> coupling -> SPEED_FRAC slew -> publish.
Obs path:  MLP(...) -> pos_err * POS_ERR_SPEED_FRAC -> encoder prop.
  (POS_ERR_SPEED_FRAC=None means *1.0, i.e. no obs scaling)

Edit top flags:
  SPEED_FRAC, POS_ERR_SPEED_FRAC, CHECKPOINT, etc.

Safety check (run before HW):
  python deploy_tactile_mlp.py --rmse-check
  Uses sim_policy_log_trial15_seed42.npz (reconstructs q13/cmd13/err;
  finite-diff velocity). Expect overall RMSE ~0.35–0.40 rad, corr ~0.99.
  Checkpoint's own val_rmse_overall (~0.03) was on the MLP training split,
  not this play log — we gate on the trial15 reconstruction metrics.

Does NOT modify other deploy scripts.
"""

from __future__ import print_function

import argparse
import os
import sys
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from fsr_pad_map import FSR_CHANNELS, FSR_NAMES

# ============================================================
# PATHS / PROTOCOL
# ============================================================
REPLAY_Q_FILE = "sim_policy_log_trial15_seed42.npz"
CHECKPOINT = "best_agent_legacy_padtac_bt_scratch_trial15.pt"
POS_ERR_MLP_PATH = "pos_error_mlp.pt"
RMSE_LOG_FILE = "sim_policy_log_trial15_seed42.npz"

WARMUP_REPEATS = 5
WARMUP_NPZ = "hw_warmup_trial15_mlp_fsr.npz"
POLICY_NPZ = "hw_policy_log_trial15_tactile_mlp.npz"

CONTROL_HZ = 60

# Plant: rate-limit published joint targets (None = no slew).
SPEED_FRAC = 0.5

# Obs: multiply MLP pos_err by this after predict (independent of SPEED_FRAC).
#   None -> *1.0 (raw MLP prediction)
#   0.5  -> half the predicted error, etc.
POS_ERR_SPEED_FRAC = None

# Offline RMSE gate (trial15 reconstruction; see --rmse-check)
RMSE_MAX = 0.60          # rad; trial15 check ~0.36
CORR_MIN = 0.90          # Pearson over all joints/steps; trial15 ~0.99
FORCE_DEPLOY = False     # set True to skip gate failure (or pass --force)

# Empty-motion envelope thresholds (same as deploy_warmup_trial15)
P_HI = 99.5
P_LO = 50.0
MARGIN_ABS = 5.0
MARGIN_LO = 2.0
WIDE_ENVELOPE_ADC = 15.0
HYST_BAND_WIDE = 3.0
STD_FLOOR_BT = 1.0

# ============================================================
# MODEL DIMS
# ============================================================
OBS_DIM = 304
PROP_DIM = 52
NUM_J = 13
NUM_TACTILE = 24
OBS_STACK = 4
MLP_HIST = 4
MLP_IN_DIM = 169  # 52+52+52+13


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


class PosErrorMLP(nn.Module):
    """Soft-sim joint_pos_error predictor.

    Architecture matches pos_error_mlp.pt:
      Linear 169->256, LN, ELU,
      Linear 256->256, LN, ELU,
      Linear 256->128, LN,
      Linear 128->13
    Inputs/outputs are de-/re-standardised with checkpoint x_*/y_* stats.
    """

    def __init__(self, ckpt):
        super(PosErrorMLP, self).__init__()
        assert int(ckpt["input_dim"]) == MLP_IN_DIM
        assert int(ckpt["target_dim"]) == NUM_J
        self.net = nn.Sequential(
            nn.Linear(MLP_IN_DIM, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, 128), nn.LayerNorm(128),
        )
        self.head = nn.Linear(128, NUM_J)
        self.load_state_dict(ckpt["state_dict"], strict=True)
        self.register_buffer(
            "x_mean", torch.tensor(np.asarray(ckpt["x_mean"]), dtype=torch.float32)
        )
        self.register_buffer(
            "x_std", torch.tensor(np.asarray(ckpt["x_std"]), dtype=torch.float32)
        )
        self.register_buffer(
            "y_mean", torch.tensor(np.asarray(ckpt["y_mean"]), dtype=torch.float32)
        )
        self.register_buffer(
            "y_std", torch.tensor(np.asarray(ckpt["y_std"]), dtype=torch.float32)
        )
        self.control_names = list(ckpt["control_names"])
        self.val_rmse_overall = float(ckpt["val_rmse_overall"])
        self.input_layout = str(ckpt["input_layout"])

    def forward(self, x):
        # x: (B, 169) or (169,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        xn = (x - self.x_mean) / (self.x_std + 1e-8)
        y = self.head(self.net(xn))
        return y * self.y_std + self.y_mean


# ============================================================
# JOINTS / COUPLING (same as deploy_warmup_trial15)
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

N_FSR = 12
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


def action_to_publish(action, meas_j2, prev_pub):
    """Map policy action -> 16-d publish target; optional SPEED_FRAC slew."""
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


def load_pos_err_mlp(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    mlp = PosErrorMLP(ckpt)
    mlp.eval()
    if mlp.control_names != POLICY_JOINTS:
        raise RuntimeError(
            "MLP control_names %s != POLICY_JOINTS %s"
            % (mlp.control_names, POLICY_JOINTS)
        )
    return mlp


def mlp_pos_err_scale():
    """Independent scale for MLP obs error (not plant slew)."""
    if POS_ERR_SPEED_FRAC is None:
        return 1.0
    return float(POS_ERR_SPEED_FRAC)


def build_mlp_features(pos_hist, vel_hist, act4, cur_action):
    """Alignment C (best on trial15 RMSE):
    pos4/vel4 = 4 frames ending at *current* post-step state (oldest->newest),
    act4      = 4 actions *before* cur_action,
    cur_action = last applied policy action.
    """
    pos4 = np.concatenate(list(pos_hist), axis=0)
    vel4 = np.concatenate(list(vel_hist), axis=0)
    a4 = np.concatenate(list(act4), axis=0)
    return np.concatenate([pos4, vel4, a4, cur_action.astype(np.float32)]).astype(
        np.float32
    )


def predict_pos_err(mlp, pos_hist, vel_hist, act4, cur_action):
    x = build_mlp_features(pos_hist, vel_hist, act4, cur_action)
    with torch.no_grad():
        y = mlp(torch.from_numpy(x)).numpy()[0].astype(np.float32)
    return y


# ============================================================
# OFFLINE RMSE SAFETY CHECK
# ============================================================
def _policy_indices_from_joints(joints16):
    names = list(joints16)
    return [names.index(j) for j in POLICY_JOINTS]


def rmse_check(mlp_path, log_path, verbose=True):
    """Replay MLP on a sim log; return (ok, metrics_dict).

    Accepts either:
      - full play log with q13, qd13, act, pos_err13
      - trial15-style log with q(16), cmd(16), act, joints  (finite-diff qd)
    Uses alignment C (see build_mlp_features).
    """
    mlp = load_pos_err_mlp(mlp_path)
    data = np.load(log_path, allow_pickle=True)
    files = set(data.files)

    if {"q13", "qd13", "act", "pos_err13"} <= files:
        q13 = data["q13"].astype(np.float32)
        qd13 = data["qd13"].astype(np.float32)
        act = data["act"].astype(np.float32)
        err = data["pos_err13"].astype(np.float32)
        source = "full_q13"
    elif {"q", "cmd", "act", "joints"} <= files:
        idx = _policy_indices_from_joints(data["joints"])
        q13 = data["q"][:, idx].astype(np.float32)
        cmd13 = data["cmd"][:, idx].astype(np.float32)
        act = data["act"].astype(np.float32)
        err = (cmd13 - q13).astype(np.float32)
        qd13 = np.zeros_like(q13)
        qd13[1:] = (q13[1:] - q13[:-1]) * float(CONTROL_HZ)
        source = "trial15_reconstructed_fdvel"
    else:
        raise RuntimeError(
            "RMSE log needs (q13,qd13,act,pos_err13) or (q,cmd,act,joints); got %s"
            % sorted(files)
        )

    pos = unscale(q13, JOINT_LOWER, JOINT_UPPER)
    vel = qd13 / JOINT_VEL_LIMIT

    preds = []
    gts = []
    # i indexes post-step frame after act[i]; need i >= 4
    for i in range(4, len(act)):
        pos_hist = [pos[i - 3 + k] for k in range(4)]          # i-3..i
        vel_hist = [vel[i - 3 + k] for k in range(4)]
        act4 = [act[i - 4 + k] for k in range(4)]              # i-4..i-1
        cur = act[i]
        x = np.concatenate(
            [np.concatenate(pos_hist), np.concatenate(vel_hist),
             np.concatenate(act4), cur]
        ).astype(np.float32)
        with torch.no_grad():
            pred = mlp(torch.from_numpy(x)).numpy()[0]
        preds.append(pred)
        gts.append(err[i])

    pred = np.stack(preds)
    gt = np.stack(gts)
    mse = ((pred - gt) ** 2).mean()
    rmse = float(np.sqrt(mse))
    per_joint = np.sqrt(((pred - gt) ** 2).mean(axis=0))
    corr = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1])

    ok = (rmse <= RMSE_MAX) and (corr >= CORR_MIN)
    metrics = {
        "ok": ok,
        "rmse": rmse,
        "corr": corr,
        "per_joint_rmse": per_joint,
        "n": len(preds),
        "source": source,
        "ckpt_val_rmse": mlp.val_rmse_overall,
        "rmse_max": RMSE_MAX,
        "corr_min": CORR_MIN,
        "layout": mlp.input_layout,
    }

    if verbose:
        print("=== pos_err MLP RMSE check ===")
        print("mlp:        ", os.path.abspath(mlp_path))
        print("log:        ", os.path.abspath(log_path))
        print("source:     ", source)
        print("layout:     ", mlp.input_layout)
        print("N steps:    ", metrics["n"])
        print("RMSE:       %.4f rad  (gate <= %.2f)" % (rmse, RMSE_MAX))
        print("corr:       %.4f       (gate >= %.2f)" % (corr, CORR_MIN))
        print("ckpt val_rmse_overall (training split): %.4f" % mlp.val_rmse_overall)
        print("per-joint RMSE:")
        for name, e in zip(POLICY_JOINTS, per_joint):
            print("  %-10s %.4f" % (name, float(e)))
        print("RESULT:     ", "PASS" if ok else "FAIL")
        if not ok:
            print(
                "Gate failed. Re-check MLP/log alignment, or pass --force to deploy anyway."
            )
    return ok, metrics


def verify_encoder_load(load_result, checkpoint_path, logfn):
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    weight_missing = [k for k in missing if k.startswith("net.")]
    if weight_missing:
        logfn(
            "ENCODER INCOMPLETE — missing weight keys: %s  (checkpoint: %s)"
            % (weight_missing, checkpoint_path)
        )
    elif missing:
        logfn("Encoder missing_keys (non-weight): %s" % missing)
    else:
        logfn(
            "Encoder load OK: all weights present. checkpoint=%s" % checkpoint_path
        )
    if unexpected:
        logfn("Encoder unexpected_keys: %s" % unexpected)
    return not weight_missing


# ============================================================
# ROS DEPLOY (imported only when running full deploy)
# ============================================================
def run_deploy(force=False):
    import rospy
    import serial
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sr_robot_msgs.msg import BiotacAll

    # ----- resolve paths relative to CWD (usually roto_2 or laptop copy) -----
    for label, path in [
        ("REPLAY_Q_FILE", REPLAY_Q_FILE),
        ("CHECKPOINT", CHECKPOINT),
        ("POS_ERR_MLP_PATH", POS_ERR_MLP_PATH),
        ("RMSE_LOG_FILE", RMSE_LOG_FILE),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError("%s not found: %s" % (label, path))

    # ----- RMSE gate before touching the hand -----
    ok, metrics = rmse_check(POS_ERR_MLP_PATH, RMSE_LOG_FILE, verbose=True)
    if not ok and not (force or FORCE_DEPLOY):
        raise SystemExit("RMSE gate failed — aborting deploy (use --force to override).")

    pos_err_mlp = load_pos_err_mlp(POS_ERR_MLP_PATH)

    # ----- mutable state -----
    current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
    current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
    current_joint_pos16 = np.zeros(16, dtype=np.float32)
    current_joint_vel16 = np.zeros(16, dtype=np.float32)
    last_action = np.zeros(NUM_J, dtype=np.float32)
    joint_ready = [False]
    biotac_ready = [False]

    prop_buffer = deque(maxlen=OBS_STACK)
    tactile_buffer = deque(maxlen=OBS_STACK)
    pos_hist = deque(maxlen=MLP_HIST)
    vel_hist = deque(maxlen=MLP_HIST)
    act4 = deque(maxlen=MLP_HIST)  # actions before cur (last_action)

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
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            for i, j in enumerate(POLICY_JOINTS):
                current_joint_pos[i] = msg.position[idx[j]]
                current_joint_vel[i] = msg.velocity[idx[j]]
            for i, j in enumerate(PUBLISH_JOINTS):
                current_joint_pos16[i] = msg.position[idx[j]]
                current_joint_vel16[i] = msg.velocity[idx[j]]
            joint_ready[0] = True
        except KeyError as e:
            rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)

    def serial_reader():
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
                    latest_fsr[:] = vals

    def biotac_cb(msg):
        with bt_lock:
            for i, t in enumerate(msg.tactiles):
                latest_biotac_pdc[i] = float(t.pdc)
        biotac_ready[0] = True

    def publish_target(pub, target, duration):
        msg = JointTrajectory()
        msg.joint_names = PUBLISH_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = target.tolist()
        pt.time_from_start = rospy.Duration(duration)
        msg.points.append(pt)
        pub.publish(msg)

    def _lo_from_envelope(baseline, phi, hi):
        envelope = (phi - baseline).astype(np.float32)
        lo_narrow = (baseline + MARGIN_LO).astype(np.float32)
        lo_wide = (hi - HYST_BAND_WIDE).astype(np.float32)
        wide = envelope > WIDE_ENVELOPE_ADC
        lo = np.where(wide, lo_wide, lo_narrow).astype(np.float32)
        lo = np.minimum(lo, hi - 1.0).astype(np.float32)
        return lo, wide, envelope

    def fit_thresholds_from_warmup(fsr_arr, bt_arr):
        nonlocal fsr_baseline, fsr_noise, fsr_hi, fsr_lo
        nonlocal bt_baseline, bt_noise, bt_hi, bt_lo

        fsr_baseline = np.percentile(fsr_arr, P_LO, axis=0).astype(np.float32)
        fsr_phi = np.percentile(fsr_arr, P_HI, axis=0).astype(np.float32)
        fsr_noise = np.maximum(fsr_arr.std(axis=0), 1e-6).astype(np.float32)
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
                rospy.logwarn(
                    "BioTac tip idx %d: few valid warmup samples (%d)",
                    BIOTAC_IDX[k], valid.size,
                )
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
        rospy.loginfo(
            "FSR wide_pads=%s (%s)",
            np.where(fsr_wide)[0].tolist(),
            [FSR_NAMES[i] for i in np.where(fsr_wide)[0]],
        )
        rospy.loginfo("FSR hi=%s", fsr_hi)
        rospy.loginfo("FSR lo=%s", fsr_lo)
        rospy.loginfo("BioTac baseline=%s hi=%s", bt_baseline, bt_hi)

    def apply_fsr_mute(fsr_on):
        out = fsr_on.astype(np.float32).copy()
        for mux_i in FSR_MUTE_MUX:
            out[mux_i] = 0.0
        return out

    def read_tactile():
        nonlocal fsr_state, bt_state
        with fsr_lock:
            fsr_vals = latest_fsr.copy()
        fsr_state = np.where(
            fsr_vals > fsr_hi,
            True,
            np.where(fsr_vals < fsr_lo, False, fsr_state),
        )
        with bt_lock:
            bt_vals = np.array(
                [latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32
            )
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

    def build_prop_mlp():
        """Prop with MLP pos_err (not cmd - q); optional * POS_ERR_SPEED_FRAC."""
        pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
        vel_norm = current_joint_vel / JOINT_VEL_LIMIT
        # refresh newest hist frame to current measurement
        if len(pos_hist) == MLP_HIST:
            pos_hist.append(pos_norm.copy())
            vel_hist.append(vel_norm.copy())
        else:
            while len(pos_hist) < MLP_HIST:
                pos_hist.append(pos_norm.copy())
                vel_hist.append(vel_norm.copy())
        while len(act4) < MLP_HIST:
            act4.append(np.zeros(NUM_J, dtype=np.float32))

        pred_raw = predict_pos_err(pos_err_mlp, pos_hist, vel_hist, act4, last_action)
        scale = mlp_pos_err_scale()
        pred_err = (pred_raw * scale).astype(np.float32)
        return (
            np.concatenate([pos_norm, vel_norm, pred_err, last_action]).astype(np.float32),
            pred_err,
            pred_raw,
            pos_norm,
            vel_norm,
        )

    def run_q_replay(pub, rec_q, episode_id, log):
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

    # ----- ROS node -----
    rospy.init_node("deploy_tactile_mlp")
    pub = rospy.Publisher(
        "/rh_trajectory_controller/command", JointTrajectory, queue_size=1
    )
    rospy.Subscriber("/joint_states", JointState, joint_callback)
    rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready[0] and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states OK.")

    rospy.loginfo("Waiting for /rh/tactile ...")
    while not biotac_ready[0] and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("BioTac OK.")

    th = threading.Thread(target=serial_reader)
    th.daemon = True
    th.start()
    rospy.sleep(0.5)

    data = np.load(REPLAY_Q_FILE, allow_pickle=True)
    rec_q = data["q"].astype(np.float32)
    assert rec_q.ndim == 2 and rec_q.shape[1] == 16, rec_q.shape
    if "joints" in data.files:
        assert list(data["joints"]) == PUBLISH_JOINTS, (
            list(data["joints"]), PUBLISH_JOINTS,
        )
    rospy.loginfo(
        "Loaded %s  T=%d for warmup x%d", REPLAY_Q_FILE, len(rec_q), WARMUP_REPEATS
    )

    input(
        "HAND EMPTY — cup pose ready. Press Enter to start %d x q-replay warmup..."
        % WARMUP_REPEATS
    )

    first = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    publish_target(pub, first, 2.0)
    rospy.sleep(3.0)

    warmup = {"t": [], "episode": [], "cmd": [], "q": [], "fsr": [], "biotac_pdc": []}
    for ep in range(WARMUP_REPEATS):
        rospy.loginfo("Warmup replay %d/%d", ep + 1, WARMUP_REPEATS)
        run_q_replay(pub, rec_q, ep, warmup)
        rospy.sleep(0.3)

    fsr_arr = np.asarray(warmup["fsr"], dtype=np.float32)
    bt_arr = np.asarray(warmup["biotac_pdc"], dtype=np.float32)
    fit_thresholds_from_warmup(fsr_arr, bt_arr)

    fsr_state[:] = False
    bt_state[:] = False
    tac_warmup = []
    for i in range(len(fsr_arr)):
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
        speed_frac=np.float32(SPEED_FRAC if SPEED_FRAC is not None else -1.0),
        pos_err_mode=np.array("mlp"),
        pos_err_mlp=POS_ERR_MLP_PATH,
        rmse_gate_rmse=np.float32(metrics["rmse"]),
        rmse_gate_corr=np.float32(metrics["corr"]),
    )
    rospy.loginfo("Saved warmup sensors -> %s", WARMUP_NPZ)

    policy_start_q = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    rospy.loginfo("Warmup done. Moving to sim q[0] ...")
    publish_target(pub, policy_start_q, 2.0)
    rospy.sleep(3.0)

    input("At sim q[0]. PLACE BALLS in cup, then press Enter to start POLICY...")

    rospy.loginfo("Loading policy checkpoint: %s", CHECKPOINT)
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
    encoder, policy = Encoder(), Policy()
    e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    verify_encoder_load(e_res, CHECKPOINT, rospy.loginfo)
    policy.load_state_dict(
        {k: v for k, v in ckpt["policy"].items() if k != "log_std_parameter"},
        strict=True,
    )
    encoder.eval()
    policy.eval()

    fsr_state[:] = False
    bt_state[:] = False
    last_action[:] = 0.0
    pos_hist.clear()
    vel_hist.clear()
    act4.clear()
    p0, pred0, pred0_raw, _, _ = build_prop_mlp()
    t0 = read_tactile()
    prop_buffer.clear()
    tactile_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())
        tactile_buffer.append(t0.copy())

    rospy.sleep(1.0)
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = None
    rec = {
        "t": [], "q": [], "cmd": [], "tac": [], "fsr": [], "biotac_pdc": [],
        "act": [], "pred_pos_err": [], "pred_pos_err_raw": [],
    }
    rospy.loginfo(
        "Policy loop @ %d Hz | MLP pos_err | plant SPEED_FRAC=%s | "
        "POS_ERR_SPEED_FRAC=%s (obs x%s) | real tactile",
        CONTROL_HZ, SPEED_FRAC, POS_ERR_SPEED_FRAC, mlp_pos_err_scale(),
    )

    try:
        while not rospy.is_shutdown():
            prop, pred_err, pred_raw, _, _ = build_prop_mlp()
            prop_buffer.append(prop)
            tactile_buffer.append(read_tactile())
            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape

            with torch.no_grad():
                action = (
                    policy(encoder(torch.from_numpy(obs).unsqueeze(0)))
                    .numpy()[0]
                    .astype(np.float32)
                )

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
            rec["pred_pos_err"].append(pred_err.copy())
            rec["pred_pos_err_raw"].append(pred_raw.copy())

            # Update MLP action history: push previous cur into act4, then set cur.
            act4.append(last_action.copy())
            last_action[:] = action
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
                pos_err_mode=np.array("mlp"),
                pos_err_mlp=POS_ERR_MLP_PATH,
                speed_frac=np.float32(SPEED_FRAC if SPEED_FRAC is not None else -1.0),
                pos_err_speed_frac=np.float32(
                    POS_ERR_SPEED_FRAC if POS_ERR_SPEED_FRAC is not None else -1.0
                ),
                mlp_pos_err_scale=np.float32(mlp_pos_err_scale()),
                checkpoint=CHECKPOINT,
                rmse_gate_rmse=np.float32(metrics["rmse"]),
                rmse_gate_corr=np.float32(metrics["corr"]),
            )
            rospy.loginfo("saved %s: %d steps", POLICY_NPZ, len(rec["t"]))


def main():
    parser = argparse.ArgumentParser(description="Trial-15 tactile + MLP pos_err deploy")
    parser.add_argument(
        "--rmse-check",
        action="store_true",
        help="Only run offline MLP RMSE gate (no ROS / no HW).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deploy even if RMSE gate fails.",
    )
    parser.add_argument("--mlp", default=POS_ERR_MLP_PATH, help="pos_error_mlp.pt path")
    parser.add_argument("--rmse-log", default=RMSE_LOG_FILE, help="sim log for RMSE")
    args = parser.parse_args()

    if args.rmse_check:
        ok, _ = rmse_check(args.mlp, args.rmse_log, verbose=True)
        sys.exit(0 if ok else 1)

    run_deploy(force=args.force)


if __name__ == "__main__":
    main()
