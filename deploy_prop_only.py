#!/usr/bin/env python
"""Deploy Ayush prop-only Baoding policy on Shadow Hand Lite (ROS).

Observation contract (prop only, no tactile):
  - 208-d = 4 * prop(52)
  - prop = [pos_norm(13), vel_norm(13), cmd_error(13), last_action(13)]
  - 13 actions; FFJ2/MFJ2/RFJ2 are combined-curl proxies (same coupling as deploy_policy_new)

No FSR serial, no BioTac. Does not modify deploy_policy_new.py.

Laptop:
  1. Copy this file + best_agent_legacy_new_coup_prop_only.pt
  2. Set CHECKPOINT path below
  3. python deploy_prop_only.py
"""

from __future__ import print_function

import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ============================================================
# MODEL — prop-only (encoder first layer in_features = 208)
# ============================================================
OBS_DIM = 208
PROP_DIM = 52
NUM_J = 13
OBS_STACK = 4
CONTROL_HZ = 60

CHECKPOINT = "/home/user/experiments/best_agent_legacy_new_coup_prop_only.pt"
OUT_LOG = "hw_policy_log_prop_only.npz"
SPEED_FRAC = None


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


current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
last_action = np.zeros(NUM_J, dtype=np.float32)
last_command = np.zeros(NUM_J, dtype=np.float32)
joint_ready = False
prop_buffer = deque(maxlen=OBS_STACK)


def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    idx = dict((n, i) for i, n in enumerate(msg.name))
    try:
        for i, j in enumerate(POLICY_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


def build_prop():
    pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
    vel_norm = current_joint_vel / JOINT_VEL_LIMIT
    error = last_command - current_joint_pos
    return np.concatenate([pos_norm, vel_norm, error, last_action]).astype(np.float32)


def main():
    global last_action, last_command

    rospy.init_node("deploy_prop_only")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
    rospy.Subscriber("/joint_states", JointState, joint_callback)

    rospy.loginfo("Loading prop-only checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    encoder, policy = Encoder(), Policy()
    e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    rospy.loginfo("encoder load missing=%s unexpected=%s", e_res.missing_keys, e_res.unexpected_keys)
    # First linear must be 208-d
    w0 = ckpt["encoder"]["net.0.weight"]
    if tuple(w0.shape) != (1024, OBS_DIM):
        raise RuntimeError(
            "Checkpoint encoder in_features=%s, expected %s (prop-only). Wrong file?"
            % (w0.shape[1], OBS_DIM)
        )
    policy.load_state_dict(
        dict((k, v) for k, v in ckpt["policy"].items() if k != "log_std_parameter"),
        strict=True,
    )
    encoder.eval()
    policy.eval()

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states received. Prop-only OBS_DIM=%d (no FSR/BioTac).", OBS_DIM)

    input("Place balls (optional), then press Enter to start prop-only policy...")

    last_command[:] = current_joint_pos
    last_action[:] = 0.0
    p0 = build_prop()
    prop_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())

    rospy.sleep(1.0)
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = None
    rec = {"t": [], "q": [], "cmd": [], "act": []}

    try:
        while not rospy.is_shutdown():
            prop_buffer.append(build_prop())
            obs = np.concatenate(list(prop_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)

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

            rec["t"].append(rospy.get_time())
            rec["q"].append(current_joint_pos.copy())
            rec["cmd"].append(pub_target.copy())
            rec["act"].append(action.copy())

            last_action[:] = action
            last_command[:] = raw_cmd
            last_command[CURL_J2_IDX] = j2_cmd
            rate.sleep()
    finally:
        if rec["t"]:
            np.savez(OUT_LOG, **dict((k, np.array(v)) for k, v in rec.items()))
            rospy.loginfo("saved %s: %d steps", OUT_LOG, len(rec["t"]))


if __name__ == "__main__":
    main()
