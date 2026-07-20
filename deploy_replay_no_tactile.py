#!/usr/bin/env python
"""Open-loop replay of a sim joint trajectory on the Shadow Hand Lite (ROS).

For Ayush / hands without FSR or BioTac wiring:
  - NO serial FSR
  - NO BioTac topics
  - NO policy / checkpoint / torch
  - Publishes measured sim ``q`` (16 joints) from a play.py npz

Expected npz keys (from scripts/play.py):
  q      : (T, 16) achieved sim joint positions [rad]
  joints : (16,) names matching PUBLISH_JOINTS below

Usage on control laptop:
  1. Copy this file + the npz next to each other (or set REPLAY_FILE path).
  2. roslaunch your hand bringup as usual.
  3. python deploy_replay_no_tactile.py

Does not modify deploy_policy_new.py.
"""

from __future__ import print_function

import rospy
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ============================================================
# CONFIG — edit these on the laptop
# ============================================================
REPLAY_FILE = "sim_policy_log_padtac_new_rollout.npz"
CONTROL_HZ = 60
# Optional velocity governor: None = off (match full-speed sim).
# 0.25 = slow bring-up. 1.0 = rated-speed safety cap only.
SPEED_FRAC = None
OUT_LOG = "hw_q_replay_no_tactile.npz"

# 16-joint publish order (must match play.py actuated_dof_indices / npz["joints"])
PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]

# Hardware-safe clamps (Shadow Hand Lite: J2/J1 max 90 deg)
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

# Measured joint state (all 16 publish joints)
current_joint_pos = np.zeros(16, dtype=np.float32)
current_joint_vel = np.zeros(16, dtype=np.float32)
joint_ready = False


def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    idx = dict((n, i) for i, n in enumerate(msg.name))
    try:
        for i, j in enumerate(PUBLISH_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


def apply_governor(target, prev_pub):
    if prev_pub is None or SPEED_FRAC is None:
        return target
    max_delta = PUB_VEL_LIMIT * float(SPEED_FRAC) / float(CONTROL_HZ)
    return prev_pub + np.clip(target - prev_pub, -max_delta, max_delta)


def publish_traj(pub, positions, duration_s):
    msg = JointTrajectory()
    msg.joint_names = PUBLISH_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = np.asarray(positions, dtype=np.float32).tolist()
    pt.time_from_start = rospy.Duration(duration_s)
    msg.points.append(pt)
    pub.publish(msg)


def main():
    rospy.init_node("deploy_replay_no_tactile")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
    rospy.Subscriber("/joint_states", JointState, joint_callback)

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states received.")

    data = np.load(REPLAY_FILE, allow_pickle=True)
    if "q" not in data.files:
        raise KeyError("REPLAY_FILE missing 'q'. Need a play.py sim log with measured joint positions.")
    rec_q = data["q"].astype(np.float32)
    if "joints" in data.files:
        joints = [str(x) for x in data["joints"]]
        if joints != PUBLISH_JOINTS:
            raise AssertionError(
                "npz joint order mismatch.\n  file=%s\n  expected=%s" % (joints, PUBLISH_JOINTS)
            )
    if rec_q.ndim != 2 or rec_q.shape[1] != 16:
        raise ValueError("Expected q shape (T, 16), got %s" % (rec_q.shape,))

    rospy.loginfo(
        "Loaded %s: %d steps @ %d Hz (~%.1f s). No FSR/BioTac.",
        REPLAY_FILE, rec_q.shape[0], CONTROL_HZ, rec_q.shape[0] / float(CONTROL_HZ),
    )
    input("Hand ready? Press Enter to move to first pose, then replay...")

    first = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    publish_traj(pub, first, 2.0)
    rospy.sleep(3.0)

    log = {"t": [], "cmd": [], "q": [], "qd": []}
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = first.copy()

    for q_sim in rec_q:
        if rospy.is_shutdown():
            break
        target = np.clip(q_sim, PUB_LOWER, PUB_UPPER)
        target = apply_governor(target, prev_pub).astype(np.float32)
        prev_pub = target
        publish_traj(pub, target, 1.0 / float(CONTROL_HZ))

        log["t"].append(rospy.get_time())
        log["cmd"].append(target.copy())
        log["q"].append(current_joint_pos.copy())
        log["qd"].append(current_joint_vel.copy())
        rate.sleep()

    np.savez(OUT_LOG, **dict((k, np.array(v)) for k, v in log.items()), joints=PUBLISH_JOINTS)
    rospy.loginfo("Saved %s (%d steps). q/cmd are 16-D in PUBLISH_JOINTS order.", OUT_LOG, len(log["t"]))


if __name__ == "__main__":
    main()
