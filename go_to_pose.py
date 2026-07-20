
#!/usr/bin/env python
import rospy, numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

POLICY_JOINTS = [
    'rh_FFJ4','rh_MFJ4','rh_RFJ4','rh_THJ5',
    'rh_FFJ3','rh_MFJ3','rh_RFJ3','rh_THJ4',
    'rh_FFJ2','rh_MFJ2','rh_RFJ2',
    'rh_FFJ1','rh_MFJ1','rh_RFJ1',
    'rh_THJ2','rh_THJ1',
]
JOINT_LOWER = np.array([-0.3491,-0.3491,-0.3491,-1.0472,-0.2618,-0.2618,-0.2618,
                        0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.6981,-0.2618], dtype=np.float32)
JOINT_UPPER = np.array([0.3491,0.3491,0.3491,1.0472,1.5708,1.5708,1.5708,
                        1.2217,1.5708,1.5708,1.5708,1.5708,1.5708,1.5708,0.6981,1.5708], dtype=np.float32)
COUPLED_PAIRS = [(11,8),(12,9),(13,10)]

# sim reset pose (q[0]), POLICY_JOINTS order
#POSE = np.array([
 #   0.0,   # FFJ4  spread ~neutral
  #  0.0,   # MFJ4
   # 0.0,   # RFJ4
    #0.9,   # THJ5  thumb OPPOSITION (the big one for a cup)
    #0.6,   # FFJ3  proximal flex (curl finger up)
    #0.6,   # MFJ3
#    0.6,   # RFJ3
 #   0.7,   # THJ4  thumb flex
  #  0.5,   # FFJ2  middle flex (curl tip inward)
   # 0.5,   # MFJ2
    #0.5,   # RFJ2
#    0.5,   # FFJ1  distal (kept = J2 so coupling is happy)
 #   0.5,   # MFJ1
  #  0.5,   # RFJ1
   # 0.0,   # THJ2
    #0.5,   # THJ1  thumb distal flex
#], dtype=np.float32)

#POSE = np.array([-0.1342,-0.0373,-0.0021,0.1088,0.1139,0.0776,-0.0433,0.0,
#                 0.0838,0.1647,0.1819,0.0,0.0879,0.0743,0.0236,-0.2297], dtype=np.float32)

# Sim reset/cup pose (ShadowLiteEnvCfg init_state), verified from the settle steps of
# sim_policy_log_seed42.npz (joint_pos_cmd during reset == default_joint_pos).
# PUBLISH order = POLICY_JOINTS. NOTE: FFJ1/MFJ1/RFJ1 = 0.0 in sim (distal straight);
# the deploy coupling keeps them ~0, so starting there avoids a jump when the policy
# takes over. THJ1 = 0.0 too.
POSE = np.array([
    -0.349,  # FFJ4
     0.0,    # MFJ4
    -0.349,  # RFJ4
     0.4,    # THJ5
     0.65,   # FFJ3
     0.65,   # MFJ3
     0.65,   # RFJ3
     0.5,    # THJ4
     0.87,   # FFJ2  (~50 deg proximal-interphalangeal curl)
     0.87,   # MFJ2
     0.87,   # RFJ2
     0.0,    # FFJ1  (sim reset = 0; distal straight)
     0.0,    # MFJ1
     0.0,    # RFJ1
     0.35,   # THJ2
     0.0,    # THJ1
], dtype=np.float32)

MOVE_TIME = 4.0   # slow move to cup; increase if it still snaps

def main():
    rospy.init_node("go_to_pose")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)

    # wait for the controller to actually connect, else the message is dropped
    t0 = rospy.get_time()
    while pub.get_num_connections() == 0 and rospy.get_time()-t0 < 5.0 and not rospy.is_shutdown():
        rospy.sleep(0.1)

    pose = np.clip(POSE, JOINT_LOWER, JOINT_UPPER)
    for j1, j2 in COUPLED_PAIRS:
        pose[j1] = min(pose[j1], pose[j2])     # respect J1 <= J2

    msg = JointTrajectory()
    msg.joint_names = POLICY_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = pose.tolist()
    pt.time_from_start = rospy.Duration(MOVE_TIME)
    msg.points.append(pt)

    rospy.loginfo("Sending start pose...")
    pub.publish(msg)
    rospy.sleep(MOVE_TIME + 0.5)
    rospy.loginfo("At pose. Try to seat the balls now.")

if __name__ == "__main__":
    main()

