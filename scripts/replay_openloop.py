"""Replay a recorded joint trajectory in sim, open-loop: no policy, no checkpoint.

The simulation counterpart of ``deploy_openloop_aug4_trial5.py``. That script
publishes a recorded trajectory at the real hand; this one publishes the same
trajectory at the simulated hand, so the two clips show the same commanded
motion driving two different plants. Nothing here loads a checkpoint -- the
policy already ran, inside ``record_openloop.py``, and its output is baked into
the npz. That is the whole point: whatever happens here is what the trajectory
alone produces, with no feedback correcting it.

How the trajectory gets in: ``_apply_action`` (roto_env.py:539-546) reads only
``joint_pos_cmd[:, actuated_dof_indices]`` and drives
``robot.set_joint_position_target(...)`` -- a real PD target, so the balls are
pushed by the fingers rather than teleported. So replacing ``_pre_physics_step``
with one that writes the recorded 16-vector straight into ``joint_pos_cmd`` is a
complete open-loop replay. Same monkey-patch idiom ``record_openloop.py`` uses
for ``_get_dones``.

Deliberately bypassed: ``scale()``, ``_handle_coupled_joints()`` and
``_apply_cmd_slew()``. The npz already holds all 16 POST-coupling joint values,
and the forward coupling is not cleanly invertible anyway
(``couple_gate_j1_on_measured=True`` gates J1 on *measured* J2), so re-deriving a
13-d action would introduce error rather than remove it.

Usage:
    python replay_openloop.py --npz ../openloop_aug4_trial5_60s_seed42.npz --source cmd \
        --headless --video --max_steps 1200 \
        --cam_res 3840 2160 --hdr qwantani_dusk_2_puresky_4k \
        --cam_eye 0.0016 -0.5238 0.5494 --cam_lookat 0.0016 -0.2438 0.4094
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Open-loop replay of a recorded joint trajectory in sim (no policy)."
)
parser.add_argument("--npz", type=str, required=True,
                    help="Trajectory from record_openloop.py: needs q (T,16) + joints.")
parser.add_argument("--source", type=str, default="cmd", choices=["q", "cmd"],
                    help="Which column to replay as the PD target. Default 'cmd' -- the targets "
                         "the policy actually issued. Do NOT use 'q' unless you know why: "
                         "measured in sim, replaying the achieved positions gives 0 rotations "
                         "vs 9 for 'cmd' over the same 10 s, because a PD plant lags whatever "
                         "it is given, so commanding the already-lagged q under-travels the "
                         "fingers and they never complete a cycle. Against the ORIGINAL achieved "
                         "q, 'cmd' reproduces the run to 0.003 rad mean; 'q' is off by 0.052.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite_padtac_bt")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt",
                    help="Only used to build a matching env; no policy is loaded from it.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=None,
                    help="Replay only the first N frames (default: the whole file).")
parser.add_argument("--out", type=str, default=None, help="Output npz for the replayed run.")

parser.add_argument("--ball_mass_g", type=float, default=55.0)
parser.add_argument("--keep_ball_disturb", action="store_true", default=False)
parser.add_argument("--fsr_corrupt_max", type=int, default=0)
parser.add_argument("--tactile_flip_prob", type=float, default=0.0)

parser.add_argument("--cam_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--cam_lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--cam_res", type=int, nargs=2, default=None, metavar=("W", "H"))
parser.add_argument("--hdr", type=str, default=None)

parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=None)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--renderer", type=str, default="RayTracedLighting",
                    choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict

from multimodal_rl.tools.writer import Writer

PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]


def main():
    data = np.load(args_cli.npz, allow_pickle=True)
    Q_np = data[args_cli.source].astype(np.float32)
    if Q_np.ndim != 2 or Q_np.shape[1] != 16:
        raise SystemExit(f"{args_cli.npz}['{args_cli.source}'] must be (T,16), got {Q_np.shape}")
    if "joints" in data.files and list(data["joints"]) != PUBLISH_JOINTS:
        raise SystemExit(
            "npz joint order does not match the env's actuated order:\n"
            f"  npz: {list(data['joints'])}\n  env: {PUBLISH_JOINTS}\n"
            "Replaying it would send each target to the wrong joint."
        )
    if args_cli.max_steps is not None:
        Q_np = Q_np[: int(args_cli.max_steps)]

    # A trajectory spliced across an env reset would teleport the hand mid-clip.
    gap = float(np.abs(np.diff(Q_np, axis=0)).max()) if len(Q_np) > 1 else 0.0
    if gap > 0.15:
        raise SystemExit(
            f"{args_cli.npz} jumps {gap:.3f} rad in one frame -- that is a reset "
            "discontinuity, not a continuous trajectory. Re-record with record_openloop.py."
        )

    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    agent_cfg = update_dict(agent_cfg, load_hand_task_agent_cfg(
        args_cli.task, args_cli.robot, args_cli.agent_cfg))

    agent_cfg["seed"] = args_cli.seed
    set_seed(args_cli.seed)
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    steps_per_s = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    n_frames = len(Q_np)

    # Reset-free: stretch the episode past the whole replay so the time-out reset
    # never lands inside it.
    needed_s = n_frames / steps_per_s + 2.0
    if env_cfg.episode_length_s < needed_s:
        print(f"[INFO] episode_length_s {env_cfg.episode_length_s:g} -> {needed_s:.1f} s")
        env_cfg.episode_length_s = needed_s

    # Deterministic start: the stock reset perturbs the initial pose, which would
    # put the hand somewhere the recorded frame 0 does not expect.
    env_cfg.reset_joint_pos_noise = 0.0

    env_cfg.tactile_fsr_corrupt_max = (
        None if args_cli.fsr_corrupt_max <= 0 else int(args_cli.fsr_corrupt_max))
    env_cfg.tactile_flip_prob_off_to_on = float(args_cli.tactile_flip_prob)
    env_cfg.tactile_flip_prob_on_to_off = float(args_cli.tactile_flip_prob)
    if not args_cli.keep_ball_disturb:
        env_cfg.ball_push_vel_range = None
        env_cfg.ball_push_angvel_range = None
        env_cfg.ball_force_range = None
        env_cfg.ball_torque_range = None
        print("[INFO] ball disturbance DR forced OFF")

    if args_cli.hdr is not None:
        import roto.tasks.baoding.baoding as _b
        hdr_path = _b._BAODING_HDR.parent / f"{args_cli.hdr}.hdr"
        if not hdr_path.is_file():
            raise SystemExit(f"Unknown HDRI {args_cli.hdr!r}: no such file {hdr_path}")
        _b._BAODING_HDR = hdr_path
        print(f"[INFO] HDRI background -> {hdr_path}")

    # Read once, lazily, at the first render(): must be set before make_env or it
    # silently stays at the old resolution.
    if args_cli.cam_res is not None:
        env_cfg.viewer.resolution = tuple(args_cli.cam_res)
        print(f"[INFO] render resolution -> {env_cfg.viewer.resolution}")
    if (args_cli.cam_eye is None) != (args_cli.cam_lookat is None):
        raise SystemExit("--cam_eye and --cam_lookat must be given together.")
    if args_cli.cam_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        print(f"[INFO] camera eye={env_cfg.viewer.eye} lookat={env_cfg.viewer.lookat}")

    if args_cli.video and args_cli.video_length is None:
        args_cli.video_length = n_frames + 1

    writer = Writer(agent_cfg, play=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    raw = env.env.unwrapped

    m_kg = args_cli.ball_mass_g / 1000.0
    raw.cfg.ball_mass_range = (m_kg, m_kg)
    print(f"[INFO] ball mass pinned to {args_cli.ball_mass_g:g} g")

    idx = raw.actuated_dof_indices
    joints16 = [raw.robot.joint_names[i] for i in idx]
    if joints16 != PUBLISH_JOINTS:
        raise SystemExit(f"env actuated order {joints16} != {PUBLISH_JOINTS}")

    Q = torch.as_tensor(Q_np, device=raw.device)

    # Suppress every termination so a dropped ball cannot splice an auto-reset
    # teleport into the middle of the clip. The step it would have dropped at is
    # still recorded.
    orig_get_dones = raw._get_dones
    drop = {"step": -1}
    clock = {"t": 0}

    def _no_dones():
        termination, time_out = orig_get_dones()
        if drop["step"] < 0 and bool(termination[0].item()):
            drop["step"] = clock["t"]
        return torch.zeros_like(termination), torch.zeros_like(time_out)

    raw._get_dones = _no_dones

    # The real _pre_physics_step holds default_joint_pos while settle_counter runs
    # down (balls still falling). Feeding Q[0] over that window keeps the hand
    # still until the balls have landed, instead of starting mid-motion.
    settle_steps = int(getattr(raw.cfg, "settle_steps", 0) or 0)

    def _replay_pre_physics(actions):
        raw.prev_joint_pos_cmd[:] = raw.joint_pos_cmd
        t = clock["t"] - settle_steps
        t = 0 if t < 0 else min(t, Q.shape[0] - 1)
        raw.joint_pos_cmd[:, idx] = Q[t]

    raw._pre_physics_step = _replay_pre_physics

    total = n_frames + settle_steps
    print(f"[INFO] replaying {n_frames} frames ({n_frames / steps_per_s:.1f} s) "
          f"from '{args_cli.source}', + {settle_steps} settle steps")

    prop_idx = raw.prop_dof_indices          # 13 policy-order control joints

    dummy = torch.zeros((raw.num_envs, raw.cfg.num_actions), device=raw.device)
    # Same schema as record_openloop.py / play.py so every condition's log can be
    # compared column-for-column, open-loop or not.
    rec = {
        "q": [],            # (16) achieved joint pos, rad
        "cmd": [],          # (16) joint_pos_cmd actually applied, rad
        "ref": [],          # (16) the recorded trajectory frame we asked for
        "qd": [],           # (16) joint vel, rad/s
        "pos_err": [],      # (16) cmd - q, rad
        "q13": [], "qd13": [], "cmd13": [], "pos_err13": [],
        "tac": [],          # (24) binary tactile
        "num_rotations": [],
    }

    with torch.inference_mode():
        env.reset(hard=True)
        t0 = time.time()
        for t in range(total):
            if not simulation_app.is_running():
                print("[WARN] simulator closed early")
                break
            clock["t"] = t
            env.step(dummy)
            if t < settle_steps:
                continue
            k = t - settle_steps
            rec["q"].append(raw.robot.data.joint_pos[0, idx].detach().cpu().numpy().copy())
            rec["cmd"].append(raw.joint_pos_cmd[0, idx].detach().cpu().numpy().copy())
            rec["ref"].append(Q_np[min(k, n_frames - 1)])
            rec["qd"].append(raw.joint_vel[0, idx].detach().cpu().numpy().copy())
            rec["pos_err"].append(raw.joint_pos_error[0, idx].detach().cpu().numpy().copy())
            rec["q13"].append(raw.joint_pos[0, prop_idx].detach().cpu().numpy().copy())
            rec["qd13"].append(raw.joint_vel[0, prop_idx].detach().cpu().numpy().copy())
            rec["cmd13"].append(raw.joint_pos_cmd[0, prop_idx].detach().cpu().numpy().copy())
            rec["pos_err13"].append(raw.joint_pos_error[0, prop_idx].detach().cpu().numpy().copy())
            rec["tac"].append(raw.tactile[0].detach().cpu().numpy().copy())
            rec["num_rotations"].append(float(raw.num_rotations[0].item()))
            if (k + 1) % 600 == 0:
                print(f"  ... {k + 1}/{n_frames}  rotations={rec['num_rotations'][-1]:.1f}  "
                      f"({time.time() - t0:.0f}s wall)")

    q = np.asarray(rec["q"], dtype=np.float32)
    ref = np.asarray(rec["ref"], dtype=np.float32)
    err = np.abs(q - ref)

    out = args_cli.out or (
        f"replay_{os.path.splitext(os.path.basename(args_cli.npz))[0]}_{len(q)}steps.npz")
    np.savez(
        out,
        **{k: np.asarray(v, dtype=np.float32) for k, v in rec.items()},
        joints=np.array(PUBLISH_JOINTS),
        joints13=np.array([raw.robot.joint_names[i] for i in prop_idx]),
        source_npz=np.array(os.path.abspath(args_cli.npz)),
        source_col=np.array(args_cli.source),
        control_hz=np.float32(steps_per_s),
        ball_mass_g=np.float32(args_cli.ball_mass_g),
        drop_step=np.int32(drop["step"]),
        settle_steps=np.int32(settle_steps),
        open_loop=np.bool_(True),
    )

    print("\n" + "=" * 68)
    print(f"saved {out}")
    print(f"  frames replayed : {len(q)}  ({len(q) / steps_per_s:.1f} s)")
    print(f"  rotations       : {rec['num_rotations'][-1]:.1f}" if len(q) else "  rotations: n/a")
    print(f"  balls dropped   : {'no' if drop['step'] < 0 else 'YES at step %d' % drop['step']}")
    # Tracking error against the reference. Large error here means the replay is
    # not reproducing the recorded motion -- a bad sign for the whole comparison.
    print(f"  |q - ref|       : mean={err.mean():.5f}  max={err.max():.5f} rad "
          f"(worst joint {PUBLISH_JOINTS[int(err.mean(axis=0).argmax())]})")
    tac = np.asarray(rec["tac"], dtype=np.float32)
    print(f"  tactile         : mean={tac.mean():.6f}  frac_on={100.0 * (tac > 0.5).mean():.2f}%")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    main()
