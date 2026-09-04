"""Record a continuous open-loop joint trajectory from a Baoding policy checkpoint.

Why this exists (and why play.py is not enough): play.py hard-resets the env every
``max_episode_length - 1`` steps (10 s / 600 steps for ShadowLite), so anything longer
than one episode comes back with reset discontinuities baked in -- jumps that would be
published straight at the hardware. This script runs ONE uninterrupted rollout:

  * ``episode_length_s`` is stretched to cover the whole recording, and
  * ``_get_dones`` is suppressed for the duration, so a dropped ball does not
    trigger an auto-reset mid-trajectory (the step it *would* have dropped at is
    recorded as ``drop_step`` and printed loudly instead).

Everything the domain randomisation would otherwise scramble is pinned to the nominal
deploy condition by default (clean tactile, no ball disturbance, 55 g balls, fixed
command slew), because the output is a trajectory that gets published to a real hand,
not a robustness eval.

Output npz is key-compatible with ``play.py``'s ``sim_policy_log_seed*.npz``
(``q`` (T,16) + ``joints``), so it drops straight into the deploy scripts'
replay path -- ``deploy_openloop_aug4_trial5.py``, ``deploy_replay_no_tactile.py``,
and the Phase-A warmup of ``deploy_warmup_trial15_zerotac.py``.

Usage (60 s @ 60 Hz):
    python record_openloop.py \
        --checkpoint ../best_agent_aug4_slew_fsr_no_noise_trial5.pt \
        --record_steps 3600 --headless --out openloop_aug4_trial5_60s.npz
"""

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Record one continuous (reset-free) open-loop trajectory from a policy checkpoint."
)
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite_padtac_bt")
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt_sweep")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=1, help="Only env 0 is recorded; keep this small.")
parser.add_argument(
    "--record_steps", type=int, default=3600,
    help="Control steps to record @ 60 Hz (3600 = 60 s, 1800 = 30 s).",
)
parser.add_argument(
    "--settle_steps", type=int, default=30,
    help="Steps to run and DISCARD after reset before recording starts, so the "
         "trajectory does not open with the ball-settle transient.",
)
parser.add_argument("--out", type=str, default=None, help="Output npz path (default: auto-named).")

# --- plant / DR pinning (deploy-nominal defaults, all overridable) ---
parser.add_argument(
    "--cmd_speed_frac", type=float, default=0.6,
    help="Fixed command-rate limiter in sim, matching the hardware deploy SPEED_FRAC "
         "so the recorded motion is already the slewed motion. Pass -1 to disable slew.",
)
parser.add_argument(
    "--ball_mass_g", type=float, default=55.0,
    help="Pin both balls to this mass (g), overriding mass DR. Match the real balls.",
)
parser.add_argument(
    "--fsr_corrupt_max", type=int, default=0,
    help="Per-episode FSR taxel corruption DR. 0 = clean tactile (default for a deploy recording).",
)
parser.add_argument(
    "--tactile_flip_prob", type=float, default=0.0,
    help="Per-step taxel dither probability (both directions). 0 = off.",
)
parser.add_argument(
    "--keep_ball_disturb", action="store_true", default=False,
    help="Leave the random ball push/force DR ON (default: forced off for a clean recording).",
)
parser.add_argument(
    "--zero_tactile", action="store_true", default=False,
    help="Feed the policy all-zero tactile at the source (prop-only rollout).",
)
parser.add_argument(
    "--allow_drop", action="store_true", default=False,
    help="Keep recording after the balls would have dropped instead of stopping. "
         "The tail is the hand cycling on nothing -- rarely what you want on hardware.",
)
parser.add_argument("--video", action="store_true", default=False, help="Also record a video of the rollout.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"],
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--video_length", type=int, default=None)

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
    make_models,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict

from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer

# 16-joint publish order the deploy scripts assert against (PUBLISH_JOINTS).
PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task not in ("Bounce", "Baoding"):
        raise ValueError("record_openloop.py only supports --task Bounce/Baoding.")
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    steps_per_s = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    total_steps = int(args_cli.settle_steps) + int(args_cli.record_steps)

    # Stretch the episode past the whole recording so the time-out reset never lands
    # inside it. +2 s of slack keeps us clear of the max_episode_length - 1 boundary.
    needed_s = total_steps / steps_per_s + 2.0
    if env_cfg.episode_length_s < needed_s:
        print(
            f"[INFO] episode_length_s {env_cfg.episode_length_s:g} -> {needed_s:.1f} s "
            f"(covers {total_steps} steps @ {steps_per_s:g} Hz, reset-free)"
        )
        env_cfg.episode_length_s = needed_s

    # These are read once at env construction, so they must be set before make_env.
    env_cfg.tactile_fsr_corrupt_max = (
        None if args_cli.fsr_corrupt_max <= 0 else int(args_cli.fsr_corrupt_max)
    )
    print(f"[INFO] tactile_fsr_corrupt_max -> {env_cfg.tactile_fsr_corrupt_max}")
    env_cfg.tactile_flip_prob_off_to_on = float(args_cli.tactile_flip_prob)
    env_cfg.tactile_flip_prob_on_to_off = float(args_cli.tactile_flip_prob)
    print(f"[INFO] tactile_flip_prob -> {args_cli.tactile_flip_prob:g} (both directions)")

    if args_cli.cmd_speed_frac is not None and args_cli.cmd_speed_frac >= 0:
        env_cfg.cmd_speed_frac = float(args_cli.cmd_speed_frac)
        env_cfg.cmd_speed_frac_range = None
        print(f"[INFO] cmd slew FIXED at {env_cfg.cmd_speed_frac:g} (matches hardware SPEED_FRAC)")
    else:
        env_cfg.cmd_speed_frac = None
        env_cfg.cmd_speed_frac_range = None
        print("[INFO] cmd slew OFF")

    if not args_cli.keep_ball_disturb:
        env_cfg.ball_push_vel_range = None
        env_cfg.ball_push_angvel_range = None
        env_cfg.ball_force_range = None
        env_cfg.ball_torque_range = None
        print("[INFO] ball disturbance DR forced OFF")

    if args_cli.video and args_cli.video_length is None:
        args_cli.video_length = total_steps + 1

    writer = Writer(agent_cfg, play=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder, policy, value, value_preprocessor,
        memory=None, cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer, ssl_task=None, dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    raw = env.env.unwrapped

    # Ball mass: the env samples per-env from cfg.ball_mass_range at every reset, so a
    # fixed mass is just the degenerate range (m, m) -- reuses the env's own inertia code.
    m_kg = args_cli.ball_mass_g / 1000.0
    raw.cfg.ball_mass_range = (m_kg, m_kg)
    print(f"[INFO] ball mass pinned to {args_cli.ball_mass_g:g} g")

    if args_cli.zero_tactile:
        if raw.tactile_cfg is None:
            raw.tactile_cfg = {"binary_tactile": True, "binary_threshold": 0.0}
        raw.tactile_cfg["zero_tactile"] = True
        print("[INFO] tactile forced to ZERO at the source")

    idx = raw.actuated_dof_indices          # 16: full actuated (incl. J1 mimics)
    prop_idx = raw.prop_dof_indices         # 13: policy control joints
    joints16 = [raw.robot.joint_names[i] for i in idx]
    if joints16 != PUBLISH_JOINTS:
        raise RuntimeError(
            "Actuated joint order does not match the deploy scripts' PUBLISH_JOINTS.\n"
            f"  env:    {joints16}\n  deploy: {PUBLISH_JOINTS}\n"
            "Replaying this npz would send each target to the wrong joint."
        )

    # Suppress every termination for the length of the recording. Isaac Lab auto-resets
    # on a done, which would splice a teleport into the middle of a trajectory we are
    # about to publish at a real hand. We still evaluate the real dones to record the
    # step at which the balls were first lost.
    orig_get_dones = raw._get_dones
    drop_state = {"step": -1}
    step_counter = {"n": 0}

    def _no_dones():
        termination, time_out = orig_get_dones()
        if drop_state["step"] < 0 and bool(termination[0].item()):
            drop_state["step"] = step_counter["n"]
        return torch.zeros_like(termination), torch.zeros_like(time_out)

    raw._get_dones = _no_dones

    rec = {
        "act": [],          # (13,) policy output (unitless, [-1, 1])
        "q": [],            # (16,) achieved joint pos, rad  <- what gets replayed
        "cmd": [],          # (16,) joint_pos_cmd, rad
        "qd": [],           # (16,) joint vel, rad/s
        "pos_err": [],      # (16,) cmd - q, rad
        "q13": [], "qd13": [], "cmd13": [], "pos_err13": [],
        "tac": [],          # (24,) binary tactile the policy saw
        "num_rotations": [],
    }

    with torch.inference_mode():
        states, infos = env.reset(hard=True)

        for _ in range(int(args_cli.settle_steps)):
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, _, _, _, infos = env.step(actions)
            step_counter["n"] += 1
        drop_state["step"] = -1  # ignore anything during the discarded settle window
        step_counter["n"] = 0
        print(f"[INFO] settled ({args_cli.settle_steps} steps discarded); recording "
              f"{args_cli.record_steps} steps (~{args_cli.record_steps / steps_per_s:.1f} s)")

        t_start = time.time()
        for t in range(int(args_cli.record_steps)):
            if not simulation_app.is_running():
                print("[WARN] simulator closed early")
                break
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, rewards, terminated, truncated, infos = env.step(actions)
            step_counter["n"] = t

            rec["act"].append(actions[0].detach().cpu().numpy().copy())
            rec["q"].append(raw.robot.data.joint_pos[0, idx].detach().cpu().numpy().copy())
            rec["cmd"].append(raw.joint_pos_cmd[0, idx].detach().cpu().numpy().copy())
            rec["qd"].append(raw.joint_vel[0, idx].detach().cpu().numpy().copy())
            rec["pos_err"].append(raw.joint_pos_error[0, idx].detach().cpu().numpy().copy())
            rec["q13"].append(raw.joint_pos[0, prop_idx].detach().cpu().numpy().copy())
            rec["qd13"].append(raw.joint_vel[0, prop_idx].detach().cpu().numpy().copy())
            rec["cmd13"].append(raw.joint_pos_cmd[0, prop_idx].detach().cpu().numpy().copy())
            rec["pos_err13"].append(raw.joint_pos_error[0, prop_idx].detach().cpu().numpy().copy())
            rec["tac"].append(raw.tactile[0].detach().cpu().numpy().copy())
            rec["num_rotations"].append(float(raw.num_rotations[0].item()))

            if drop_state["step"] >= 0 and not args_cli.allow_drop:
                print(f"[WARN] balls lost at step {drop_state['step']} "
                      f"({drop_state['step'] / steps_per_s:.1f} s) -- stopping the recording "
                      "there. Re-run with a different --seed, or --allow_drop to keep going.")
                break

            if (t + 1) % 600 == 0:
                print(f"  ... {t + 1}/{args_cli.record_steps} steps  "
                      f"rotations={rec['num_rotations'][-1]:.2f}  "
                      f"({time.time() - t_start:.0f}s wall)")

    n = len(rec["act"])
    q = np.asarray(rec["q"], dtype=np.float32)

    out = args_cli.out
    if out is None:
        stem = os.path.splitext(os.path.basename(resume_path))[0].replace("best_agent_", "")
        out = f"openloop_{stem}_{n}steps_seed{agent_cfg['seed']}.npz"

    np.savez(
        out,
        **{k: np.asarray(v, dtype=np.float32) for k, v in rec.items()},
        joints=np.array(PUBLISH_JOINTS),
        joints13=np.array([raw.robot.joint_names[i] for i in prop_idx]),
        control_hz=np.float32(steps_per_s),
        checkpoint=np.array(resume_path),
        seed=np.int32(agent_cfg["seed"]),
        robot=np.array(args_cli.robot),
        agent_cfg_name=np.array(args_cli.agent_cfg),
        cmd_speed_frac=np.float32(
            -1.0 if env_cfg.cmd_speed_frac is None else env_cfg.cmd_speed_frac
        ),
        ball_mass_g=np.float32(args_cli.ball_mass_g),
        fsr_corrupt_max=np.int32(args_cli.fsr_corrupt_max),
        tactile_flip_prob=np.float32(args_cli.tactile_flip_prob),
        ball_disturb_off=np.bool_(not args_cli.keep_ball_disturb),
        zero_tactile=np.bool_(args_cli.zero_tactile),
        settle_steps=np.int32(args_cli.settle_steps),
        drop_step=np.int32(drop_state["step"]),
        reset_free=np.bool_(True),
    )

    dq = np.abs(np.diff(q, axis=0))
    print("\n" + "=" * 68)
    print(f"saved {out}")
    print(f"  steps           : {n}  ({n / steps_per_s:.1f} s @ {steps_per_s:g} Hz)")
    print(f"  rotations       : {rec['num_rotations'][-1]:.2f}" if n else "  rotations: n/a")
    drop_txt = "no" if drop_state["step"] < 0 else (
        "YES at step %d (%.1f s)" % (drop_state["step"], drop_state["step"] / steps_per_s)
    )
    print(f"  balls dropped   : {drop_txt}")
    print(f"  max |dq| / step : {dq.max():.4f} rad  ({dq.max() * steps_per_s:.2f} rad/s)")
    print(f"  q range (rad)   : [{q.min():.3f}, {q.max():.3f}]")
    print(f"  tactile ON      : {np.asarray(rec['tac']).mean() * 100:.1f}% of taxel-steps")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    main()
