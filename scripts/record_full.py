"""Record the FULL policy inference pipeline for one Baoding play rollout.

Captures, at every RL step (env 0 only):
  - encoder input (prop, tactile, and their fused concat)
  - encoder output (latent z)
  - policy input (z) / policy output (raw action, [-1,1], 13-D)
  - action scaled to joint targets, BEFORE coupling (13-D, control joints)
  - commanded joint targets AFTER coupling (16-D, all actuated joints)
  - achieved joint position / velocity (16-D)
  - applied torque (16-D)
  - raw 24-channel tactile

By default the coupled dependent joints (FF/MF/RF J1) are UNLOCKED (the
env's ``lock_coupled_dependent_at_zero`` is forced to False) so the real
J2-gated coupling law actually drives J1, instead of the training-time
hard lock to 0. Pass --lock_j1 to reproduce the locked (training) behaviour
for comparison.

Usage (from TactileBaoding_ShadowHandLite/scripts/, inside the s2r conda env):
    python record_full.py \
        --task Baoding --robot shadowlite_padtac_bt \
        --agent_cfg rl_only_pt_padtac_bt \
        --checkpoint ../checpointbiotac_final/best_agent_padtac_bt_scratch_trial27.pt \
        --num_envs 1 --num_episodes 2 --seed 42 \
        --output ../logs/full_pipeline_seed42.npz --headless
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the full encoder/policy/action pipeline.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite_padtac_bt")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=2, help="Stop after this many episodes for env 0.")
parser.add_argument("--output", type=str, default="full_pipeline_recording.npz")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--lock_j1", action="store_true", default=False,
    help="Keep the training-time hard lock (coupled J1 forced to 0). Default is UNLOCKED.",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
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

import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils import update_dict  # noqa: E402
from isaaclab_tasks.utils.hydra import register_task_to_hydra  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from common_utils import (  # noqa: E402
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    make_models,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG  # noqa: E402
from multimodal_rl.tools.writer import Writer  # noqa: E402
from roto.tasks.roto_env import scale as roto_scale  # noqa: E402


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # --- unlock the coupled J1 dependent joints (see module docstring) --------
    unlocked = not args_cli.lock_j1
    env_cfg.lock_coupled_dependent_at_zero = not unlocked
    print(f"[INFO] lock_coupled_dependent_at_zero = {env_cfg.lock_coupled_dependent_at_zero} "
          f"({'UNLOCKED — J1 follows the gated coupling law' if unlocked else 'LOCKED — J1 forced to 0'})")

    writer = Writer(agent_cfg, play=True)

    env_cfg.num_eval_envs = 0
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

    # --- introspect env internals -------------------------------------------
    raw = env.env.unwrapped
    actuated_idx = sorted(raw.actuated_dof_indices)      # list[int], len 16
    control_idx = raw.control_dof_indices                # list[int], policy order, len 13
    driver_idx = raw.coupled_driver_indices               # J2s, len 3
    dependent_idx = raw.coupled_dependent_indices         # J1s, len 3

    joint_names_all = list(raw.robot.joint_names)
    actuated_names = [joint_names_all[i] for i in actuated_idx]
    control_names = [joint_names_all[i] for i in control_idx]
    driver_names = [joint_names_all[i] for i in driver_idx]
    dependent_names = [joint_names_all[i] for i in dependent_idx]
    rl_dt = raw.cfg.sim.dt * raw.cfg.decimation

    print(f"[INFO] Actuated joints ({len(actuated_names)}): {actuated_names}")
    print(f"[INFO] Control joints  ({len(control_names)}): {control_names}")
    print(f"[INFO] Coupled driver (J2) joints: {driver_names}")
    print(f"[INFO] Coupled dependent (J1) joints: {dependent_names}")
    print(f"[INFO] RL dt = {rl_dt:.4f} s  ({1 / rl_dt:.1f} Hz)")

    control_lower = raw.robot_joint_pos_lower_limits[control_idx].clone()
    control_upper = raw.robot_joint_pos_upper_limits[control_idx].clone()

    # --- forward hooks to grab the encoder's internal fused input / latent --
    captured = {}

    def _pre_hook(module, inputs):
        captured["enc_in_fused"] = inputs[0].detach().cpu().float().numpy().copy()

    def _fwd_hook(module, inputs, output):
        captured["z"] = output.detach().cpu().float().numpy().copy()

    h1 = encoder.net.register_forward_pre_hook(_pre_hook)
    h2 = encoder.net.register_forward_hook(_fwd_hook)

    # --- data buffers (env-0 only) -------------------------------------------
    buf_keys = [
        "enc_in_prop", "enc_in_tactile", "enc_in_fused", "z",
        "action", "action_scaled_13",
        "joint_pos_cmd", "joint_pos", "joint_vel", "joint_pos_error", "torque",
        "tac",
    ]
    rec = {k: [] for k in buf_keys}
    episode_ends = []

    ep_count = 0
    timestep = 0

    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    while simulation_app.is_running() and ep_count < args_cli.num_episodes:
        with torch.inference_mode():
            # states["policy"][k] is a LazyFrames object (see FrameStack); [:] "activates"
            # it into the real concatenated (num_envs, feat*obs_stack) tensor — same as
            # Encoder._get_raw_states / IsaacLabWrapper._check_instability do internally.
            enc_in_prop = states["policy"]["prop"][:][0].detach().cpu().float().numpy().copy()
            enc_in_tactile = states["policy"]["tactile"][:][0].detach().cpu().float().numpy().copy()

            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            action_scaled_13 = roto_scale(actions, control_lower, control_upper)[0].cpu().float().numpy().copy()

            states, rewards, terminated, truncated, infos = env.step(actions)

            action_np = actions[0].detach().cpu().float().numpy().copy()
            cmd_np = raw.joint_pos_cmd[0, actuated_idx].detach().cpu().float().numpy().copy()
            pos_np = raw.joint_pos[0, actuated_idx].detach().cpu().float().numpy().copy()
            vel_np = raw.joint_vel[0, actuated_idx].detach().cpu().float().numpy().copy()
            err_np = raw.joint_pos_error[0, actuated_idx].detach().cpu().float().numpy().copy()
            tac_np = raw.tactile[0].detach().cpu().float().numpy().copy()

            applied_torque = getattr(raw.robot.data, "applied_torque", None)
            if applied_torque is not None:
                torque_np = applied_torque[0, actuated_idx].detach().cpu().float().numpy().copy()
            else:
                # Fall back to the implicit-actuator PD law: tau = Kp*(cmd-q) - Kd*qd.
                torque_np = 1.0 * (cmd_np - pos_np) - 0.1 * vel_np

            done_0 = bool(terminated[0].item()) or bool(truncated[0].item())

            if done_0 and ep_count + 1 < args_cli.num_episodes:
                states, _ = env.reset(hard=True)

        rec["enc_in_prop"].append(enc_in_prop)
        rec["enc_in_tactile"].append(enc_in_tactile)
        rec["enc_in_fused"].append(captured["enc_in_fused"][0])
        rec["z"].append(captured["z"][0])
        rec["action"].append(action_np)
        rec["action_scaled_13"].append(action_scaled_13)
        rec["joint_pos_cmd"].append(cmd_np)
        rec["joint_pos"].append(pos_np)
        rec["joint_vel"].append(vel_np)
        rec["joint_pos_error"].append(err_np)
        rec["torque"].append(torque_np)
        rec["tac"].append(tac_np)

        if done_0:
            ep_count += 1
            episode_ends.append(timestep)
            print(f"[INFO] Episode {ep_count} ended at step {timestep}")

        timestep += 1

    h1.remove()
    h2.remove()
    env.close()

    # --- save ------------------------------------------------------------------
    out_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        **{k: np.array(v, dtype=np.float32) for k, v in rec.items()},
        episode_ends=np.array(episode_ends, dtype=np.int32),
        # --- metadata: everything needed to interpret the arrays above ---
        actuated_names=np.array(actuated_names),
        control_names=np.array(control_names),
        coupled_driver_names=np.array(driver_names),
        coupled_dependent_names=np.array(dependent_names),
        actuated_dof_indices=np.array(actuated_idx, dtype=np.int32),
        control_dof_indices=np.array(control_idx, dtype=np.int32),
        coupled_driver_indices=np.array(driver_idx, dtype=np.int32),
        coupled_dependent_indices=np.array(dependent_idx, dtype=np.int32),
        joint_lower=control_lower.cpu().numpy(),
        joint_upper=control_upper.cpu().numpy(),
        joint_vel_limits=raw.robot_joint_vel_limits[control_idx].cpu().numpy(),
        Kp=np.float32(1.0),
        Kd=np.float32(0.1),
        coupling_theta=np.float32(raw.coupling_theta),
        lock_coupled_dependent_at_zero=np.bool_(env_cfg.lock_coupled_dependent_at_zero),
        rl_dt=np.float32(rl_dt),
        seed=np.int32(agent_cfg["seed"]),
        checkpoint=str(resume_path),
    )
    print(f"[INFO] Saved {timestep} steps ({ep_count} episodes) -> {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
