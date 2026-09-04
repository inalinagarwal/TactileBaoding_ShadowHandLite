# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect (window -> next-step joint_pos_error) training data from a Baoding policy rollout.

Runs the deterministic policy for one nominal episode duration across ALL parallel
environments simultaneously (default 8190 envs), and for every env at every step records:

  - 4 stacked frames of normalised_joint_pos   (52-D: "last 3 pos + current pos")
  - 4 stacked frames of normalised_joint_vel   (52-D: "last 4 vel")
  - 4 stacked frames of the previous action    (52-D: "prev actions")
  - the action the policy just emitted, a_t    (13-D: "current action")
  - target: joint_pos_error read AFTER env.step(a_t), i.e. the error the encoder will
    see at t+1 as a consequence of a_t (13-D)

Also records a `done` flag and `steps_since_reset` (the env's episode_length_buf at the
time the window/action were produced) so invalid rows -- FrameStack warmup after a reset,
and steps whose target belongs to a freshly-reset episode rather than being caused by
a_t -- can be filtered out offline by the trainer.

Usage (from TactileBaoding_ShadowHandLite/scripts/, inside the icra conda env):
    python collect_pos_error_data.py \
        --task Baoding --robot shadowlite_padtac_bt \
        --agent_cfg rl_only_pt_padtac_bt \
        --checkpoint best_agent_padtac_bt_scratch_trial15.pt \
        --num_envs 8190 --seed 42 --headless \
        --output ../logs/pos_error_data/seed42.npz
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect pos-error prediction data from a Baoding policy rollout.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite_padtac_bt")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt")
parser.add_argument("--num_envs", type=int, default=8190)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None, help="Output .npz path (default: ../logs/pos_error_data/seed{seed}.npz)")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False, help="Unused; make_env() checks this flag.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils import update_dict  # noqa: E402

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

OBS_STACK = 4
PROP_FRAME_DIM = 52  # pos(13) + vel(13) + pos_error(13) + action(13)


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = None

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    assert env_cfg.obs_stack == OBS_STACK, f"expected obs_stack={OBS_STACK}, got {env_cfg.obs_stack}"

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

    raw = env.env.unwrapped
    control_idx = raw.control_dof_indices  # policy order, len 13
    joint_names_all = list(raw.robot.joint_names)
    control_names = [joint_names_all[i] for i in control_idx]

    num_envs = env.num_envs
    num_steps = int(raw.max_episode_length)
    rl_dt = raw.cfg.sim.dt * raw.cfg.decimation

    print(f"[INFO] num_envs={num_envs}  num_steps={num_steps}  rl_dt={rl_dt:.5f}s ({1 / rl_dt:.1f} Hz)")
    print(f"[INFO] Control joints ({len(control_names)}): {control_names}")

    # --- preallocated CPU buffers -------------------------------------------
    pos4 = np.empty((num_steps, num_envs, PROP_FRAME_DIM), dtype=np.float32)
    vel4 = np.empty((num_steps, num_envs, PROP_FRAME_DIM), dtype=np.float32)
    act4 = np.empty((num_steps, num_envs, PROP_FRAME_DIM), dtype=np.float32)
    cur_action = np.empty((num_steps, num_envs, len(control_idx)), dtype=np.float32)
    target = np.empty((num_steps, num_envs, len(control_idx)), dtype=np.float32)
    done_buf = np.empty((num_steps, num_envs), dtype=np.bool_)
    steps_since_reset_buf = np.empty((num_steps, num_envs), dtype=np.int16)

    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    for t in range(num_steps):
        with torch.inference_mode():
            # Age of the CURRENT window/state (used to build a_t) -- captured BEFORE
            # stepping, since env.step() advances/resets episode_length_buf internally.
            steps_since_reset = raw.episode_length_buf.detach().cpu().numpy().copy()

            prop = states["policy"]["prop"][:]  # (num_envs, 208), LazyFrames -> tensor
            prop = prop.view(num_envs, OBS_STACK, PROP_FRAME_DIM)
            pos4_t = prop[:, :, 0:13].reshape(num_envs, -1)
            vel4_t = prop[:, :, 13:26].reshape(num_envs, -1)
            act4_t = prop[:, :, 39:52].reshape(num_envs, -1)

            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            states, rewards, terminated, truncated, infos = env.step(actions)

            done = torch.logical_or(terminated, truncated).squeeze(-1)
            err_np = raw.joint_pos_error[:, control_idx].detach().cpu().float().numpy()

            pos4[t] = pos4_t.detach().cpu().float().numpy()
            vel4[t] = vel4_t.detach().cpu().float().numpy()
            act4[t] = act4_t.detach().cpu().float().numpy()
            cur_action[t] = actions.detach().cpu().float().numpy()
            target[t] = err_np
            done_buf[t] = done.detach().cpu().numpy()
            steps_since_reset_buf[t] = steps_since_reset.astype(np.int16)

        if (t + 1) % 50 == 0 or t == num_steps - 1:
            print(f"[INFO] step {t + 1}/{num_steps}")

    env.close()

    # --- save -----------------------------------------------------------------
    out_path = args_cli.output or os.path.abspath(
        os.path.join("..", "logs", "pos_error_data", f"seed{agent_cfg['seed']}.npz")
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        pos4=pos4, vel4=vel4, act4=act4,
        cur_action=cur_action, target=target,
        done=done_buf, steps_since_reset=steps_since_reset_buf,
        control_names=np.array(control_names),
        control_dof_indices=np.array(control_idx, dtype=np.int32),
        joint_lower=raw.robot_joint_pos_lower_limits[control_idx].cpu().numpy(),
        joint_upper=raw.robot_joint_pos_upper_limits[control_idx].cpu().numpy(),
        joint_vel_limits=raw.robot_joint_vel_limits[control_idx].cpu().numpy(),
        obs_stack=np.int32(OBS_STACK),
        rl_dt=np.float32(rl_dt),
        seed=np.int32(agent_cfg["seed"]),
        checkpoint=str(resume_path),
        input_layout=(
            "X = concat(pos4[52], vel4[52], act4[52], cur_action[13]) -> 169-D; "
            "pos4/vel4/act4 frame order is oldest(t-3) -> newest(t); "
            "cur_action = a_t (about to be applied); "
            "target = joint_pos_error AFTER env.step(a_t), i.e. error seen by encoder at t+1; "
            "drop rows where done[t] or steps_since_reset[t] < obs_stack"
        ),
    )
    print(f"[INFO] Saved {num_steps} steps x {num_envs} envs -> {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
