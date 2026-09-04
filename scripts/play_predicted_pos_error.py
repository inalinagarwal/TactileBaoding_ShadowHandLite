# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop validation: run the Baoding policy with the LEARNED pos-error model
standing in for the measured joint_pos_error in the encoder's observation.

At every step:
  1. Read the current 4-frame prop stack -> pos4/vel4/act4 (same windowing as
     collect_pos_error_data.py).
  2. z = encoder(states); a_t = policy.act(z, deterministic=True)  (unchanged).
  3. Predict pos_error_{t+1} from (pos4, vel4, act4, a_t) with the trained MLP.
  4. states, ... = env.step(a_t)   -- physics advances for real, using a_t exactly
     as it would in normal play.
  5. Overwrite the newest stacked frame's pos_error slot ([26:39] of the 52-D prop
     frame) IN PLACE with the model's PREDICTION instead of the true measured error,
     for every env that did not just auto-reset this step. Because this happens every
     step, after the 4-step warmup the entire stacked window the encoder sees is
     built from predicted (not measured) pos_error -- true closed-loop substitution.

Reports, per episode: return, ball_dist, num_rotations (successful ball-goal swaps),
and whether the episode ended in a physics failure (ball dropped) vs a timeout, so
"does it still do baoding" is a checkable number, not just a video.

Usage (from TactileBaoding_ShadowHandLite/scripts/, inside the s2r conda env):
    python play_predicted_pos_error.py \
        --task Baoding --robot shadowlite_padtac_bt \
        --agent_cfg rl_only_pt_padtac_bt \
        --checkpoint best_agent_padtac_bt_scratch_trial15.pt \
        --pos_error_model ../logs/pos_error_model/pos_error_mlp.pt \
        --num_envs 64 --num_episodes 1 --seed 100 --headless
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play the policy with the learned pos-error model closing the loop.")
parser.add_argument("--task", type=str, default="Baoding")
parser.add_argument("--robot", type=str, default="shadowlite_padtac_bt")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt")
parser.add_argument("--pos_error_model", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=1, help="Stop after this many episodes for env 0.")
parser.add_argument("--seed", type=int, default=100, help="A seed NOT used during data collection.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
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
from train_pos_error_model import PosErrorMLP  # noqa: E402

OBS_STACK = 4
PROP_FRAME_DIM = 52


def main():
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)
    dtype = torch.float32

    agent_cfg["seed"] = args_cli.seed
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    assert env_cfg.obs_stack == OBS_STACK

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
    print(f"[INFO] Loaded policy checkpoint: {resume_path}")

    # --- load the learned pos-error model ------------------------------------
    pe_path = os.path.abspath(args_cli.pos_error_model)
    pe_ckpt = torch.load(pe_path, map_location=env.device, weights_only=False)
    pe_model = PosErrorMLP(pe_ckpt["input_dim"], tuple(pe_ckpt["hiddens"]), pe_ckpt["target_dim"]).to(env.device)
    pe_model.load_state_dict(pe_ckpt["state_dict"])
    pe_model.eval()
    x_mean = torch.as_tensor(pe_ckpt["x_mean"], device=env.device, dtype=dtype)
    x_std = torch.as_tensor(pe_ckpt["x_std"], device=env.device, dtype=dtype)
    y_mean = torch.as_tensor(pe_ckpt["y_mean"], device=env.device, dtype=dtype)
    y_std = torch.as_tensor(pe_ckpt["y_std"], device=env.device, dtype=dtype)
    print(f"[INFO] Loaded pos-error model: {pe_path}  "
          f"(val_rmse_overall={pe_ckpt.get('val_rmse_overall', float('nan')):.5f} rad)")

    raw = env.env.unwrapped
    control_idx = raw.control_dof_indices
    num_envs = env.num_envs

    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    pred_err_mae_sum = torch.zeros(num_envs, device=env.device)
    step_count = torch.zeros(num_envs, device=env.device)
    num_steps = int(raw.max_episode_length) * args_cli.num_episodes
    if args_cli.video:
        num_steps = min(num_steps, args_cli.video_length)

    for t in range(num_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            prop = states["policy"]["prop"][:].view(num_envs, OBS_STACK, PROP_FRAME_DIM)
            pos4 = prop[:, :, 0:13].reshape(num_envs, -1)
            vel4 = prop[:, :, 13:26].reshape(num_envs, -1)
            act4 = prop[:, :, 39:52].reshape(num_envs, -1)

            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)

            x = torch.cat([pos4, vel4, act4, actions], dim=-1)
            pred_err = pe_model((x - x_mean) / x_std) * y_std + y_mean  # raw radians (num_envs, 13)

            states, rewards, terminated, truncated, infos = env.step(actions)

            true_err = raw.joint_pos_error[:, control_idx]
            pred_err_mae_sum += (pred_err - true_err).abs().mean(dim=-1)
            step_count += 1

            done = torch.logical_or(terminated, truncated).squeeze(-1)
            not_done = ~done

            # Substitute the PREDICTED pos_error into the just-appended newest stacked
            # frame (in place -- also mutates the tensor already referenced by `states`).
            newest = env.frames["prop"][-1]
            newest[not_done, 26:39] = pred_err[not_done]

            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                num_rot = infos.get("counters", {}).get("num_rotations", None)
                ball_dist = infos.get("log", {}).get("ball_dist", None)
                for i in done_ids.tolist():
                    failed = bool(terminated[i].item())
                    rot = num_rot[i].item() if num_rot is not None else float("nan")
                    bd = ball_dist[i].item() if ball_dist is not None else float("nan")
                    mae = (pred_err_mae_sum[i] / step_count[i]).item()
                    print(f"[EP END step {t}] env={i:4d}  {'FAILED (ball dropped)' if failed else 'timeout (ok)':22s}  "
                          f"num_rotations={rot:.0f}  ball_dist={bd:.4f}  mean|pred-true|={mae:.5f} rad")
                pred_err_mae_sum[done_ids] = 0.0
                step_count[done_ids] = 0.0

        if (t + 1) % 100 == 0:
            print(f"[INFO] step {t + 1}/{num_steps}")

    env.close()
    print("[INFO] Closed-loop predicted-pos-error rollout complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
