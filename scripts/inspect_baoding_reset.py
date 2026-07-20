# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize baoding reset geometry without training or a policy checkpoint.

Edit ``BaodingShadowLitePadTacCfg`` (or other baoding cfg), run this script,
check printed ball/goal distances and optional headless video, then pilot-train.

Example::

    cd ~/roto/scripts
    python inspect_baoding_reset.py --robot shadowlite_padtac --headless --video \\
        --device cuda:1 --settle_steps 10
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect baoding spawn and goal markers (no checkpoint).")
parser.add_argument("--task", type=str, default="Baoding", help="Task name.")
parser.add_argument(
    "--robot",
    type=str,
    default="shadowlite_padtac",
    help="Robot variant (shadowlite, shadowlite_padtac, ...).",
)
parser.add_argument(
    "--agent_cfg",
    type=str,
    default="rl_only_pt_padtac",
    help="Agent yaml key (merged on default_cfg for env obs settings).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (use 1 for video).")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=10,
    help="Zero-action physics steps after reset before reporting (0 = report at reset only).",
)
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record a short headless video.")
parser.add_argument("--video_length", type=int, default=60, help="Video length in steps.")
parser.add_argument("--video_dir", type=str, default=None, help="Unused; required by Writer.")
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer when --video is set.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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


def get_roto_env(env):
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner.unwrapped


def _fmt_vec(v) -> str:
    return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"


def print_geometry_report(roto_env, *, label: str) -> None:
    roto_env._compute_intermediate_values()
    cfg = roto_env.cfg
    i = 0

    b1 = roto_env.ball_1_pos[i].detach().cpu()
    b2 = roto_env.ball_2_pos[i].detach().cpu()
    g1 = roto_env.ball_1_goal_pos[i].detach().cpu()
    g2 = roto_env.ball_2_goal_pos[i].detach().cpu()
    m1 = roto_env.goal_pos1[i].detach().cpu()
    m2 = roto_env.goal_pos2[i].detach().cpu()

    d11 = torch.norm(b1 - g1).item()
    d22 = torch.norm(b2 - g2).item()
    d12 = torch.norm(b1 - g2).item()
    d21 = torch.norm(b2 - g1).item()
    ball_sep = roto_env.ball_dist[i].item()

    print(f"\n===== {label} (env 0) =====")
    print(f"  ball_1_pos:      {_fmt_vec(b1)}")
    print(f"  ball_2_pos:      {_fmt_vec(b2)}")
    print(f"  ball_1 -> goal:  {_fmt_vec(g1)}  dist={d11:.4f} m")
    print(f"  ball_2 -> goal:  {_fmt_vec(g2)}  dist={d22:.4f} m")
    print(f"  (swap check) b1->g2={d12:.4f}  b2->g1={d21:.4f}  min pairing={min(d11+d22, d12+d21):.4f}")
    print(f"  marker palm:     {_fmt_vec(m1)}  (cfg palm target)")
    print(f"  marker diagonal: {_fmt_vec(m2)}  (cfg diagonal target)")
    print(f"  ball-ball sep:   {ball_sep:.4f} m  (diameter {cfg.ball_diameter_m:.4f} m)")
    print(f"  success_tol:     {cfg.success_tolerance} m")
    print(f"  min ball z:      {min(b1[2].item(), b2[2].item()):.4f} m  (fall if < 0.3)")


def print_cfg_summary(roto_env) -> None:
    cfg = roto_env.cfg
    print("\n===== CONFIG (from env cfg) =====")
    print(f"  ball_reset_height = {cfg.ball_reset_height}")
    print(f"  ball_1_init       = ({cfg.ball_1_init_x}, {cfg.ball_1_init_y})")
    print(f"  ball_2_init       = ({cfg.ball_2_init_x}, {cfg.ball_2_init_y})")
    print(f"  palm_target       = ({cfg.palm_target_x}, {cfg.palm_target_y}, {cfg.palm_target_z})")
    print(
        f"  diagonal_target   = ({cfg.diagonal_target_x}, {cfg.diagonal_target_y}, {cfg.diagonal_target_z})"
    )
    print(f"  target_offset     = {cfg.target_offset}")
    print(f"  ball_diameter     = {cfg.ball_diameter_inches} in ({cfg.ball_diameter_m:.4f} m)")


def main() -> None:
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    agent_cfg["seed"] = args_cli.seed
    set_seed(args_cli.seed)
    agent_cfg["log_path"] = LOG_PATH

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0
    env_cfg.reset_joint_pos_noise = 0.0

    writer = Writer(agent_cfg, play=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    roto_env = get_roto_env(env)

    print_cfg_summary(roto_env)

    zero = torch.zeros((roto_env.num_envs, roto_env.cfg.num_actions), device=env.device)
    env.reset(hard=True)
    print_geometry_report(roto_env, label="AT RESET")

    steps_to_run = args_cli.video_length if args_cli.video else args_cli.settle_steps
    for step in range(steps_to_run):
        env.step(zero)
        if not args_cli.video and step + 1 == args_cli.settle_steps:
            print_geometry_report(roto_env, label=f"AFTER {args_cli.settle_steps} ZERO-ACTION STEPS")

    if args_cli.video:
        print_geometry_report(roto_env, label=f"AFTER {args_cli.video_length} ZERO-ACTION STEPS (video end)")
        print("\n[INFO] Video saved under ./videos/ (typically rl-video-step-0.mp4)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
