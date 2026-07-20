# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tune baoding ball spawn with Optuna (no RL); palm/diagonal goals stay fixed.

Objective after reset (+ brief settle):
  - ball_1 (green) close to palm goal, ball_2 (blue) close to diagonal goal
  - penalize swapped pairing (green near diagonal, blue near palm)
  - ball–ball separation ~5.5 cm (avoid landing clash on 1.5 in balls)
  - no falls

Does NOT maximize zero-action survival — the Shadow Lite hand is tilted -15 deg.

Paste the best trial ball lines into ``BaodingShadowLitePadTacCfg`` before pilot train.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Import Optuna before Isaac AppLauncher. Isaac pulls in an older system
# libstdc++, which then breaks conda sqlite3 (CXXABI_1.3.15) used by Optuna.
import optuna

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Optuna search for baoding ball spawn (fixed goals).")
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
    help="Specialised agent yaml key (merged on top of default_cfg).",
)
parser.add_argument("--num_envs", type=int, default=8, help="Parallel envs per trial (averaged in score).")
parser.add_argument("--n_trials", type=int, default=80, help="Number of Optuna trials.")
parser.add_argument(
    "--eval_steps",
    type=int,
    default=10,
    help="Physics steps after reset before scoring (settle only; not long survival).",
)
parser.add_argument(
    "--study",
    type=str,
    default="baoding_balls_only",
    help="Optuna study name (use a new name vs old goal-moving studies).",
)
parser.add_argument(
    "--output",
    type=str,
    default="tune_baoding_balls_only.json",
    help="Path to write the best trial JSON.",
)
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--video", action="store_true", default=False, help="Unused; required by make_env.")
parser.add_argument("--video_length", type=int, default=200, help="Unused; required by make_env.")
parser.add_argument("--video_dir", type=str, default=None, help="Unused; required by Writer path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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
from roto.tasks.baoding.baoding import apply_baoding_object_cfgs_from_scalars

_SQRT3 = 1.73205080757

# Fixed palm goal — diagonal is derived in apply_diagonal_targets (not optimized).
_FIXED_GOALS = {
    "palm_target_x": 0.0,
    "palm_target_y": -0.22,
    "palm_target_z": 0.41,
}

# Warm-start / search box centre (ball_1=green→palm, ball_2=blue→diagonal).
_BASE = {
    **_FIXED_GOALS,
    "ball_reset_height": 0.44,
    "ball_1_init_x": 0.009,
    "ball_1_init_y": -0.225,
    "ball_2_init_x": -0.032,
    "ball_2_init_y": -0.21,
}

# Target centre-to-centre at spawn (~1.45× diameter for 1.5 in balls).
_IDEAL_BALL_SEP = 0.055


def with_fixed_goals(params: dict) -> dict:
    """Merge fixed palm targets into a ball-only trial dict."""
    return {**_FIXED_GOALS, **params}


def get_roto_env(env):
    """Reach the underlying Isaac Lab baoding env through training wrappers."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner.unwrapped


def apply_diagonal_targets(cfg, *, diagonal_x_sign: float) -> None:
    cfg.target_offset = cfg.ball_diameter_m / _SQRT3 + 0.001
    cfg.diagonal_target_x = cfg.palm_target_x + diagonal_x_sign * cfg.target_offset
    cfg.diagonal_target_y = cfg.palm_target_y + cfg.target_offset
    cfg.diagonal_target_z = cfg.palm_target_z + cfg.target_offset


def apply_spawn_params(roto_env, params: dict, *, diagonal_x_sign: float) -> None:
    """Patch cfg + in-sim defaults so the next reset uses the trial geometry."""
    cfg = roto_env.cfg
    for key, value in params.items():
        setattr(cfg, key, value)

    apply_baoding_object_cfgs_from_scalars(cfg)
    apply_diagonal_targets(cfg, diagonal_x_sign=diagonal_x_sign)

    device = roto_env.device
    target_1 = torch.tensor(
        (cfg.palm_target_x, cfg.palm_target_y, cfg.palm_target_z),
        dtype=torch.float,
        device=device,
    )
    target_2 = torch.tensor(
        (cfg.diagonal_target_x, cfg.diagonal_target_y, cfg.diagonal_target_z),
        dtype=torch.float,
        device=device,
    )
    roto_env.goal_pos1[:] = target_1.repeat(roto_env.num_envs, 1)
    roto_env.goal_pos2[:] = target_2.repeat(roto_env.num_envs, 1)
    roto_env.update_goal_pos()
    roto_env.target1.visualize(roto_env.goal_pos1 + roto_env.scene.env_origins, roto_env.goal_rot)
    roto_env.target2.visualize(roto_env.goal_pos2 + roto_env.scene.env_origins, roto_env.goal_rot)

    z = cfg.ball_reset_height
    for ball, x_attr, y_attr in (
        (roto_env.ball_1, "ball_1_init_x", "ball_1_init_y"),
        (roto_env.ball_2, "ball_2_init_x", "ball_2_init_y"),
    ):
        state = ball.data.default_root_state.clone()
        state[:, 0] = getattr(cfg, x_attr)
        state[:, 1] = getattr(cfg, y_attr)
        state[:, 2] = z
        state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
        state[:, 7:] = 0.0
        ball.data.default_root_state[:] = state


def evaluate_spawn(env, roto_env, params: dict, *, eval_steps: int, diagonal_x_sign: float) -> tuple[float, dict]:
    """Score correct green→palm / blue→diagonal pairing after reset (+ brief settle)."""
    apply_spawn_params(roto_env, with_fixed_goals(params), diagonal_x_sign=diagonal_x_sign)

    zero = torch.zeros((roto_env.num_envs, roto_env.cfg.num_actions), device=env.device)
    env.reset(hard=True)

    for _ in range(max(0, eval_steps)):
        env.step(zero)

    roto_env._compute_intermediate_values()

    # ball_1 (green) → goal_pos1 (palm); ball_2 (blue) → goal_pos2 (diagonal).
    d11 = torch.norm(roto_env.ball_1_pos - roto_env.goal_pos1, dim=-1)
    d22 = torch.norm(roto_env.ball_2_pos - roto_env.goal_pos2, dim=-1)
    d12 = torch.norm(roto_env.ball_1_pos - roto_env.goal_pos2, dim=-1)
    d21 = torch.norm(roto_env.ball_2_pos - roto_env.goal_pos1, dim=-1)

    correct_pairing = (d11 + d22).mean().item()
    swap_pairing = (d12 + d21).mean().item()
    swap_penalty = max(0.0, swap_pairing - correct_pairing)
    mean_b1_palm = d11.mean().item()
    mean_b2_diag = d22.mean().item()
    # Penalize green sitting on blue's marker (or blue on green's).
    wrong_marker_penalty = max(0.0, 0.03 - mean_b1_palm) + max(0.0, 0.03 - mean_b2_diag)
    if d12.mean().item() < d11.mean().item():
        wrong_marker_penalty += d11.mean().item() - d12.mean().item()
    if d21.mean().item() < d22.mean().item():
        wrong_marker_penalty += d22.mean().item() - d21.mean().item()

    mean_ball_dist = roto_env.ball_dist.mean().item()
    sep_penalty = abs(mean_ball_dist - _IDEAL_BALL_SEP)
    if mean_ball_dist < roto_env.cfg.ball_diameter_m:
        sep_penalty += (roto_env.cfg.ball_diameter_m - mean_ball_dist) * 2.0

    min_z = torch.minimum(roto_env.ball_1_pos[:, 2], roto_env.ball_2_pos[:, 2]).min().item()
    fall_penalty = max(0.0, 0.32 - min_z)

    score = (
        100.0
        - 40.0 * correct_pairing
        - 30.0 * swap_penalty
        - 25.0 * wrong_marker_penalty
        - 15.0 * sep_penalty
        - 100.0 * fall_penalty
    )
    metrics = {
        "correct_pairing": correct_pairing,
        "swap_pairing": swap_pairing,
        "swap_penalty": swap_penalty,
        "mean_b1_palm": mean_b1_palm,
        "mean_b2_diag": mean_b2_diag,
        "mean_ball_dist": mean_ball_dist,
        "sep_penalty": sep_penalty,
        "wrong_marker_penalty": wrong_marker_penalty,
        "min_z": min_z,
        "fall_penalty": fall_penalty,
        "target_offset": roto_env.cfg.target_offset,
        "diagonal_target_x": roto_env.cfg.diagonal_target_x,
        "diagonal_target_y": roto_env.cfg.diagonal_target_y,
        "diagonal_target_z": roto_env.cfg.diagonal_target_z,
    }
    return score, metrics


def suggest_params(trial: optuna.Trial) -> dict:
    """Search ball spawn only; palm / diagonal goals are fixed."""
    b = _BASE
    return {
        "ball_reset_height": trial.suggest_float(
            "ball_reset_height", b["ball_reset_height"] - 0.02, b["ball_reset_height"] + 0.02
        ),
        "ball_1_init_x": trial.suggest_float("ball_1_init_x", -0.02, 0.025),
        "ball_1_init_y": trial.suggest_float("ball_1_init_y", -0.24, -0.20),
        "ball_2_init_x": trial.suggest_float("ball_2_init_x", -0.05, -0.01),
        "ball_2_init_y": trial.suggest_float("ball_2_init_y", -0.24, -0.18),
    }


def format_cfg_snippet(params: dict, cfg) -> str:
    offset = cfg.ball_diameter_m / _SQRT3 + 0.001
    diag_x = _FIXED_GOALS["palm_target_x"] - offset
    diag_y = _FIXED_GOALS["palm_target_y"] + offset
    diag_z = _FIXED_GOALS["palm_target_z"] + offset
    return (
        f"    ball_reset_height = {params['ball_reset_height']:.4f}\n"
        f"    ball_1_init_x = {params['ball_1_init_x']:.4f}  # green → palm\n"
        f"    ball_1_init_y = {params['ball_1_init_y']:.4f}\n"
        f"    ball_2_init_x = {params['ball_2_init_x']:.4f}  # blue → diagonal\n"
        f"    ball_2_init_y = {params['ball_2_init_y']:.4f}\n"
        f"    # goals fixed (not optimized):\n"
        f"    palm_target_x = {_FIXED_GOALS['palm_target_x']:.4f}\n"
        f"    palm_target_y = {_FIXED_GOALS['palm_target_y']:.4f}\n"
        f"    palm_target_z = {_FIXED_GOALS['palm_target_z']:.4f}\n"
        f"    target_offset = ball_diameter_m / 1.73205080757 + 0.001\n"
        f"    diagonal_target_x = palm_target_x - target_offset  # -> {diag_x:.4f}\n"
        f"    diagonal_target_y = palm_target_y + target_offset  # -> {diag_y:.4f}\n"
        f"    diagonal_target_z = palm_target_z + target_offset  # -> {diag_z:.4f}\n"
    )


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

    diagonal_x_sign = -1.0 if args_cli.robot in ("shadowlite", "shadowlite_padtac") else 1.0

    # Baseline score for comparison (current manual PadTac ball spawn).
    _ball_keys = (
        "ball_reset_height",
        "ball_1_init_x",
        "ball_1_init_y",
        "ball_2_init_x",
        "ball_2_init_y",
    )
    base_params = {k: _BASE[k] for k in _ball_keys}
    base_score, base_metrics = evaluate_spawn(
        env, roto_env, base_params, eval_steps=args_cli.eval_steps, diagonal_x_sign=diagonal_x_sign
    )
    print("\n===== BASELINE (fixed goals, manual ball spawn) =====")
    print(f"  score: {base_score:.3f}")
    for key, value in base_metrics.items():
        print(f"  {key}: {value}")

    storage = f"sqlite:///{os.path.abspath(args_cli.study)}.db"
    study = optuna.create_study(
        study_name=args_cli.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score, metrics = evaluate_spawn(
            env,
            roto_env,
            params,
            eval_steps=args_cli.eval_steps,
            diagonal_x_sign=diagonal_x_sign,
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return score

    study.optimize(objective, n_trials=args_cli.n_trials, show_progress_bar=True, gc_after_trial=True)

    best = study.best_trial
    print("\n===== BEST TRIAL =====")
    print(f"  score: {best.value:.3f}  (baseline was {base_score:.3f})")
    for key, value in best.params.items():
        print(f"  {key}: {value:.4f}")
    for key, value in best.user_attrs.items():
        print(f"  {key}: {value}")

    apply_spawn_params(roto_env, with_fixed_goals(best.params), diagonal_x_sign=diagonal_x_sign)
    snippet = format_cfg_snippet(best.params, roto_env.cfg)
    print("\nPaste into BaodingShadowLitePadTacCfg:\n")
    print(snippet)
    if best.value <= base_score:
        print(
            "\n[NOTE] Best Optuna ≤ baseline. Prefer keeping the manual spawn "
            "unless correct_pairing is clearly better."
        )

    result = {
        "score": best.value,
        "baseline_score": base_score,
        "baseline_metrics": base_metrics,
        "fixed_goals": _FIXED_GOALS,
        "params": best.params,
        "metrics": best.user_attrs,
        "cfg_snippet": snippet,
        "robot": args_cli.robot,
        "eval_steps": args_cli.eval_steps,
        "num_envs": args_cli.num_envs,
    }
    out_path = os.path.abspath(args_cli.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved best trial to {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
