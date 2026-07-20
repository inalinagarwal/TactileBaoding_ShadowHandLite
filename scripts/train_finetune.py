# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fine-tune an existing RoTO checkpoint on a (possibly different) hand/env.

Separate from ``train.py`` so scratch training stays untouched. Typical use:
warm-start PadTac best weights into ``shadowlite_padtac_bt`` without writing
back into the PadTac log folder.

Example::

    cd ~/roto_2/scripts
    PYTHONPATH=/home/nalin/roto_2:$PYTHONPATH python train_finetune.py \\
      --task Baoding --robot shadowlite_padtac_bt \\
      --agent_cfg rl_only_pt_padtac_bt_ft \\
      --checkpoint logs/shadowlite_baoding/rl_only_pt_padtac/2026-07-11_16-09-44/checkpoints/best_agent.pt \\
      --num_envs 4096 --headless --seed 42 --device cuda:1

20 Hz RL (matches deploy CONTROL_HZ; does not change global physics.py)::

    ... --agent_cfg rl_only_pt_padtac_bt_ft_20hz ...
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Fine-tune a RoTO checkpoint (warm-start). Does not modify train.py."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save recorded videos.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--robot",
    type=str,
    default=None,
    help="Robot: Bounce/Baoding → shadow|shadowlite|orca|allegro|shadowlite_padtac|shadowlite_padtac_bt.",
)
parser.add_argument("--agent_cfg", type=str, default=None, help="Name of the agent configuration.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to source checkpoint to warm-start from (read-only; never overwritten).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--renderer", type=str, default="PathTracing", choices=["RayTracedLighting", "PathTracing"], help="Renderer to use."
)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1, help="Number of samples per pixel per frame.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_aux,
    make_env,
    make_memory,
    make_models,
    make_trainer,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer

# Hard-block saving into these experiment folders (source PadTac / link keepers).
_PROTECTED_EXPERIMENT_NAMES = frozenset(
    {
        "rl_only_pt_padtac",
        "rl_only_pt",
    }
)

# Optional RL control rate overrides (agent_cfg name -> Hz). Only applied when listed;
# default padtac_bt / _ft stay at global DECIMATION=4 (~60 Hz). physics.py untouched.
_RL_HZ_BY_AGENT_CFG = {
    "rl_only_pt_padtac_bt_ft_20hz": 20,
}


def _maybe_apply_rl_hz(env_cfg, agent_cfg_name: str) -> None:
    """Set env_cfg.decimation so 1/(sim.dt * decimation) matches the target Hz."""
    hz = _RL_HZ_BY_AGENT_CFG.get(agent_cfg_name)
    if hz is None:
        return
    dt = float(env_cfg.sim.dt)
    dec = int(round(1.0 / (dt * float(hz))))
    if dec < 1:
        raise ValueError(f"Bad RL Hz={hz} with dt={dt}: computed decimation={dec}")
    env_cfg.decimation = dec
    rl_hz = 1.0 / (dt * dec)
    print(
        f"[finetune] agent_cfg={agent_cfg_name!r}: target RL {hz} Hz -> "
        f"decimation={dec} (actual {rl_hz:.4g} Hz; sim.dt={dt})"
    )


def _assert_safe_paths(checkpoint: str, agent_cfg: dict) -> str:
    """Resolve checkpoint path and refuse configs that would overwrite protected runs."""
    ckpt = os.path.abspath(checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    exp_name = agent_cfg["experiment"]["experiment_name"]
    if exp_name in _PROTECTED_EXPERIMENT_NAMES:
        raise RuntimeError(
            f"Refuse to fine-tune with experiment_name={exp_name!r}. "
            f"That would write new checkpoints under the protected PadTac/link log tree. "
            f"Use agent_cfg rl_only_pt_padtac_bt_ft (experiment_name=rl_only_pt_padtac_bt_ft)."
        )

    log_root = os.path.abspath(
        os.path.join(
            agent_cfg.get("log_path") or os.getcwd(),
            "logs",
            agent_cfg["experiment"]["directory"],
            exp_name,
        )
    )
    ckpt_dir = os.path.abspath(os.path.dirname(ckpt))
    # Never save into the same directory tree as the source checkpoint file.
    if ckpt_dir == log_root or ckpt_dir.startswith(log_root + os.sep):
        raise RuntimeError(
            f"Checkpoint lives under this run's log root ({log_root}). "
            "Pick a different experiment_name so saves cannot overwrite the source."
        )
    if log_root.rstrip("/").endswith("/rl_only_pt_padtac"):
        raise RuntimeError(f"Log root resolves to protected PadTac folder: {log_root}")

    print("[finetune] source checkpoint (READ ONLY):", ckpt)
    print("[finetune] experiment_name (WRITE):", exp_name)
    print("[finetune] new logs will go under:", log_root, "/<timestamp>/")
    return ckpt


def fine_tune_one_seed(args_cli, env, agent_cfg, env_cfg, writer, seed, checkpoint: str) -> None:
    """Build PPO, load warm-start weights, then train (saves only under writer.log_dir)."""
    dtype = torch.float32
    agent_cfg["seed"] = seed
    set_seed(agent_cfg["seed"])

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)
    rl_memory = make_memory(
        env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=env.num_train_envs
    )
    ssl_task = make_aux(env, rl_memory, encoder, value, value_preprocessor, env_cfg, agent_cfg, writer)

    ppo_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_agent_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder,
        policy,
        value,
        value_preprocessor,
        memory=rl_memory,
        cfg=ppo_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        writer=writer,
        ssl_task=ssl_task,
        dtype=dtype,
        debug=agent_cfg["experiment"]["debug"],
    )

    print(f"[finetune] loading warm-start weights from {checkpoint}")
    agent.load(checkpoint)
    print("[finetune] load done; starting training (new best_agent.pt only under this run's folder)")

    trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)
    trainer.train()
    print("Fine-tune complete!")


def main() -> None:
    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir

    ckpt = _assert_safe_paths(args_cli.checkpoint, agent_cfg)

    writer = Writer(agent_cfg)
    print("[finetune] this run log_dir:", writer.log_dir)
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    _maybe_apply_rl_hz(env_cfg, args_cli.agent_cfg)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)
    fine_tune_one_seed(
        args_cli, env, agent_cfg=agent_cfg, env_cfg=env_cfg, writer=writer, seed=seed, checkpoint=ckpt
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("ERROR DURING FINE-TUNE", err)
        raise
    finally:
        print("CLOSING")
        simulation_app.close()
