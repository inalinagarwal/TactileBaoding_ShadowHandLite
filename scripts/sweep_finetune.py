# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warm-start Optuna sweep for padtac_bt (loads PadTac checkpoint every trial).

Sweeps the **full** PPO hyperparam set (same knobs as ``sweep.py``):
rollouts, mini_batches, learning_epochs, learning_rate, entropy_loss_scale,
value_loss_scale, ratio_clip (+ SSL knobs if present). Entropy upper bound is
0.12 (scratch sweep capped at 0.01) so fine-tune-scale entropy is searchable.

Does NOT modify ``sweep.py`` / ``train.py``. Checkpoints are written only under
study-prefixed experiment names (``Baoding_rl_only_pt_padtac_bt_ft_sweep_*``).

Protected (never written to):
  - ``rl_only_pt_padtac`` (PadTac best)
  - ``rl_only_pt_padtac_bt_ft`` (tonight's deploy fine-tune run)
  - ``rl_only_pt`` (link baseline)

Example::

    cd ~/roto_2/scripts
    PYTHONPATH=/home/nalin/roto_2:$PYTHONPATH python sweep_finetune.py \\
      --task Baoding --robot shadowlite_padtac_bt \\
      --agent_cfg rl_only_pt_padtac_bt_ft_sweep \\
      --checkpoint logs/shadowlite_baoding/rl_only_pt_padtac/2026-07-11_16-09-44/checkpoints/best_agent.pt \\
      --study jul14_ft --num_envs 4096 --headless --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import optuna
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Warm-start Optuna sweep (PadTac ckpt -> padtac_bt). Separate from sweep.py."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--video_interval", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument(
    "--robot",
    type=str,
    default=None,
    help="Use shadowlite_padtac_bt for BioTac fine-tune sweeps.",
)
parser.add_argument("--agent_cfg", type=str, default="rl_only_pt_padtac_bt_ft_sweep")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="PadTac (or other) checkpoint to warm-start from — READ ONLY.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--study", type=str, default="default", help="Optuna study name (also folds into log dir).")
parser.add_argument("--n_trials", type=int, default=40, help="Total Optuna trial budget.")
parser.add_argument(
    "--rerun-trial",
    type=int,
    default=None,
    metavar="N",
    help="Load trial N from study and multi-seed train (still warm-starts from --checkpoint).",
)
parser.add_argument("--rerun-seeds", type=int, nargs="+", default=None)

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

# Experiment folder names that must never be written by this script.
_PROTECTED_EXPERIMENT_NAMES = frozenset(
    {
        "rl_only_pt",
        "rl_only_pt_padtac",
        "rl_only_pt_padtac_bt_ft",  # tonight's deploy FT run
    }
)


def _resolve_checkpoint(path: str) -> str:
    ckpt = os.path.abspath(path)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _assert_experiment_safe(experiment_name: str) -> None:
    if experiment_name in _PROTECTED_EXPERIMENT_NAMES:
        raise RuntimeError(
            f"Refuse experiment_name={experiment_name!r} — protected (PadTac / deploy FT / link). "
            f"Use agent_cfg rl_only_pt_padtac_bt_ft_sweep so logs go under a Baoding_*_sweep_* name."
        )
    # Also block if the resolved name equals a protected leaf somehow.
    for forbidden in _PROTECTED_EXPERIMENT_NAMES:
        if experiment_name == forbidden or experiment_name.endswith("/" + forbidden):
            raise RuntimeError(f"Refuse protected experiment path component: {experiment_name!r}")


def _assert_log_root_safe(agent_cfg: dict) -> str:
    exp_name = agent_cfg["experiment"]["experiment_name"]
    _assert_experiment_safe(exp_name)
    log_root = os.path.abspath(
        os.path.join(
            agent_cfg.get("log_path") or os.getcwd(),
            "logs",
            agent_cfg["experiment"]["directory"],
            exp_name,
        )
    )
    for forbidden in _PROTECTED_EXPERIMENT_NAMES:
        protected_root = os.path.abspath(
            os.path.join(
                agent_cfg.get("log_path") or os.getcwd(),
                "logs",
                agent_cfg["experiment"]["directory"],
                forbidden,
            )
        )
        if log_root == protected_root or log_root.startswith(protected_root + os.sep):
            raise RuntimeError(
                f"Log root {log_root} would write under protected tree {protected_root}"
            )
    return log_root


def apply_optuna_trial_params(agent_cfg: dict, trial: optuna.trial.FrozenTrial) -> None:
    p = trial.params
    agent_cfg["agent"]["rollouts"] = 2 ** p["rollouts_pow"]
    agent_cfg["agent"]["mini_batches"] = p["mini_batches"]
    agent_cfg["agent"]["learning_epochs"] = p["learning_epochs"]
    agent_cfg["agent"]["learning_rate"] = p["learning_rate"]
    agent_cfg["agent"]["entropy_loss_scale"] = p["entropy_loss_scale"]
    agent_cfg["agent"]["value_loss_scale"] = p["value_loss_scale"]
    agent_cfg["agent"]["ratio_clip"] = p["ratio_clip"]
    if "ssl_task" in agent_cfg:
        agent_cfg["ssl_task"]["learning_rate"] = p["learning_rate_aux"]
        agent_cfg["ssl_task"]["loss_weight"] = p["loss_weight_aux"]
        if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            agent_cfg["ssl_task"]["seq_length"] = p["seq_length"]


def warm_start_train(env, env_cfg, agent_cfg, writer, seed: int, checkpoint: str, trial=None):
    """Build PPO, load warm-start weights, train. Saves only under writer.log_dir."""
    dtype = torch.float32
    agent_cfg["seed"] = seed
    set_seed(seed)

    writer.get_new_log_path()
    print("[sweep_ft] log_dir (WRITE):", writer.log_dir)
    print("[sweep_ft] warm-start (READ):", checkpoint)

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, dtype)
    num_training_envs = env_cfg.scene.num_envs - agent_cfg["trainer"]["num_eval_envs"]
    rl_memory = make_memory(env, env_cfg, size=agent_cfg["agent"]["rollouts"], num_envs=num_training_envs)
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
    agent.load(checkpoint)

    trainer = make_trainer(env, agent, agent_cfg, ssl_task, writer)
    if trial is None:
        return trainer.train()
    return trainer.train(trial=trial)


class WarmStartOptimisationRunner:
    def __init__(self, study_name, n_startup_trials, n_warmup_steps, interval_steps, storage, checkpoint):
        self.checkpoint = checkpoint
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, multivariate=True)
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps
        )
        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    def run(self, n_trials=50):
        self.study.optimize(
            lambda trial: self.objective(trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        print(f"Number of finished trials: {len(self.study.trials)}")
        trial = self.study.best_trial
        print("Best trial value:", trial.value)
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        return trial

    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg) -> float:
        print(f"Starting warm-start trial: {trial.number}")
        TRAIN_SEEDS = [0, 1, 2, 3, 4]
        seed = int(np.random.choice(TRAIN_SEEDS))

        if "ssl_task" in agent_cfg and agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            max_rollouts_pow = 5
        else:
            max_rollouts_pow = 6

        # Full PPO hyperparam set (same knobs as sweep.py) + wider entropy upper
        # bound so fine-tune-friendly values (e.g. 0.05–0.08) are in-range.
        rollouts = 2 ** trial.suggest_int("rollouts_pow", 4, max_rollouts_pow)
        mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32])
        learning_epochs = trial.suggest_int("learning_epochs", low=4, high=10, step=1)
        learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=5e-4, log=True)
        entropy_loss_scale = trial.suggest_float("entropy_loss_scale", 1e-4, 0.12, log=True)
        value_loss_scale = trial.suggest_float("value_loss_scale", low=0.1, high=1.0, log=True)
        ratio_clip = trial.suggest_float("ratio_clip", low=0.1, high=0.2)

        if "ssl_task" in agent_cfg and agent_cfg["ssl_task"]["type"] == "forward_dynamics":
            mini_batches = min(mini_batches, 8)

        agent_cfg["agent"]["rollouts"] = rollouts
        agent_cfg["agent"]["mini_batches"] = mini_batches
        agent_cfg["agent"]["learning_epochs"] = learning_epochs
        agent_cfg["agent"]["learning_rate"] = learning_rate
        agent_cfg["agent"]["entropy_loss_scale"] = entropy_loss_scale
        agent_cfg["agent"]["value_loss_scale"] = value_loss_scale
        agent_cfg["agent"]["ratio_clip"] = ratio_clip

        if "ssl_task" in agent_cfg:
            agent_cfg["ssl_task"]["learning_rate"] = trial.suggest_float(
                "learning_rate_aux", low=1e-5, high=5e-4, log=True
            )
            agent_cfg["ssl_task"]["loss_weight"] = trial.suggest_float(
                "loss_weight_aux", low=1e-3, high=1, log=True
            )
            if agent_cfg["ssl_task"]["type"] == "forward_dynamics":
                agent_cfg["ssl_task"]["seq_length"] = trial.suggest_int("seq_length", low=2, high=10, step=1)

        writer.close_wandb()
        writer.setup_wandb(name=trial.number)

        should_prune = False
        best_return = float("nan")
        try:
            best_return, should_prune = warm_start_train(
                env, env_cfg, agent_cfg, writer, seed, self.checkpoint, trial=trial
            )
        except AssertionError as e:
            print("Trial AssertionError (often NaN):", e)

        if should_prune:
            raise optuna.TrialPruned()
        return best_return


if __name__ == "__main__":
    print("Running WARM-START Optuna sweep (PadTac ckpt -> padtac_bt)")

    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)
    if args_cli.task in ("Bounce", "Baoding"):
        env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
        specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg)
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args_cli.gym_env_id, "default_cfg")
        specialised_cfg = load_cfg_from_registry(args_cli.gym_env_id, args_cli.agent_cfg)
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    checkpoint = _resolve_checkpoint(args_cli.checkpoint)
    print("[sweep_ft] source checkpoint (READ ONLY):", checkpoint)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    args_cli.video = agent_cfg["experiment"]["upload_videos"]
    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    max_sweep_timesteps_M = agent_cfg["sweeper"]["max_sweep_timesteps_M"]
    max_training_timesteps_M = agent_cfg["trainer"]["max_global_timesteps_M"]
    storage = agent_cfg["sweeper"]["storage"]

    if args_cli.rerun_trial is not None:
        study = optuna.load_study(study_name=args_cli.study, storage=storage)
        trial = next((t for t in study.trials if t.number == args_cli.rerun_trial), None)
        if trial is None:
            raise ValueError(f"No trial {args_cli.rerun_trial} in study {args_cli.study!r}")
        apply_optuna_trial_params(agent_cfg, trial)
        agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
        suffix = f"_trial_{args_cli.rerun_trial}"
        agent_cfg["experiment"]["experiment_name"] = (
            args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study + suffix
        )
        agent_cfg["experiment"]["wandb_kwargs"]["group"] = agent_cfg["experiment"]["experiment_name"]
        log_root = _assert_log_root_safe(agent_cfg)
        print("[sweep_ft] rerun log root:", log_root)

        seeds = args_cli.rerun_seeds if args_cli.rerun_seeds is not None else [5, 6, 7, 8, 9, 10]
        writer = Writer(agent_cfg, delay_wandb_startup=True)
        env = make_env(agent_cfg, env_cfg, writer, args_cli)
        for seed in seeds:
            writer.setup_wandb(name=f"trial_{args_cli.rerun_trial}_seed_{seed}")
            warm_start_train(env, env_cfg, agent_cfg, writer, seed, checkpoint, trial=None)
            writer.close_wandb()
        env.close()
        simulation_app.close()
        sys.exit(0)

    # Optuna sweep path
    agent_cfg["experiment"]["experiment_name"] = (
        args_cli.task + "_" + args_cli.agent_cfg + "_" + args_cli.study
    )
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = agent_cfg["experiment"]["experiment_name"]
    log_root = _assert_log_root_safe(agent_cfg)
    print("[sweep_ft] sweep log root:", log_root)
    print("[sweep_ft] Optuna storage:", storage)

    agent_cfg["trainer"]["max_global_timesteps_M"] = max_sweep_timesteps_M
    n_warmup_steps = agent_cfg["sweeper"]["warmup_timesteps_M"] * 1e6
    study_name = args_cli.study
    total_trials = args_cli.n_trials
    n_startup_trials = min(8, max(1, total_trials // 5))
    interval_steps = 1

    writer = Writer(agent_cfg, delay_wandb_startup=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    runner = WarmStartOptimisationRunner(
        study_name, n_startup_trials, n_warmup_steps, interval_steps, storage, checkpoint
    )
    trials_already_done = len(runner.study.trials)
    remaining_trials = max(0, total_trials - trials_already_done)
    if remaining_trials <= 0:
        print(f"Study already reached {total_trials} trials.")
        env.close()
        simulation_app.close()
        sys.exit(0)

    print("Running remaining warm-start trials:", remaining_trials)
    best_trial = runner.run(remaining_trials)
    writer.close_wandb()

    apply_optuna_trial_params(agent_cfg, best_trial)
    agent_cfg["experiment"]["experiment_name"] = (
        args_cli.task + "_" + args_cli.agent_cfg + "_seeded"
    )
    agent_cfg["experiment"]["wandb_kwargs"]["group"] = agent_cfg["experiment"]["experiment_name"]
    agent_cfg["trainer"]["max_global_timesteps_M"] = max_training_timesteps_M
    _assert_log_root_safe(agent_cfg)

    test_seeds = [5, 6, 7, 8, 9, 10]
    print("Running best trial on seeds (still warm-start):", test_seeds)
    writer = Writer(agent_cfg, delay_wandb_startup=True)
    for seed in test_seeds:
        writer.setup_wandb(name=str(seed))
        warm_start_train(env, env_cfg, agent_cfg, writer, seed, checkpoint, trial=None)
        writer.close_wandb()

    env.close()
    simulation_app.close()
