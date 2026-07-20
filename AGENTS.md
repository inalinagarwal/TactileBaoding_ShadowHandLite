# RoTO Codebase Index (Agent Context)

> **Purpose:** Onboarding doc for AI agents and developers. Read this first before exploring `roto-main/`.
> **User focus:** Shadow Hand Lite only — sim-to-real deployment with **16 physical FSR sensors** mapped to pretrained policies expecting **14 binary tactile channels**.

---

## 1. What this repo is

**RoTO (Robot Tactile Olympiad)** — RL benchmark for tactile dexterous manipulation, built on **Isaac Sim + Isaac Lab**.

| Component | Location | Role |
|-----------|----------|------|
| **This repo (`roto`)** | `roto-main/` | Environments: robots, tasks, rewards, observations, agent YAML paths |
| **`multimodal_rl`** (external) | separate clone | PPO, encoders, self-supervision, trainer, W&B — **not in this workspace** |
| **Isaac Lab** | pip/conda | `DirectRLEnv`, physics, vectorized sim |

Split is intentional: `roto` = world/task; `multimodal_rl` = learning loop.

**Install chain:** conda → Isaac Sim + Isaac Lab → `pip install -e multimodal_rl` → `pip install -e roto`

---

## 2. Directory map

```
roto-main/
├── scripts/
│   ├── train.py          # Main training entry
│   ├── play.py           # Eval / video / checkpoint playback
│   ├── sweep.py          # Optuna hyperparameter sweeps
│   └── common_utils.py   # Env creation, model build, hydra registration, hand-task helpers
├── roto/
│   ├── assets/           # Robot spawn configs (USD/URDF) + meshes
│   │   ├── shadow_hand_lite.py
│   │   └── shadow_lite/sr_hand.urdf
│   └── tasks/
│       ├── roto_env.py   # Shared base env (control, obs dict, tactile, camera)
│       ├── physics.py    # Sim tuning (240 Hz physics, decimation=4 → 60 Hz control)
│       ├── find/         # Franka only
│       ├── bounce/       # 4 hands
│       ├── baoding/      # 4 hands
│       └── robots/
│           ├── shadowlite/shadowlite.py   # ← USER'S ROBOT
│           ├── shadow/, orca/, allegro/, franka/
└── setup.py
```

---

## 3. Class inheritance (core design)

```
Isaac Lab DirectRLEnv
    └── RotoEnv / RotoEnvCfg          roto/tasks/roto_env.py
            └── ShadowLiteEnv         roto/tasks/robots/shadowlite/shadowlite.py
                    └── BaodingShadowLiteEnv / BounceShadowLiteEnv
                            roto/tasks/{baoding, bounce}/*.py
```

| Layer | Owns |
|-------|------|
| `RotoEnv` | Joint pos control, obs dict (`prop`, `tactile`, `rgb`, `depth`, `gt`), contact sensors, cameras |
| `ShadowLiteEnv` | Spawn Shadow Lite URDF, tactile reading override |
| `*Task*Env` | Objects, rewards, resets, ground-truth obs |

Cfg pattern: `BaodingShadowLiteCfg(BaodingTaskCfg, ShadowLiteEnvCfg)` — task params + robot params via multiple inheritance.

---

## 4. Tasks × robots

| Task | Gym ID | Robots | Reward (summary) |
|------|--------|--------|------------------|
| **Find** | `Find` | Franka only | Distance EE → ball |
| **Bounce** | `Bounce` | shadow, shadowlite, orca, allegro | Sparse bounce bonus (10s ep) |
| **Baoding** | `Baoding` | shadow, shadowlite, orca, allegro | Distance + rotation bonus (10s ep) |

For Bounce/Baoding: Gym ID stays `Bounce` / `Baoding`; **`--robot shadowlite`** selects env cfg class + agent YAML folder.

```bash
cd scripts/
python train.py --task Baoding --robot shadowlite --agent_cfg rl_only_pt --num_envs 2048 --headless --seed 1234
python play.py  --task Baoding --robot shadowlite --agent_cfg rl_only_pt --checkpoint <path> --headless
```

See **§5 Training (Shadow Lite Baoding)** for RTX 4090 budget / timestep settings.

---

## 5. Training (Shadow Lite Baoding)

### Prerequisites

1. Fix hardcoded URDF paths in `roto/assets/shadow_hand_lite.py` (lines 31–32) to your local `roto/assets/shadow_lite/` tree.
2. Install chain: Isaac Sim + Isaac Lab → `pip install -e multimodal_rl` → `pip install -e roto`.

### Recommended command (RTX 4090 24GB)

Default `max_global_timesteps_M: 250` in `default.yaml` is a **multi-day** run. For a first decent run (~hours), lower it first:

```yaml
# roto/tasks/baoding/agents/shadowlite/default.yaml
trainer:
  max_global_timesteps_M: 50   # bump to 100–200 once learning; 250 for paper-scale
```

```bash
cd /path/to/roto-main/scripts

python train.py \
  --task Baoding \
  --robot shadowlite \
  --agent_cfg rl_only_pt \
  --num_envs 2048 \
  --headless \
  --seed 1234
```

| Setting | Value | Notes |
|---------|-------|-------|
| `--num_envs` | **2048** | Safe on 24GB; try **4096** if VRAM headroom; drop to **1024** on OOM |
| `max_global_timesteps_M` | **50** (edit yaml) | No CLI override; full budget is **250** |
| `--agent_cfg` | `rl_only_pt` | Blind prop + tactile (sim-to-real target) |
| `--headless` | required | Normal for SSH / remote GPU (no Isaac GUI) |

`rl_only_pt.yaml` sets `wandb: 0`; merged `default.yaml` otherwise has `wandb: 1` — blind preset disables W&B.

### Outputs

| What | Where |
|------|-------|
| Checkpoints, TensorBoard | `./shadowlite_baoding/` (cwd = `scripts/`; `LOG_PATH = os.getcwd()`) |
| Eval envs | 100 parallel (`trainer.num_eval_envs` in default.yaml) |
| Videos during train | Off by default (`upload_videos: 0`); use `play.py --headless --video` for rollouts |

### Baoding Shadow Lite task params (`BaodingShadowLiteCfg`)

| Param | Value |
|-------|-------|
| Ball mass | **55 g** (inherited from `BaodingTaskCfg`) |
| Ball diameter | **1.2 in** (Lite-specific; full Shadow uses 1.5 in) |
| Ball reset height | **0.46 m** |
| Episode | 10 s @ 60 Hz control |
| Hand tilt | rot `(0, 0, -0.7933, 0.6087)` (~15° forward) |

---

## 6. Training / inference flow

```
train.py / play.py
  → common_utils.resolve_gym_env_id(task, robot)
  → register_hand_task_to_hydra()  # picks ShadowLite cfg + loads agents/shadowlite/*.yaml
  → make_env()                     # copies obs_list, tactile_cfg from agent YAML into env cfg
  → gym.make("Baoding", cfg=...)
  → baoding_make_env(cfg)          # returns BaodingShadowLiteEnv if cfg is BaodingShadowLiteCfg
  → multimodal_rl: Encoder + PPO (+ optional SSL)
```

**Agent configs:** `roto/tasks/{baoding,bounce}/agents/shadowlite/{agent_cfg}.yaml`

Common presets:
- `rl_only_pt` — blind (prop + tactile)
- `rl_only_ptg` — + ground truth (privileged)
- `forward_dynamics` — blind + self-supervised forward dynamics

Obs toggles live in YAML under `observations.obs_list`, `obs_stack`, `tactile_cfg`, `pixel_cfg`.

---

## 7. Observations (Shadow Lite blind policy)

From **RoTO 2.0 paper Table I** (authoritative for published checkpoints):

| Modality | Dim (per step) | Notes |
|----------|---------------:|-------|
| Tactile (binary) | **14** | One value per task-relevant link |
| Joint positions | 16 | |
| Joint velocities | 16 | |
| Joint command error | 13 | Paper; code may use 16 actuated joints |
| Last action | 13 | Paper; code has `num_actions=16` in shadowlite.py |
| **Total (1 step)** | **72** | |
| **Total (stack k=4)** | **288** | All shadowlite agent YAMLs use `obs_stack: 4` |
| **Actions out** | **13** (paper) / **16** (current code) | **Verify against your checkpoint** |

Proprio is built in `RotoEnv._get_proprioception()`: normalized joint pos/vel, joint error, last action for `actuated_dof_indices`.

Tactile is built in `ShadowLiteEnv._get_tactile()`: contact force norm → binary via `tactile_cfg.binary_threshold`.

---

## 8. Tactile / contact sensors (Shadow Lite) — CRITICAL

### Where configured

| What | File | Key |
|------|------|-----|
| Which links get contact sensors | `roto/tasks/robots/shadowlite/shadowlite.py` | `robot_contact_sensor_cfg.prim_path` |
| Enable contact on asset | `roto/assets/shadow_hand_lite.py` | `activate_contact_sensors=True` |
| Binary threshold | `roto/tasks/*/agents/shadowlite/*.yaml` | `tactile_cfg.binary_threshold` (0.0 or 0.01) |
| Collision geometry (sim contact locations) | `roto/assets/shadow_lite/sr_hand.urdf` | per-link `<collision>` meshes |
| Reading logic | `shadowlite.py::_get_tactile()` | force norm → binary |

### Current code vs paper

**Paper:** 14 binary contacts on task-relevant links.

**Current code issue:** Shadow Lite uses a **broad** regex:
```python
prim_path="/World/envs/env_.*/Robot/.*"   # ALL robot links (~19), not filtered to 14
```
Full Shadow Hand uses a selective regex (distal/middle/proximal/palm/lfmetacarpal).

**Reindexing:** `_get_tactile()` docstring says legacy order `[distal, proximal, middle, palm, metacarpal]` but **reindex is NOT implemented** — order = Isaac `body_names` default.

**Always verify at runtime:**
```python
names = env.unwrapped.robot_contact_sensor.body_names
print(len(names), names)
print(env.unwrapped.robot_contact_sensor.data.net_forces_w.shape)  # [N, B, 3]
```

### Intended 14 channels (inferred — verify before sim-to-real)

Legacy Shadow ordering adapted for Lite (no little finger):

| Idx | URDF link | Segment |
|----:|-----------|---------|
| 0–3 | `rh_ffdistal`, `rh_mfdistal`, `rh_rfdistal`, `rh_thdistal` | distals |
| 4–7 | `rh_ffproximal`, `rh_mfproximal`, `rh_rfproximal`, `rh_thproximal` | proximals |
| 8–11 | `rh_ffmiddle`, `rh_mfmiddle`, `rh_rfmiddle`, `rh_thmiddle` | middles |
| 12 | `rh_palm` | palm |
| 13 | `rh_thbase` | thumb base (14th — **confirm via body_names**) |

**Excluded from intended 14:** `rh_forearm`, knuckles (`rh_*knuckle`), `rh_thhub`, tips (`rh_*tip`), `rh_imu`, `rh_manipulator`.

URDF has legacy Gazebo contact sensors on 4 distal links only — **Isaac uses ContactSensorCfg, not Gazebo tags**.

---

## 9. User's sim-to-real plan (16 FSR → 14 policy)

**User intent:** Deploy pretrained Shadow Lite policy on real hardware with **16 FSR pads at custom positions** (not necessarily on URDF link centroids).

**Gap:** Policy expects **14-dim binary tactile × obs_stack 4**; hardware gives **16 FSR readings**.

**Required:** A `fsr_16 → tactile_14` mapping layer before policy input. Options:
1. **Semantic OR grouping** — map each FSR to nearest logical channel; merge extras (e.g. 2 palm FSRs → idx 12)
2. **Learned adapter** — small MLP trained on logged sim/real pairs
3. **Retrain** — add 16 collision spheres in URDF at real FSR sites, retrain with 16-dim tactile (best long-term)

Also match: proprio joint order, normalization (`RotoEnv` scale/unscale), control rate **60 Hz**, tactile history buffer depth **4**, action dim from checkpoint.

### Shadow Lite actuated joints (proprio order in code)

```python
['rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_THJ5',
 'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_THJ4',
 'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2',
 'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1',
 'rh_THJ2', 'rh_THJ1']
```

Hand spawn: `hand_height=0.5`, rot `(0, 0, -0.7933, 0.6087)` (~15° forward tilt).

### Asset path caveat

`shadow_hand_lite.py` has hardcoded dev paths (`/home/elle/code/debug/roto/...`) — **must fix for local install**.

---

## 10. Key files quick reference

| Need to change… | File |
|-----------------|------|
| Shadow Lite contact links / tactile logic | `roto/tasks/robots/shadowlite/shadowlite.py` |
| URDF / collision geometry | `roto/assets/shadow_lite/sr_hand.urdf` |
| Robot spawn / actuators | `roto/assets/shadow_hand_lite.py` |
| Baoding task params (ball size, positions) | `roto/tasks/baoding/baoding.py` (`BaodingShadowLiteCfg`) |
| Bounce task params | `roto/tasks/bounce/bounce.py` (`BounceShadowLiteCfg`) |
| Agent hyperparams + obs config | `roto/tasks/{baoding,bounce}/agents/shadowlite/*.yaml` |
| Physics (240 Hz, contact tuning) | `roto/tasks/physics.py` |
| Shared obs/control base | `roto/tasks/roto_env.py` |
| Train/play/sweep glue | `scripts/common_utils.py`, `scripts/train.py` |
| Gym registration (Baoding) | `roto/tasks/baoding/__init__.py` → `baoding_make_env()` |

---

## 11. Checkpoints & paper results

Local clone: **`../roto_paper_results-main/`** — see **`CHECKPOINTS.md`** there for full index of all 30 `.pt` files.

**Shadow Lite Baoding blind:** no published `.pt` (only TensorBoard logs). **`shadow_lite.pt`** = vision (`rl_only_ptg`), duplicate of `shadowlite_bounce_vision.pt`. **Retrain** for blind sim-to-real.

NeurIPS checkpoints (`checkpoints/baoding/`) are **full Shadow Hand only** — do not use with `--robot shadowlite`.

## 12. External references

- **Paper (RoTO 2.0):** [arXiv:2605.21429](https://arxiv.org/abs/2605.21429) — Table I has Shadow Lite obs dims (14 tactile, 72 total/step)
- **NeurIPS 2025 (RoTO 1.0):** tactile RL with Shadow Hand, 17 contacts
- **Checkpoints / logs:** [roto_paper_results](https://github.com/elle-miller/roto_paper_results)
- **Agent code:** [multimodal_rl](https://github.com/elle-miller/multimodal_rl)
- **Project page:** https://elle-miller.github.io/roto/

---

## 13. Known gotchas (don't re-discover these)

1. **`Robot/.*` on Shadow Lite** may expose ~19 contacts, not paper's 14 — filter/reindex before sim-to-real.
2. **`_get_tactile()` reindex is a stub** — docstring ≠ implementation.
3. **Action count mismatch:** paper 13 vs code 16 actuated joints — check checkpoint.
4. **`shadow_hand_lite.py` asset paths** are machine-specific.
5. **`train.py` calls `main()` twice** (lines 87–90) — likely bug.
6. **`max_global_timesteps_M` has no CLI flag** — edit `default.yaml` (or agent yaml) to shorten runs.
7. **No real-robot / FSR bridge code** in this repo — deployment layer is user-built.
8. **Tactile ≠ proprio** — 16 FSRs is unrelated to 16 joint position dims.
9. **Bounce reward** uses tactile contact transitions (`last_tactile` vs `tactile`) in `bounce.py`.

---

## 14. Suggested next steps for this project

- [ ] Fix URDF asset paths in `shadow_hand_lite.py` (lines 31–32)
- [ ] Set `max_global_timesteps_M: 50` for first 4090 training run; scale up if learning
- [ ] Run sim once; log `robot_contact_sensor.body_names` for ground-truth 14 order
- [ ] Tighten `prim_path` regex to match paper's 14 links (mirror full Shadow pattern)
- [ ] Implement explicit tactile reindex in `ShadowLiteEnv._get_tactile()`
- [ ] Document 16 FSR physical IDs → 14 policy channel mapping
- [ ] Build real robot bridge: proprio + FSR → obs dict → policy → joint commands @ 60 Hz
- [ ] Confirm checkpoint action/obs dims before commanding hardware

---

*Last updated: 2026-06-18 — Shadow Hand Lite Baoding blind retrain (4090 24GB), tactile sim-to-real.*
