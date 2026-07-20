#!/usr/bin/env bash
# Trial 15 scratch PadTac+BT play: RayTracedLighting video + sim_policy_log_seed42.npz
# Run from anywhere; cd's into scripts/ so npz/video land next to play.py.
set -euo pipefail

ROOT="/home/nalin/roto_2"
CKPT="${ROOT}/best_agent_padtac_bt_scratch_trial15.pt"
OUT_TAG="trial15"

cd "${ROOT}/scripts"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Avoid clobbering the Trial 10 log if it still sits here.
if [[ -f sim_policy_log_seed42.npz ]]; then
  mv -n sim_policy_log_seed42.npz "sim_policy_log_seed42_prev_$(date +%Y%m%d_%H%M%S).npz" || true
fi
if [[ -f videos/rl-video-step-0.mp4 ]]; then
  mv -n videos/rl-video-step-0.mp4 "videos/rl-video-step-0_prev_$(date +%Y%m%d_%H%M%S).mp4" || true
fi

python play.py \
  --task Baoding \
  --robot shadowlite_padtac_bt \
  --agent_cfg rl_only_pt_padtac_bt \
  --checkpoint "${CKPT}" \
  --num_envs 1 \
  --seed 42 \
  --headless \
  --video \
  --video_length 300 \
  --renderer RayTracedLighting

# Rename outputs so Trial 15 is unambiguous for Gate C / J1–J2 checks.
if [[ -f sim_policy_log_seed42.npz ]]; then
  cp -f sim_policy_log_seed42.npz "sim_policy_log_${OUT_TAG}_seed42.npz"
  echo "npz: $(pwd)/sim_policy_log_${OUT_TAG}_seed42.npz"
fi
if [[ -f videos/rl-video-step-0.mp4 ]]; then
  cp -f videos/rl-video-step-0.mp4 "videos/rl-video-${OUT_TAG}.mp4"
  echo "video: $(pwd)/videos/rl-video-${OUT_TAG}.mp4"
fi

echo "Done. Bring both files back for J1/J2 + sim-tactile deploy."
