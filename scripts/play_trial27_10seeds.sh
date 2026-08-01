#!/usr/bin/env bash
# Trial 27 scratch PadTac+BT: 10 sim plays (seeds 0..9), npz by default (no video).
# Optional: VIDEO_SEED=0 to record one RayTracedLighting video for that seed.
#
# Duration: control is 60 Hz. Episode length in cfg is 10 s.
#   RECORD_STEPS=300 →  5 s  (old default)
#   RECORD_STEPS=600 → 10 s  (recommended: one full episode, clean)
#   RECORD_STEPS=900 → 15 s  (spans a hard reset at ~10 s — tac/q jump; avoid for warmup)
#
# Usage:
#   bash scripts/play_trial27_10seeds.sh
#   RECORD_STEPS=600 SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/play_trial27_10seeds.sh
#   VIDEO_SEED=0 bash scripts/play_trial27_10seeds.sh   # which seed gets the video
#   VIDEO_SEED=-1 bash ...                              # npz only (no video)
set -euo pipefail

ROOT="/home/nalin/roto_2"
CKPT="${ROOT}/best_agent_padtac_bt_scratch_trial27.pt"
OUT_TAG="trial27"
RECORD_STEPS="${RECORD_STEPS:-600}"   # ~10 s @ 60 Hz
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
VIDEO_SEED="${VIDEO_SEED:--1}"        # only this seed records video; default -1 = all npz (no video)

cd "${ROOT}/scripts"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "videos/${OUT_TAG}" "sim_logs/${OUT_TAG}"

echo "Checkpoint: ${CKPT}"
echo "RECORD_STEPS=${RECORD_STEPS} (~$(awk "BEGIN{printf \"%.1f\", ${RECORD_STEPS}/60}")s @ 60Hz)  seeds=${SEEDS}"
echo "VIDEO_SEED=${VIDEO_SEED} (only this seed gets --video; others npz-only)"
echo "Episode limit is 10s — prefer RECORD_STEPS<=600 for continuous rollouts."

for SEED in ${SEEDS}; do
  echo ""
  echo "========== seed ${SEED} =========="

  VIDEO_ARGS=()
  if [[ "${SEED}" == "${VIDEO_SEED}" ]]; then
    VIDEO_ARGS=(--video --video_length "${RECORD_STEPS}" --renderer RayTracedLighting)
    echo "(recording video for this seed)"
  else
    echo "(npz only)"
  fi

  python play.py \
    --task Baoding \
    --robot shadowlite_padtac_bt \
    --agent_cfg rl_only_pt_padtac_bt \
    --checkpoint "${CKPT}" \
    --num_envs 1 \
    --seed "${SEED}" \
    --headless \
    --record_steps "${RECORD_STEPS}" \
    "${VIDEO_ARGS[@]}"

  NPZ="sim_policy_log_seed${SEED}.npz"
  if [[ -f "${NPZ}" ]]; then
    DEST="sim_logs/${OUT_TAG}/sim_policy_log_${OUT_TAG}_seed${SEED}.npz"
    cp -f "${NPZ}" "${DEST}"
    cp -f "${NPZ}" "sim_policy_log_${OUT_TAG}_seed${SEED}.npz"
    echo "npz -> ${DEST}"
  else
    echo "WARNING: missing ${NPZ}"
  fi

  if [[ "${SEED}" == "${VIDEO_SEED}" && -f videos/rl-video-step-0.mp4 ]]; then
    DESTV="videos/${OUT_TAG}/rl-video-${OUT_TAG}_seed${SEED}.mp4"
    cp -f videos/rl-video-step-0.mp4 "${DESTV}"
    echo "video -> ${DESTV}"
  fi
done

echo ""
echo "Done. NPZs under scripts/sim_logs/${OUT_TAG}/ and scripts/sim_policy_log_${OUT_TAG}_seed*.npz"
echo "Legacy deploy ckpt: ${ROOT}/best_agent_legacy_padtac_bt_scratch_trial27.pt"
