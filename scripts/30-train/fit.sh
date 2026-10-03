#!/usr/bin/env bash
# 30 · 회로 학습(F-006) 1 단계 CMA-ES — 무거움 · 요구자 확인 뒤
# 사용: scripts/30-train/fit.sh <범위> [뇌…]   예: scripts/30-train/fit.sh kc-sparse-b real rewired-s0
#   뇌마다 같은 절차(I v2 §7.4) · 결과 runs/fit-<범위>-<판>-<뇌>-<시각>/{loss.jsonl,params.npz,meta.json}
#   10-03 재기: kc-sparse-b 평가 한 번 16.9 s · 2.2 GB → 개체 13 × 50 세대 ≈ 3 h / 뇌
source "$(dirname "$0")/../common.sh"
SCOPE=${1:?범위}; shift
BRAINS=("${@:-real}")
no_lock
for b in "${BRAINS[@]}"; do
  step 40 fit --scope "$SCOPE" --method cmaes --brain "$b"
done
