#!/usr/bin/env bash
# 30 · 회로 학습 작은 시험(가능성 먼저 · 10-13) — 세대 몇 개만. 사용: scripts/30-train/fit-small.sh <범위> <뇌> [세대 5]
#   결과 runs/fit-<범위>-<판>-<뇌>-<시각>/ · 판단 기준은 범위 주석(결과 전)에.
source "$(dirname "$0")/../common.sh"
SCOPE=${1:?범위}; BRAIN=${2:-real}; GEN=${3:-5}
no_lock
step 40 fit --scope "$SCOPE" --method cmaes --brain "$BRAIN" --generations "$GEN"
