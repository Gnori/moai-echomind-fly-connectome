#!/usr/bin/env bash
# 30 · 회로 학습(F-006) 평가 한 번의 시간 · 메모리 · 처음 손실 재기 — 무거움(짧음)
# 사용: scripts/30-train/fit-probe.sh <범위> [시행 수 둘] [방법]
#   예: scripts/30-train/fit-probe.sh kc-sparse-b 1,3        (mb 범위의 '시행' = 양성 자극 씨앗 수)
#       scripts/30-train/fit-probe.sh escape-small 2,4       (10-03 실측: 시행당 3.3 s + 5.8 s · 2.3 GB · 처음 손실 0)
source "$(dirname "$0")/../common.sh"
SCOPE=${1:?범위}; TRIALS=${2:-1,3}; METHOD=${3:-cmaes}
no_lock
step 40 fit-probe --scope "$SCOPE" --trials "$TRIALS" --method "$METHOD"
