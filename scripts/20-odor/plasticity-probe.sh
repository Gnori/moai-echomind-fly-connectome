#!/usr/bin/env bash
# 20 · 버섯체 가소성 먼저 재기(F-007 · I v2 §8.3) — 무거움(짧음)
# 사용: scripts/20-odor/plasticity-probe.sh [냄새 ORN 종류들] [η 후보] [회로 학습 폴더] [씨앗들(쉼표)] [뇌 real|rewired-s0]
#   10-06: 씨앗 여럿 · η 0(가소성 끔 = 시행 간 요동 대조)과 나란히
#   10-03 첫 재기(DM1+VA2 · 학습 전): KC 0.2 % · MBON 침묵 → 잴 수 없음
#   M3(노트 16 §6): 식초 6 + 회로 학습 매개변수로 다시
source "$(dirname "$0")/../common.sh"
ODOR=${1:-ORN_DM1,ORN_DM4,ORN_DP1m,ORN_DM2,ORN_VA2,ORN_VM2}; ETA=${2:-0.001,0.01,0.1}; CELL=${3:-}; SEEDS=${4:-0}; BRAIN=${5:-real}
no_lock
for SEED in ${SEEDS//,/ }; do
  if [ -n "$CELL" ]; then step 40 plasticity-probe --odor "$ODOR" --eta "$ETA" --cell-from "$CELL" --seed "$SEED" --brain "$BRAIN"
  else step 40 plasticity-probe --odor "$ODOR" --eta "$ETA" --seed "$SEED"; fi
done
