#!/usr/bin/env bash
# 연구 노트 14 M3 — 자료 fly-loop-v2 를 channels-v3(smell_ln · 안테나엽 LN 4 종)으로 뇌만 재생 → 묶기 → 여러 창 선형 검사.
# 뇌 = 진짜 + 재배선 s0(M1 에서 재배선은 후각 쪽이 침묵) · 평가 전부(40 에피소드) + 학습 에피소드 0-119(train-v5 와 같은 부분).
# 실측(2026-09-26 --verify): 뇌 둘 결정당 약 4.6 s · 3.2 GB → 평가 약 30 분 + 학습 약 90 분.
set -u
cd "$(dirname "$0")/.."
LOG="runs/m3-$(date +%Y%m%d-%H%M%S).log"
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; .venv/bin/fly-connectome "$@" >>"$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') 끝 코드 $rc · $*" | tee -a "$LOG"; [ $rc -eq 0 ] || exit 5; }
step replay --pack fly-loop-v2 --split eval  --name m3-eval  --rewired-seeds 0 --no-retune
step replay --pack fly-loop-v2 --split train --name m3-train --rewired-seeds 0 --no-retune --seeds 0-119
step dataset pack --name fly-loop-v2-ln --train m3-train --eval m3-eval
step odor-windows --dataset fly-loop-v2-ln --channel smell_ln
step odor-windows --dataset fly-loop-v2-ln --channel smell
echo "=== $(date '+%F %T') 전체 끝 · $LOG" | tee -a "$LOG"
