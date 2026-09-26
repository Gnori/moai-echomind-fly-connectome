#!/usr/bin/env bash
# 연구 노트 14 M4 작은 시험 — task-v4(냄새 맡기 4 결정 · 먹이 4~7 mm · 옆 방위) · channels-v3 · 진짜 + 재배선 s0.
# 학습 60 에피소드 + 평가 40 → 묶기 fly-loop-v4-pilot → 냄새 맡기 구간(결정 < 4)과 전체를 LN4 · 투사 뉴런으로 여러 창 검사.
# 실측(2026-09-26 · 2 에피소드 · 일꾼 1): 결정당 약 7.5 s → 약 850 결정 ÷ 일꾼 3 ≈ 35 분.
set -u
cd "$(dirname "$0")/.."
LOG="runs/m4-pilot-$(date +%Y%m%d-%H%M%S).log"
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; .venv/bin/fly-connectome "$@" >>"$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') 끝 코드 $rc · $*" | tee -a "$LOG"; [ $rc -eq 0 ] || exit 5; }
step collect --name v4pilot-main --seeds 0-59 --rewired-seeds 0 --workers 3
step collect --name v4pilot-eval --seeds 1000-1039 --rewired-seeds 0 --workers 3
step dataset pack --name fly-loop-v4-pilot --train v4pilot-main --eval v4pilot-eval
for ch in smell_ln smell; do
  step odor-windows --dataset fly-loop-v4-pilot --channel "$ch" --windows 1,2,4 --max-decision 4
  step odor-windows --dataset fly-loop-v4-pilot --channel "$ch"
done
echo "=== $(date '+%F %T') 전체 끝 · $LOG" | tee -a "$LOG"
