#!/usr/bin/env bash
# 연구 노트 14 M4 전체 — task-v4 · channels-v3 · 진짜 + 재배선 s0~s2(투영기 학습과 같은 대조군 수).
# 학습 에피소드 0~199 · 평가 1000~1199(판정력 — 노트 14 §6.2) → 묶기 fly-loop-v4 → 선형 검사(전 채널 · 냄새 여러 창).
# 실측: collect-probe 뇌 4 · 결정당 9.35 s · 3.3 GB(runs/collect-probe-20260926-223206.json) · 작은 시험 뇌 2 · 100 에피소드 48 분 → 약 5 시간.
set -u
cd "$(dirname "$0")/.."
LOG="runs/m4-full-$(date +%Y%m%d-%H%M%S).log"
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; .venv/bin/fly-connectome "$@" >>"$LOG" 2>&1; rc=$?; echo "=== $(date '+%F %T') 끝 코드 $rc · $*" | tee -a "$LOG"; [ $rc -eq 0 ] || exit 5; }
step collect --name v4main --seeds 0-199 --rewired-seeds 0-2 --workers 3
step collect --name v4eval --seeds 1000-1199 --rewired-seeds 0-2 --workers 3
step dataset pack --name fly-loop-v4 --train v4main --eval v4eval
step linear-probe --dataset fly-loop-v4 --judge-only
for ch in smell_ln smell; do
  step odor-windows --dataset fly-loop-v4 --channel "$ch" --windows 1,2,4 --max-decision 4
  step odor-windows --dataset fly-loop-v4 --channel "$ch"
done
echo "=== $(date '+%F %T') 전체 끝 · $LOG" | tee -a "$LOG"
