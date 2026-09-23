#!/usr/bin/env bash
# R-3340 C 단계 전체 — 학습(진짜 · 섞은 배선) → 캡션 평가 넷 → 성적표.
# 요구자 2026-09-22 "전부 한 번에" · "이 학습에만 풀고 지금 돌림"(학습은 시작 전 여유 80 %).
# 단계마다 메모리 여유가 문턱 위로 돌아올 때까지 기다린다 — 큰 프로세스가 끝난 직후엔 회수가 늦다.
set -u
cd "$(dirname "$0")/.."
LOG="runs/full-$(date +%Y%m%d-%H%M%S).log"

free_pct() { memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; }

wait_free() {  # $1 = 문턱 %
  local need=$1 i=0
  while [ "$(free_pct)" -lt "$need" ]; do
    [ $i -eq 0 ] && echo "--- 메모리 여유 $(free_pct)% < $need% — 기다림" | tee -a "$LOG"
    i=$((i + 1)); [ $i -gt 60 ] && { echo "--- 30 분 기다려도 안 됨 — 중단" | tee -a "$LOG"; exit 3; }
    sleep 30
  done
}

run() {  # $1 = 문턱 %, 나머지 = fly-connectome 인자
  local need=$1; shift
  wait_free "$need"
  echo "=== $(date '+%F %T') 여유 $(free_pct)% · $*" | tee -a "$LOG"
  .venv/bin/fly-connectome "$@" >>"$LOG" 2>&1
  local rc=$?                      # $(date) 뒤의 $? 는 date 의 것 — 먼저 받아 둔다
  echo "=== $(date '+%F %T') 끝 코드 $rc · $*" | tee -a "$LOG"
}

# 이어 가기: bin/run_full.sh --resume-real <진짜 배선 캡션 단계 저장본> — 진짜 배선은 행동 단계만 돈다
REAL=()
[ "${1:-}" = "--resume-real" ] && REAL=(--resume-caption "$2")
run 80 train --brain real ${REAL[@]+"${REAL[@]}"}
run 80 train --brain rewired
run 65 eval-caption --brain real --checkpoint stage1-caption
run 65 eval-caption --brain rewired --checkpoint stage1-caption
run 65 eval-caption --brain real --checkpoint final
run 65 eval-caption --brain rewired --checkpoint final
run 65 scorecard --seeds 2000-2009
echo "=== $(date '+%F %T') 전체 끝 · 로그 $LOG" | tee -a "$LOG"
