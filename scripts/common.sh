#!/usr/bin/env bash
# scripts/ 공통 — 모든 단계 스크립트가 `source "$(dirname "$0")/../common.sh"` 로 읽는다.
# 규칙(CLAUDE.md 무거운 실행 규약): 한 번에 하나(runs/.heavy.lock) · 시작 전 메모리 여유 · 끝 코드를 바로 받는다.
# ⚠️ `명령 | tail` 뒤의 && 는 명령의 실패를 가린다(2026-09-27 ruff · pytest 실패가 두 번 커밋됐다) — step 은 파이프 없이 돈다.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FC="$ROOT/.venv/bin/fly-connectome"
PY="$ROOT/.venv/bin/python"
TAG="${TAG:-$(basename "$0" .sh)}"
LOG="${LOG:-runs/scripts-$TAG-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p runs

say() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

free_pct() { memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; }

# 사건 파일(대시보드 "단계" 표가 읽는다 · bin/run_full_v*.sh 와 같은 형식) — 한 줄 = 한 사건.
# 2026-10-03: scripts/ 로 돌린 실행이 대시보드에 안 남던 것(요구자 "기록이 안되든듯")을 고침.
JLOG="${LOG%.log}.jsonl"
ev() {
  local cmd=${2:-}; cmd=${cmd//\\/\\\\}; cmd=${cmd//\"/\\\"}
  printf '{"ts":"%s","event":"%s","cmd":"%s","free":%s,"code":%s}\n' \
    "$(date '+%F %T')" "$1" "$cmd" "$(free_pct)" "${3:-null}" >> "$JLOG"
}
_finish() { local rc=$?; if [ $rc -eq 0 ]; then ev done "$TAG"; else ev abort "$TAG" "$rc"; fi; }
ev begin "$TAG $*"
trap _finish EXIT

guard_cfg() { "$PY" -c "from moai_echomind_fly import config; print(config.load('guard')['$1'])"; }

# 메모리 여유가 $1 % 이상이 될 때까지 기다린다(간격 · 한도는 config/guard.yaml script_wait_*).
wait_free() {
  local need=$1 i=0 every max
  every=$(guard_cfg script_wait_every_s); max=$(guard_cfg script_wait_max_min)
  while [ "$(free_pct)" -lt "$need" ]; do
    [ $i -eq 0 ] && say "메모리 여유 $(free_pct)% < $need% — 기다림"
    i=$((i + 1)); [ $((i * every)) -gt $((max * 60)) ] && { say "$max 분 기다려도 안 됨 — 중단"; exit 3; }
    sleep "$every"
  done
}

# 다른 무거운 실행이 잠금을 쥐고 있으면 멈춘다(가드도 막지만 먼저 알린다).
no_lock() {
  if [ -f runs/.heavy.lock ]; then
    say "다른 무거운 실행 중: $(cat runs/.heavy.lock) — 끝난 뒤 다시"; exit 2
  fi
}

# step <문턱%> <fly-connectome 인자…> — 문턱 0 이면 기다리지 않는다. 실패하면 끝 코드 5 로 멈춘다.
step() {
  local need=$1; shift
  [ "$need" -gt 0 ] && wait_free "$need"
  say "시작 · 여유 $(free_pct)% · $*"
  ev start "$*"
  "$FC" "$@" >>"$LOG" 2>&1
  local rc=$?
  ev end "$*" "$rc"
  say "끝 코드 $rc · $*"
  [ $rc -eq 0 ] || { say "중단 · 로그 $LOG"; exit 5; }
}

# run <명령…> — fly-connectome 이 아닌 명령(파이썬 분석 등)을 같은 규칙으로.
run() {
  say "시작 · $*"
  ev start "$*"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  ev end "$*" "$rc"
  say "끝 코드 $rc · $*"
  [ $rc -eq 0 ] || { say "중단 · 로그 $LOG"; exit 5; }
}
