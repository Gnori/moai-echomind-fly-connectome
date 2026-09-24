#!/usr/bin/env bash
# R-7469 F-004 · F-005 전체 — 압축 · 관문(compress-v3 · 크기 1) → 투영기 학습 train-v5(진짜 + 재배선 3) → 캡션 평가 8 → 성적표(씨앗 40 · 7 구성).
# v5(2026-09-24 · 요구자 결정 A): v4 는 뇌 · 채널마다 입력 크기가 달라 재배선이 먼저 배웠다 → compress-v3(채널마다 평균 크기 1) 로 4 뇌 다시.
#   v4 진짜 배선 결과(runs/projector-real-train-v4 · 8 h 23 m)는 기록으로 남긴다. v4 순서 스크립트는 bin/run_full_v4.sh 그대로.
# ⚠️ 요구자 확인 뒤 실행한다. 그 전에 `fly-connectome train-probe --steps 12` 로 걸음당 시간 · 최대 메모리를 먼저 잰다(CLAUDE.md 무거운 실행 규약).
# 학습은 시작 전 시스템 메모리 여유 40 % 이상(2026-09-23 요구자 "문턱을 40%로 낮춰 지금 바로" · 실측 최대 31.4 GB / 137 GB ·
#   실행 중 감시 유지 · 옛 값 80 은 90 GB 먹던 때) · 평가 · 성적표도 40 %(Qwen 29 GB 는 guard 가 따로 센다).
# 관문을 통과한 채널이 하나도 없으면 v5 학습을 하지 않는다(F-004 규칙 · 끝 코드 4).
# 단계마다 메모리 여유가 문턱 위로 돌아올 때까지 기다린다 — 큰 프로세스가 끝난 직후엔 회수가 늦다. 30 분 넘게 안 돌아오면 중단(끝 코드 3).
#
# 사용법:
#   bin/run_full_v5.sh                # 압축 · 관문 → 학습 4 → 캡션 평가 8 → 성적표
#   bin/run_full_v5.sh --skip-train   # 투영기가 이미 있을 때: 압축 · 관문 → 캡션 평가 8 → 성적표
# v2 순서(진짜 · 재배선 하나 · 성적표 5 구성)는 bin/run_full.sh 그대로.
set -u
cd "$(dirname "$0")/.."
LOG="runs/full-v5-$(date +%Y%m%d-%H%M%S).log"
DATASET="fly-loop-v2"
VERSION="train-v5"
BRAINS=(real rewired-s0 rewired-s1 rewired-s2)
CONFIGS="vector,text,no_circuit,rewired-s0,rewired-s1,rewired-s2,no_llm"
SEEDS="2000-2039"
TRAIN_SEEDS="0-119"      # 학습 표본 = 에피소드 120(약 1,200 결정 · 요구자 2026-09-23 B 안 — 전부 쓰면 뇌 하나 약 16 h)

SKIP_TRAIN=0; SKIP_COMPRESS=0
for a in "$@"; do
  [ "$a" = "--skip-train" ] && SKIP_TRAIN=1
  [ "$a" = "--skip-compress" ] && SKIP_COMPRESS=1     # 압축 · 관문 결과가 이미 있을 때(다시 시작)
done

free_pct() { memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; }

# 사건 파일(R-7347 · 대시보드가 로그 대신 읽는다) — 한 줄 = 한 사건. $1 = 사건(begin/start/end/abort/done) · $2 = 명령 · $3 = 끝 코드
JLOG="${LOG%.log}.jsonl"
ev() {
  printf '{"ts":"%s","event":"%s","cmd":"%s","free":%s,"code":%s}\n' \
    "$(date '+%F %T')" "$1" "${2:-}" "$(free_pct)" "${3:-null}" >> "$JLOG"
}

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
  echo "=== $(date '+%F %T') 여유 $(free_pct)% · $*" | tee -a "$LOG"; ev start "$*"
  .venv/bin/fly-connectome "$@" >>"$LOG" 2>&1
  local rc=$?                      # $(date) 뒤에 $? 를 읽으면 date 의 0 이 된다(2026-09-24 01:15 실제 그랬다)
  echo "=== $(date '+%F %T') 끝 코드 $rc · $*" | tee -a "$LOG"; ev end "$*" "$rc"
  if [ $rc -ne 0 ]; then           # 멈춘 학습을 두고 다음 뇌로 넘어가지 않는다
    echo "=== $(date '+%F %T') 중단(끝 코드 $rc) · $* · 로그 $LOG" | tee -a "$LOG"; ev abort "$*" "$rc"; exit 5
  fi
}

echo "=== $(date '+%F %T') 시작 · 자료 $DATASET · 판 $VERSION · 학습 건너뜀 $SKIP_TRAIN · 로그 $LOG" | tee -a "$LOG"
ev begin "run_full_v5 · $DATASET · $VERSION"

# 1 · 2. 압축(뇌마다 · 채널마다 PCA · 학습 나눔으로만) → 관문(압축 특징으로 stats 판정)
if [ $SKIP_COMPRESS -eq 0 ]; then
  run 40 compress fit  --dataset "$DATASET"
  run 40 compress gate --dataset "$DATASET"
else
  echo "--- 압축 · 관문 건너뜀(--skip-compress)" | tee -a "$LOG"
fi

# 관문 확인 — gate.json 의 passed_channels 가 비어 있으면 v5 학습을 하지 않는다(jq 없이 python3 로).
GATE_MSG=$(.venv/bin/python - "$DATASET" <<'PY'
import json, sys
from moai_echomind_fly.loop import compress
path = compress.out_dir(sys.argv[1]) / "gate.json"
if not path.exists():
    print(f"관문 결과 없음 — {path}"); sys.exit(4)
passed = json.loads(path.read_text(encoding="utf-8")).get("passed_channels") or []
if not passed:
    print(f"관문 통과 채널 없음 — v5 학습을 하지 않는다 ({path})"); sys.exit(4)
print(f"관문 통과 채널 {passed} ({path})")
PY
)
GATE_RC=$?
echo "--- $GATE_MSG" | tee -a "$LOG"
[ $GATE_RC -ne 0 ] && { echo "=== $(date '+%F %T') 중단(끝 코드 4) · 로그 $LOG" | tee -a "$LOG"; ev abort "관문 통과 채널 없음" 4; exit 4; }

# 3 · 4. 투영기 학습 — 진짜 + 재배선 3 · 같은 절차(대조군)
if [ $SKIP_TRAIN -eq 0 ]; then
  for b in "${BRAINS[@]}"; do
    run 40 train --brain "$b" --dataset "$DATASET" --train-seeds "$TRAIN_SEEDS"
  done
else
  echo "--- 학습 건너뜀(--skip-train) — 기존 runs/projector-<뇌>-$VERSION 을 쓴다" | tee -a "$LOG"
fi

# 5. 캡션 평가 — 뇌 4 × 저장본 2
for ck in stage1-caption final; do
  for b in "${BRAINS[@]}"; do
    run 40 eval-caption --brain "$b" --checkpoint "$ck" --version "$VERSION"
  done
done

# 6. 성적표 — 재배선 투영기 셋과 앙상블 비교(scorecard-v2)
run 40 scorecard --seeds "$SEEDS" --configs "$CONFIGS" --version "$VERSION"

echo "=== $(date '+%F %T') 전체 끝 · 로그 $LOG" | tee -a "$LOG"; ev done "run_full_v5"
