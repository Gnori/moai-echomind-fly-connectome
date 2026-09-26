#!/usr/bin/env bash
# 연구 노트 14 M4 뒤 — 자료 fly-loop-v4(task-v4 냄새 맡기 구간 · 더듬이 · channels-v3) 로 train-v6 전체.
# 압축 · 관문(compress-v3 · stats-v3) → 투영기 학습(진짜 + 재배선 3 · 에피소드 0~149) → 채점 평가 8 → 캡션 판정 2 → 성적표(씨앗 40 · 7 구성).
# bin/run_full_v5.sh 를 바탕으로 — 바뀐 것: 자료 · 판 · 평가에 --dataset(v5 는 기본값 fly-loop-v2 를 썼다) ·
#   판정(judge-captions)을 순서에 넣음 · 자유 생성 평가(기록용 · 약 2 시간)는 뺌.
# ⚠️ 요구자 확인 뒤 실행(2026-09-26 "남은거 진행해"). 학습 입력 채널이 v5 와 다를 수 있어 train-probe 로 먼저 잰다.
#
# 사용법: bin/run_full_v6.sh [--skip-compress] [--skip-train]
set -u
cd "$(dirname "$0")/.."
LOG="runs/full-v6-$(date +%Y%m%d-%H%M%S).log"
DATASET="fly-loop-v4"
VERSION="train-v6"
BRAINS=(real rewired-s0 rewired-s1 rewired-s2)
CONFIGS="vector,text,no_circuit,rewired-s0,rewired-s1,rewired-s2,no_llm"
SEEDS="2000-2039"
QWEN_WAIT=63             # Qwen(28.6 GB / 128 GB ≈ 22.3 % → 가드는 62.4 % 이상)을 새로 올리는 평가 · 성적표는 여유가 40 % + 21 % 이상일 때까지 기다린다 —
                         # 가드는 "올린 뒤 예상 여유 ≥ 40 %"를 보는데 스크립트가 40 % 만 기다려 2026-09-25 14:45 에 막혔다(여유 62 % → 예상 39 %)
TRAIN_SEEDS="0-149"      # 학습 표본 = 에피소드 150(task-v4 는 에피소드당 약 8 결정 → 약 1,200 결정 · v5 와 같은 규모)

SKIP_TRAIN=0; SKIP_COMPRESS=0
for a in "$@"; do
  [ "$a" = "--skip-train" ] && SKIP_TRAIN=1
  [ "$a" = "--skip-compress" ] && SKIP_COMPRESS=1     # 압축 · 관문 결과가 이미 있을 때(다시 시작)
done

guard_cfg() { .venv/bin/python -c "from moai_echomind_fly import config; print(config.load('guard')['$1'])"; }
WAIT_EVERY_S=$(guard_cfg script_wait_every_s)      # 기다리는 간격 · 한도는 설정 파일에서(2026-09-26 · 스크립트에 박혀 있던 30 초 × 60 회)
WAIT_MAX_MIN=$(guard_cfg script_wait_max_min)

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
    i=$((i + 1)); [ $((i * WAIT_EVERY_S)) -gt $((WAIT_MAX_MIN * 60)) ] && { echo "--- $WAIT_MAX_MIN 분 기다려도 안 됨 — 중단" | tee -a "$LOG"; exit 3; }
    sleep "$WAIT_EVERY_S"
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
ev begin "run_full_v6 · $DATASET · $VERSION"

# 1 · 2. 압축(뇌마다 · 채널마다 PCA · 학습 나눔으로만) → 관문(압축 특징으로 stats 판정)
if [ $SKIP_COMPRESS -eq 0 ]; then
  run 40 compress fit  --dataset "$DATASET"
  run 40 compress gate --dataset "$DATASET"
else
  echo "--- 압축 · 관문 건너뜀(--skip-compress)" | tee -a "$LOG"
fi

# 관문 확인 — gate.json 의 passed_channels 가 비어 있으면 v6 학습을 하지 않는다(jq 없이 python3 로).
GATE_MSG=$(.venv/bin/python - "$DATASET" <<'PY'
import json, sys
from moai_echomind_fly.loop import compress
path = compress.out_dir(sys.argv[1]) / "gate.json"
if not path.exists():
    print(f"관문 결과 없음 — {path}"); sys.exit(4)
passed = json.loads(path.read_text(encoding="utf-8")).get("passed_channels") or []
if not passed:
    print(f"관문 통과 채널 없음 — v6 학습을 하지 않는다 ({path})"); sys.exit(4)
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

# 5. 캡션 평가 — 뇌 4 × 저장본 2 × 방식 2
#   eval-scored(eval-v2 · 채점 · 서식 무관 · 표본당 0.3 s) 를 먼저, 자유 생성 eval-caption(v1 · 기록용) 은 뒤에.
#   v4 진짜 배선에서 v1 은 서식만 쟀다(값은 맞히는데 틀을 안 배워 0 점 · 2026-09-24 요구자 결정 B).
for ck in final stage1-caption; do
  for b in "${BRAINS[@]}"; do
    run $QWEN_WAIT eval-scored --brain "$b" --checkpoint "$ck" --version "$VERSION" --dataset "$DATASET"
  done
done
for ck in final stage1-caption; do          # 판정(가벼움 · 파일만 · stats-v3 — 냄새 쪽 칸 포함)
  run 40 judge-captions --version "$VERSION" --checkpoint "$ck"
done

# 6. 성적표 — 재배선 투영기 셋과 앙상블 비교(scorecard-v2)
run $QWEN_WAIT scorecard --seeds "$SEEDS" --configs "$CONFIGS" --version "$VERSION"

echo "=== $(date '+%F %T') 전체 끝 · 로그 $LOG" | tee -a "$LOG"; ev done "run_full_v6"
