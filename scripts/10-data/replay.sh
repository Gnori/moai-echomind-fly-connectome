#!/usr/bin/env bash
# 10 · 재생(무거움 · 뇌만) — 묶음의 감각 입력(stim.npz)으로 몸 없이 뇌만 다시 돌린다.
# 쓰임: 채널을 늘렸을 때(channels-v3 smell_ln 처럼) · 시야 조율을 바꿨을 때(retino-v2) 몸을 다시 돌리지 않고 새 자료를 만든다.
# 사용: scripts/10-data/replay.sh <원본묶음> <새묶음> <재배선씨앗> [학습씨앗] [--retune]
#   예(M3 · fly-loop-v2-ln · 평가 34 분 + 학습 71 분): scripts/10-data/replay.sh fly-loop-v2 fly-loop-v2-ln 0 0-119
# ① 먼저 검증: 바꾸지 않고 앞 6 결정을 원본과 비교 — "최대 차 0" 이 아니면 멈춘다
#    (09-26 까지 직전 교사 행동을 넘기지 않는 버그가 있었다 · efa0104 에서 고침).
# ② 학습 씨앗을 주면 그 에피소드만(학습 나눔 일부) · 건너뛴 만큼 난수 흐름이 달라져 비트 단위로 같지는 않다.
source "$(dirname "$0")/../common.sh"
SRC=${1:?원본 묶음}; DST=${2:?새 묶음}; REW=${3:?재배선 씨앗}; TSEEDS=${4:-}; RETUNE=${5:-}
PRE=${DST//fly-loop-/}; PRE=${PRE//./}
no_lock
say "검증 · 원본과 같은지(재배선은 원본과 같은 수로)"
# 원본 묶음의 재배선 뇌 파일에서 씨앗을 읽는다(옛 카드는 씨앗 목록이 틀릴 수 있어 파일로 — 56d4b73)
ORIG_REW=$("$PY" -c "from moai_echomind_fly import datasets as d;b=d.rewired_brains('$SRC','eval');print(','.join(x.split('-s')[1] if '-s' in x else '0' for x in b))")
"$FC" replay --pack "$SRC" --split eval --name verify-tmp --rewired-seeds "${ORIG_REW:-$REW}" --no-retune --verify 6 >>"$LOG" 2>&1
grep "^\[verify\]" "$LOG" | tail -6 | tee /dev/stderr | grep -qv "최대 차 0$" && { say "⚠️ 원본과 다르다 — 재생 코드를 먼저 확인(로그 $LOG)"; exit 6; }
say "검증 통과(최대 차 0)"
RT=--no-retune; [ "$RETUNE" = "--retune" ] && RT=""
step 40 replay --pack "$SRC" --split eval --name "${PRE}-eval" --rewired-seeds "$REW" $RT
if [ -n "$TSEEDS" ]; then step 40 replay --pack "$SRC" --split train --name "${PRE}-train" --rewired-seeds "$REW" $RT --seeds "$TSEEDS"
else step 40 replay --pack "$SRC" --split train --name "${PRE}-train" --rewired-seeds "$REW" $RT; fi
step 0 dataset pack --name "$DST" --train "${PRE}-train" --eval "${PRE}-eval"
step 0 dataset verify "$DST"
say "끝 · data/datasets/$DST"
