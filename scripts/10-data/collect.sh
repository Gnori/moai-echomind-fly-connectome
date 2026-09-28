#!/usr/bin/env bash
# 10 · 자료 모으기(무거움 · 몸 + 뇌) — 작은 측정 → 학습 나눔 → 평가 나눔 → 묶기 → 체크섬.
# 사용: scripts/10-data/collect.sh <묶음이름> <학습씨앗> <평가씨앗> <재배선씨앗> [일꾼]
#   예(fly-loop-v4 · 3 h 36 m): scripts/10-data/collect.sh fly-loop-v4 0-199 1000-1199 0-2 3
#   예(작은 시험 · 48 분):       scripts/10-data/collect.sh fly-loop-v4-pilot 0-59 1000-1039 0 3
# 과제 · 채널 · 시야 판은 지금 config 를 쓴다(task · channels · retino) — 자료 카드에 판이 적힌다. 차원이 바뀌면 새 이름.
# 결정당 시간: 뇌 4(진짜 + 3) 일꾼 하나 9.35 s · 일꾼 3 이면 서로 느려진다(v2: 결정당 17~18 s). 먼저 collect-probe 로 잰다.
source "$(dirname "$0")/../common.sh"
NAME=${1:?묶음 이름}; TRAIN=${2:?학습 씨앗}; EVAL=${3:?평가 씨앗}; REW=${4:?재배선 씨앗}; W=${5:-3}
PRE=${NAME//fly-loop-/}; PRE=${PRE//./}
no_lock
step 40 collect-probe --seeds 0-1 --rewired-seeds "$REW"
say "위 collect-probe 의 sec_per_decision · peak_rss_gb 를 보고 계속합니다(Ctrl-C 로 멈춤 · 10 초)"; sleep 10
step 40 collect --name "${PRE}main" --seeds "$TRAIN" --rewired-seeds "$REW" --workers "$W"
step 40 collect --name "${PRE}eval" --seeds "$EVAL" --rewired-seeds "$REW" --workers "$W"
step 0 dataset pack --name "$NAME" --train "${PRE}main" --eval "${PRE}eval"
step 0 dataset verify "$NAME"
say "끝 · data/datasets/$NAME · 다음: scripts/50-analysis 의 선형 검사 · scripts/90-wrap/finalize.sh(백업)"
