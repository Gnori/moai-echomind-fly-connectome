#!/usr/bin/env bash
# 30 · 투영기 학습 → 채점 평가 → 판정 → 성적표 전체(무거움 · 약 2 일) — bin/run_full_v6.sh 를 부른다.
# 사용: scripts/30-train/train-all.sh [--train-only=rewired-s1,rewired-s2] [--skip-train]
#   학습 규칙 판은 src/moai_echomind_fly/loop/train_cli.py TRAIN_RULE(지금 train-v6) · 자료는 bin/run_full_v6.sh DATASET.
#   새 판을 만들 때: TRAIN_RULE version 을 올리고(주석에 까닭) · bin/run_full_v6.sh 를 복사해 DATASET · VERSION · TRAIN_SEEDS 를 바꾼다
#   · config/dashboard.yaml train_rule · plan 을 바꾸고 대시보드를 재시작한다.
# 한 뇌가 멈추면(GPU 오류 등): runs/projector-<뇌>-<판> 을 <폴더>.<까닭>-<시각> 으로 옮기고 --train-only=<그 뇌부터> 로 다시 —
#   이어 가지 않고 처음부터(초기값 씨앗 고정이라 같은 걸음에서 같은 손실로 재현된다 · 09-27 s1 실측).
# 09-27~29 train-v6 실측: 뇌당 5~10 h(GPU 경합에 따라) · eval-scored 한 번 약 16 분(1,708 표본) · 성적표 약 12 h.
source "$(dirname "$0")/../common.sh"
no_lock
say "bin/run_full_v6.sh --skip-compress $* (로그는 runs/full-v6-*.log · 사건은 .jsonl · 대시보드가 읽는다)"
bin/run_full_v6.sh --skip-compress "$@"
rc=$?; say "끝 코드 $rc"; exit $rc
