#!/usr/bin/env bash
# 30 · 회로 학습 뒤 M2(연구 노트 16 §6) — 학습 전 · 뒤: 손실 측정 · 안 쓴 무작위 냄새 5 세트 일반화([0.03, 0.15]) ·
#   도망 반사 재채점 · 발톱 일치 · 진짜 대 재배선 켜진 KC 겹침. 무거움(짧음 · 수 분).
# 사용: scripts/30-train/fit-eval.sh <진짜 fit 폴더> [재배선 fit 폴더]
source "$(dirname "$0")/../common.sh"
REAL=${1:?진짜 fit 폴더}; NULL=${2:-}
no_lock
step 40 fit-eval --run "$REAL"
if [ -n "$NULL" ]; then
  step 40 fit-eval --run "$NULL"
  run "$PY" -c "from moai_echomind_fly.mb_probe import overlap; import sys; print('겹침(Jaccard)', overlap(sys.argv[1], sys.argv[2]))" "runs/fit-eval-$(basename "$REAL").json" "runs/fit-eval-$(basename "$NULL").json"
fi
