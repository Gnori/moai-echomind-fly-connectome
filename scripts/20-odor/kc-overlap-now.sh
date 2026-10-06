#!/usr/bin/env bash
# 20 · 학습 냄새끼리 KC 겹침을 지금 매개변수로(kc-mbon-c 상한 정하기 전 · 10-06). 무거움(짧음 · 뇌 전체 · 한 번 약 30 초).
# 사용: scripts/20-odor/kc-overlap-now.sh   → runs/kc-overlap-now-*.json
source "$(dirname "$0")/../common.sh"
no_lock
wait_free 40
run "$PY" scripts/50-analysis/kc_overlap_now.py \
  real:none \
  real:runs/fit-kc-sparse-c-sim-v3-real-20261005-011859 \
  real:runs/fit-kc-mbon-a-sim-v3-real-20261005-114119 \
  real:runs/fit-kc-mbon-b-sim-v3-real-20261006-071233 \
  rewired-s0:runs/fit-kc-mbon-b-sim-v3-rewired-s0-20261006-105313
