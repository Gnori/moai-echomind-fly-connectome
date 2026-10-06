#!/usr/bin/env bash
# 20 · KC 겹침이 PN 단계에서 이미 생기나(10-06 · 가벼움 · 약 2 분) → runs/pn-overlap-*.json
source "$(dirname "$0")/../common.sh"
no_lock
wait_free 40
run "$PY" scripts/50-analysis/pn_overlap.py \
  real:runs/fit-kc-mbon-b-sim-v3-real-20261006-071233 \
  rewired-s0:runs/fit-kc-mbon-b-sim-v3-rewired-s0-20261006-105313
