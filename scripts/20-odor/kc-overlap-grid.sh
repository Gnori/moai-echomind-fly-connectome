#!/usr/bin/env bash
# 20 · 가능성 먼저(10-06) — 아는 해의 v7 손실 + b 에서 묶음 배율 격자 17 점 · KC 켠 채 겹침 낮은 점이 있나. 약 10 분.
source "$(dirname "$0")/../common.sh"
no_lock
wait_free 40
run "$PY" scripts/50-analysis/kc_overlap_grid.py
