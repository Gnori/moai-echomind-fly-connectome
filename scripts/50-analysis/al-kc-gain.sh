#!/usr/bin/env bash
# 50 · 촉각엽 고침 뒤 PN → KC 배율만 올려 KC 를 다시 켤 수 있나(10-12 · 가능성 먼저 · 약 5 분)
source "$(dirname "$0")/../common.sh"
no_lock
wait_free 40
run "$PY" scripts/50-analysis/al_kc_gain.py "${1:-0.5,0.25}" "${2:-1.25,1.5,2.0}"
