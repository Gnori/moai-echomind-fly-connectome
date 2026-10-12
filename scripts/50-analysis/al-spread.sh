#!/usr/bin/env bash
# 50 · 촉각엽 번짐 진단(10-12 · 가벼움 · 약 30 초) → runs/al-spread-*.json
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/al_spread.py "${1:-vinegar6}"
