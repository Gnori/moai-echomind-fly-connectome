#!/usr/bin/env bash
# 50 · S0b 촉각 이상값 까닭(10-12 · 가벼움 · CPU 수십 초) → runs/touch-anomaly-*.json
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/50-analysis/touch_anomaly.py "${1:-runs/npc-perceive-20261004-151813}"
