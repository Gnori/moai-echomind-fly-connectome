#!/usr/bin/env bash
# 50 · 관문 사후 재판정(절편 · 10-12 · CPU 몇 분) → runs/gate-intercept-*.json
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/50-analysis/gate_intercept.py "${1:-fly-loop-v4}"
