#!/usr/bin/env bash
# 50 · 촉각엽 번짐 작은 시험(10-12 · 가능성 먼저 · 약 2~3 분) → runs/al-probe-*.json
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/al_probe.py
