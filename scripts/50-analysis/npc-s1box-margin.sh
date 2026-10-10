#!/usr/bin/env bash
# 50 · S1 상자 월드 여유 어림(10-10 · 가벼움 · 기록만 읽음) → runs/npc-s1box-margin-*.json
source "$(dirname "$0")/../common.sh"
W=${1:-$HOME/Documents/GitHub/moai-echomind/moai-echomind-world-3d/logs/box/20261003-fly/wt/moai-echomind-world-3d/logs/box/fly/world_exp.jsonl}
run "$PY" scripts/50-analysis/npc_s1box_margin.py "$W"
