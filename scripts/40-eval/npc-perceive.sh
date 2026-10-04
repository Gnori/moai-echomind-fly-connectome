#!/usr/bin/env bash
# 40 · 물체 인식 시험(R-5138 S0b) — 무거움(뇌 넷 · 상자 월드 9412) → 판정(50-analysis)
# 사용: scripts/40-eval/npc-perceive.sh [씨앗 수] [섞은 배선]
#   예(작게 먼저): scripts/40-eval/npc-perceive.sh 2
#   예(본):       scripts/40-eval/npc-perceive.sh 60
source "$(dirname "$0")/../common.sh"
N=${1:-60}; REW=${2:-0,1,2}
no_lock
step 40 npc-perceive --n "$N" --rewired "$REW"
