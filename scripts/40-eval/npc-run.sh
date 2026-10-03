#!/usr/bin/env bash
# 40 · NPC 하나를 초파리 뇌로(R-5138) — 무거움(뇌 · 몸은 가짜 세계 또는 상자 월드)
# 사용: scripts/40-eval/npc-run.sh [단계] [씨앗 수] [구성]
#   예(S0 재기): scripts/40-eval/npc-run.sh s0 2 real,no_brain
#   예(S1 본):  scripts/40-eval/npc-run.sh s1 40
# 세계는 config/npc.yaml world.kind(fake | box) · 라이브 월드(9212)는 금지.
source "$(dirname "$0")/../common.sh"
STAGE=${1:-s1}; N=${2:-40}; CFGS=${3:-real,rewired-s0,rewired-s1,rewired-s2,no_brain}
no_lock
step 40 npc-run --stage "$STAGE" --n "$N" --configs "$CFGS"
