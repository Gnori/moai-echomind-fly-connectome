#!/usr/bin/env bash
# 40 · NPC 보기용 시연(R-5138) — 상자 월드에서 NPC 한 몸을 세워 두고 위험만 계속 새로(판정용 아님)
# 사용: scripts/40-eval/npc-demo.sh [씨앗 수] [구성]   예: scripts/40-eval/npc-demo.sh 8 real
# 브라우저: http://localhost:8974/village-bj.html?world=ws://localhost:9412/stream&noowner=1&view=free&at=-76,260&dist=12&pitch=0.6&neural=1
source "$(dirname "$0")/../common.sh"
N=${1:-8}; CFGS=${2:-real}
no_lock
step 40 npc-run --stage demo --n "$N" --configs "$CFGS" --persistent
