#!/usr/bin/env bash
# 40 · 작은 시험(10-10 · 가능성 먼저) — 반사 구간 100 ms(3 칸) 대 50 ms(6 칸) · 가짜 세계 · 진짜 배선 · 씨앗 6.
#   판정 아님 — 50 ms 에서도 도망 뉴런이 켜지고 · 더 멀리 비키는지만 본다. 약 10 분.
source "$(dirname "$0")/../common.sh"
N=${1:-6}
no_lock
step 40 npc-run --stage bins3-fake --world fake --bins 3 --configs real --n "$N"
step 40 npc-run --stage bins6-fake --world fake --bins 6 --configs real --n "$N"
