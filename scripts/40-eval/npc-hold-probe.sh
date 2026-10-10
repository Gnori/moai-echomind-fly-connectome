#!/usr/bin/env bash
# 40 · 작은 시험(10-10 · 가능성 먼저) — 도망 뉴런이 켜지면 비켜서기를 300 ms 이어 가기(도망 프로그램) · 구간 3 · 6 칸.
#   가짜 세계 · 진짜 배선 · 씨앗 6 · 판정 아님. 상자 월드 판정선 0.595 m = 3.97 mm 와 견준다. 약 8 분.
source "$(dirname "$0")/../common.sh"
N=${1:-6}
no_lock
step 40 npc-run --stage hold300-bins3-fake --world fake --bins 3 --reflex-hold-ms 300 --configs real --n "$N"
step 40 npc-run --stage hold300-bins6-fake --world fake --bins 6 --reflex-hold-ms 300 --configs real --n "$N"
