#!/usr/bin/env bash
# 50 · 스스로 켜지나 — 가능성 작은 시험(R-5402 2.1 · 10-18 · 판정 아님 · 결과 전 잠금 autonomy-probe-a) — 감각 뉴런 쉬는 발화
#   1 · 3 · 10 Hz × 1 s × 씨앗 0 → 몸 하강 뉴런 · 하강 뉴런 전체 · 뇌 퍼짐 · 진짜 대 섞은 배선 s0~s2. CPU 몇 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/autonomy_probe.py "${1:-autonomy_probe}"
