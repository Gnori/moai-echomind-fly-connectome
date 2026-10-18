#!/usr/bin/env bash
# 50 · 촉각 ② 가능성 작은 시험(R-1177 F-011 · 10-18 · 판정 아님 · 결과 전 잠금 touch-probe-a) — 앞다리 · 머리 털 좌우 ×
#   100 · 150 Hz × 300 ms × 씨앗 0 → 몸 하강 뉴런 · 진짜 대 섞은 배선 s0~s2. CPU 약 2~3 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/50-analysis/touch_probe.py
