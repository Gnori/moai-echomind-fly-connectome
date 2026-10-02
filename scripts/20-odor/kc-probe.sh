#!/usr/bin/env bash
# 20 · KC 희소성 M0(연구 노트 16 §6) — 학습 없이 냄새 자극별 KC 비율 · PN 발화율 · APL · 나머지 뇌. 무거움(짧음 · 수 분)
# 사용: scripts/20-odor/kc-probe.sh   → runs/kc-probe-<시각>.json
# 판정(결과 전 · 노트 16 §6): 식초 6 · 300 ms 의 KC 비율이 이미 [0.05, 0.10] 이면 학습하지 않는다 ·
#   자극 PN 발화율 ≥ 120 Hz → 범위 kc-sparse-a(29 · 코드 고침 없음) · 아래면 kc-sparse-b(24 · 코드 고침)
source "$(dirname "$0")/../common.sh"
no_lock
step 40 kc-probe
