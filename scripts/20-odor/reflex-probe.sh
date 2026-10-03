#!/usr/bin/env bash
# 20 · 반사 진단(R-5138 S1 전) — 정면 다가옴에 GF(DNp01)가 켜지나 · 시야 모드 × 증폭 상한. 무거움(짧음)
source "$(dirname "$0")/../common.sh"
no_lock
step 40 reflex-probe "$@"
