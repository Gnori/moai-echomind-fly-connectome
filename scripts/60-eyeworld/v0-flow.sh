#!/usr/bin/env bash
# 60 · V0(R-1177) — 분석식 광류 → T4/T5 → 뇌 · 진짜 · 자리 섞기 5 · 재배선 3 · 씨앗 10 · 결과 전 잠금 eyeworld.yaml v0-a.
#   인자는 그대로 넘김(시간 재기: --seeds 0 --maps 1 --brains real --no-judge). CPU · 가벼움~중간.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v0_flow.py "$@"
