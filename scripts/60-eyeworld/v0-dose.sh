#!/usr/bin/env bash
# 60 · V0d(R-1177) — 뇌 쪽 필요 입력량: V0 분석식 회전 흐름 × 배율 7 → DNa02 부호(진짜 씨앗 10 · 재배선 s0 씨앗 3).
#   결과 전 잠금 eyeworld.yaml v0d(v0d-a). 인자 그대로(--time-only 로 한 번 시간만). 크롬 없음 · CPU.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v0_dose.py "$@"
