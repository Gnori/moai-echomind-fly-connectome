#!/usr/bin/env bash
# 60 · F0(R-1177) — 앞단 v3 만(뇌 없음): 대비 곡선 · 경기장 가로 대비 · 렌더 제자리 돎 · 좌우 대응.
#   결과 전 잠금 eyeworld.yaml f0(f0-a) · frontend_v3. 그림은 V1-b 것을 다시 씀(새로 그리지 않음 · 크롬 없음). CPU 몇 분.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/f0_frontend.py "$@"
