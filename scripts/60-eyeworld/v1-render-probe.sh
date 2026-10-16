#!/usr/bin/env bash
# 60 · V1 렌더러 재기(R-1177 · 판정 아님) — arena-a 싣기 · 같은 그림 · 표지 방향 · 시간 · 크롬 메모리. 인자 그대로(--gpu).
#   헤드리스 크롬 하나(우리 프로필 · 우리 그룹) · 에코월드 헤드리스 잠금이 살아 있으면 거부.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v1_render_probe.py "$@"
