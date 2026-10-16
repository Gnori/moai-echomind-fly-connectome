#!/usr/bin/env bash
# 60 · 보기 창(R-1177 F-008 최소판) — 장면을 요구자 브라우저 탭으로 둘러봄 · 초파리가 받는 그림 나란히.
#   판정 아님 · 무거운 잠금 안 씀(정적 서버 하나 · 127.0.0.1:8792 · 읽기 전용) · 60 분 뒤 스스로 닫음. 인자 그대로(--scene).
source "$(dirname "$0")/../common.sh"
run "$PY" scripts/60-eyeworld/view.py "$@"
