#!/usr/bin/env bash
# 60 · V1 장면 진단(10-17 · 판정 아님) — 사람이 그림에 나오나 · 잔디 텍스처 · 몸 돎의 그림 변화. 헤드리스 크롬 하나 · 수십 초.
source "$(dirname "$0")/../common.sh"
no_lock
run "$PY" scripts/60-eyeworld/v1_scene_probe.py
