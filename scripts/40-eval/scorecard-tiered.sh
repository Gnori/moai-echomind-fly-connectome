#!/usr/bin/env bash
# 40 · 층 구조 성적표(R-1027 F-004 · 연구 노트 15 §7 T4) — 무거움(Qwen · 몸) · 요구자 확인 뒤
# 사용: scripts/40-eval/scorecard-tiered.sh <판> [씨앗범위] [--measure]
#   --measure : 먼저 씨앗 2 개로 재기(무거운 실행 규약 — 구성마다 에피소드 시간 · 호출률)
# 순서가 중요하다: `<바탕>-gated` 가 같은 성적표 안에서 `-random-matched` · `-periodic-matched` 보다 먼저(호출률을 거기서 맞춤).
# 문턱은 config/control.yaml(control-v2 · 8756f86 에서 잠금) · 빠른 정책은 runs/fast-policy-<자료>-<뇌>-fast-v2.
# 결과: runs/scorecard-<시각>/scorecard.json(judgement_gated · judgement_random_matched · gating) + 결정 기록(tier · trigger · z)
source "$(dirname "$0")/../common.sh"
V=${1:?판}; S=${2:-2000-2039}
[ "${3:-}" = "--measure" ] && S=2000-2001
no_lock
CFGS=vector,fast-only,vector-gated,vector-random-matched,vector-periodic-matched
CFGS=$CFGS,rewired-s0-gated,rewired-s1-gated,rewired-s2-gated
CFGS=$CFGS,rewired-s0-fast-only,rewired-s1-fast-only,rewired-s2-fast-only
step 63 scorecard --seeds "$S" --configs "$CFGS" --version "$V"
D=$(ls -td runs/scorecard-*/ | head -1); say "결과 $D"
run "$PY" scripts/50-analysis/useful_calls.py "$D"
