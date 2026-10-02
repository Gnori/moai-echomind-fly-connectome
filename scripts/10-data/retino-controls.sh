#!/usr/bin/env bash
# 10 · 시야 대응표 대조(R-7469 Task 7 · C) — 무거움(뇌만 · 진짜 뇌 하나 · 대응표 셋)
# 사용: scripts/10-data/retino-controls.sh [원본묶음] [학습씨앗]   (기본 fly-loop-v4 · 0-119)
# 과제 판은 지금 config/task.yaml(task-v5 · retinotopic2). 세 대응표로 같은 원본을 재생해 묶는다:
#   real = retino-v2 · az = retino-v2.shuffled-azimuth-s0(눈 안 방위만 섞음) · aztype = …-type-s0(두 눈까지)
# 그 뒤 50-analysis/retino_controls.py — 물체 쪽(loom_side)을 시각 채널로 읽을 때 진짜 > 섞은 대응표 둘 · p < 0.05 면
# "시야 자리가 쓰였다". 재기(10-03): 결정당 약 2.8 s · 2.7 GB → 평가 200 + 학습 120 에피소드 × 셋 ≈ 6~7 h.
source "$(dirname "$0")/../common.sh"
SRC=${1:-fly-loop-v4}; TSEEDS=${2:-0-119}
no_lock
for spec in "real:" "az:retino-v2.shuffled-azimuth-s0" "aztype:retino-v2.shuffled-azimuth-type-s0"; do
  tag=${spec%%:*}; rule=${spec#*:}
  RR=(); [ -n "$rule" ] && RR=(--retino-rule "$rule")
  step 40 replay --pack "$SRC" --split eval --name "v5ret-$tag-eval" --rewired-seeds "" "${RR[@]}"
  step 40 replay --pack "$SRC" --split train --name "v5ret-$tag-train" --rewired-seeds "" --seeds "$TSEEDS" "${RR[@]}"
  step 0 dataset pack --name "fly-loop-v5-ret-$tag" --train "v5ret-$tag-train" --eval "v5ret-$tag-eval"
  step 0 dataset verify "fly-loop-v5-ret-$tag"
done
run "$PY" scripts/50-analysis/retino_controls.py fly-loop-v5-ret-real fly-loop-v5-ret-az fly-loop-v5-ret-aztype
