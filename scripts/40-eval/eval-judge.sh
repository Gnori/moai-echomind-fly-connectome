#!/usr/bin/env bash
# 40 · 채점 평가 + 판정만 다시(학습은 그대로 · Qwen · 뇌 4 × 저장본 2 × 약 16 분 ≈ 2 h + 판정 수 초)
# 사용: scripts/40-eval/eval-judge.sh <판> <묶음> [뇌…]
#   예: scripts/40-eval/eval-judge.sh train-v6 fly-loop-v4
# ⚠️ eval-scored 의 --dataset 기본값은 fly-loop-v2 다 — 반드시 준다(09-27 발견).
# 판정 규칙은 투영기 meta 의 자료 판으로 자동(stats-v3 = 냄새 쪽 칸 · 찍기 대비 p).
source "$(dirname "$0")/../common.sh"
V=${1:?판}; DS=${2:?묶음}; shift 2; BR=${*:-real rewired-s0 rewired-s1 rewired-s2}
no_lock
for ck in final stage1-caption; do
  for b in $BR; do step 63 eval-scored --brain "$b" --checkpoint "$ck" --version "$V" --dataset "$DS"; done
  step 0 judge-captions --version "$V" --checkpoint "$ck"
done
say "판정 runs/caption-judge-$V-{final,stage1-caption}.json · 표본 화면 http://127.0.0.1:8770/runs/dashboard/samples.html"
