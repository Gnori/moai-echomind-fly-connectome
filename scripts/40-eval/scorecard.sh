#!/usr/bin/env bash
# 40 · 성적표만 다시(Qwen · 몸 · 씨앗 40 × 7 구성 · train-v6/task-v4 약 12 h · 약 29 GB)
# 사용: scripts/40-eval/scorecard.sh <판> [씨앗범위]
#   예: scripts/40-eval/scorecard.sh train-v6 2000-2039
# ⚠️ 과제는 지금 config/task.yaml(판을 바꾼 뒤 옛 성적표와 섞지 않는다) · 씨앗 40 은 95 % 구간 폭 약 ±0.15.
# 결과: runs/scorecard-<시각>/scorecard.json + <구성>.jsonl + <구성>.decisions.jsonl(scorecard-v3)
# 분석: scripts/50-analysis/decision_turns.py <폴더> · 대시보드 episodes.html
source "$(dirname "$0")/../common.sh"
V=${1:?판}; S=${2:-2000-2039}
no_lock
step 63 scorecard --seeds "$S" --configs vector,text,no_circuit,rewired-s0,rewired-s1,rewired-s2,no_llm --version "$V"
D=$(ls -td runs/scorecard-*/ | head -1); say "결과 $D"
run "$PY" scripts/50-analysis/decision_turns.py "$D"
