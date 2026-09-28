#!/usr/bin/env bash
# 20 · 과제 몸 확인(무거움이지만 짧음 · 에피소드 2 · 약 2 분) — 과제를 바꾼 뒤 수집 전에 반드시.
# 냄새 맡기 구간(task.yaml episode.sniff_decisions)에 몸이 정말 서 있나(먹이 거리 그대로) · 그 구간 더듬이 좌우 차 % · 캡션 쪽 칸.
# 09-26: 먹이 6~10 mm 에서 좌우 차 3.5~3.9 % → M1b 문턱(5 %) 아래라 뇌 결과 전에 4~7 mm 로 당김(abcb4ba).
# 사용: scripts/20-odor/m4-body-check.sh [재배선씨앗]
source "$(dirname "$0")/../common.sh"
no_lock
N="bodycheck-$(date +%H%M%S)"
step 40 collect --name "$N" --seeds 0-1 --rewired-seeds "${1:-0}" --workers 1
"$PY" - "runs/dataset-$N/samples.jsonl" <<'PY' | tee -a "$LOG"
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line); t = r["truth"]
    rel = abs(t["odor_left"] - t["odor_right"]) / max((t["odor_left"] + t["odor_right"]) / 2, 1e-9)
    print(f"씨앗 {r['seed']} 결정 {r['decision']} · 먹이 {t['food_distance_mm']:.1f} mm · 방위 {t['odor_azimuth_deg']:+.0f}° · "
          f"좌우 차 {100*rel:.1f} % · 멈춤 {r.get('action', {}).get('stop')} · {r['caption'].split('/')[1].strip()}")
PY
rm -rf "runs/dataset-$N"
say "확인: 냄새 맡기 결정들의 먹이 거리가 같아야 하고, 좌우 차가 M1b 문턱(LN4 5~10 %)에 닿는지"
