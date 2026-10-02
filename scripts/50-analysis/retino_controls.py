"""시야 대응표 대조(R-7469 Task 7 · C) — 진짜 대응표 묶음 대 뒤섞은 대응표 묶음들의 선형 검사.

같은 원본 · 같은 결정 순서로 재생한 묶음들(진짜 뇌만)에서 시각 채널로 캡션 칸(물체 쪽 등)을 읽고,
진짜 대응표가 뒤섞은 대응표들보다 나은지 에피소드 단위 부호 뒤집기로 잰다(stats · judge_cell).
"시야 자리가 쓰였다"는 물체 쪽(loom_side)에서 진짜 > 섞은 대응표 전부 · p < alpha 일 때만.
사용: uv run python scripts/50-analysis/retino_controls.py <진짜 묶음> <대조 묶음> [대조 묶음 …]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from moai_echomind_fly import datasets
from moai_echomind_fly.loop.linear_probe import ProbeSet
from moai_echomind_fly.loop.stats import finalize, judge_cell, rule_for_dataset
from moai_echomind_fly.loop.train_cli import _sub, caption_fields


def probe(pack: str, channel: str) -> tuple[dict, list]:
    ftr, rtr = datasets.load(pack, "train", "real")
    fte, rte = datasets.load(pack, "eval", "real")
    ytr = [_sub(caption_fields(r["caption"])) for r in rtr]
    yte = [_sub(caption_fields(r["caption"])) for r in rte]
    ps = ProbeSet(ftr[channel].astype(np.float64), fte[channel].astype(np.float64))
    res = {f: ps.field([y[f] for y in ytr], [y[f] for y in yte]) for f in yte[0]}
    return res, [(r["seed"], r["decision"]) for r in rte]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("real")
    ap.add_argument("controls", nargs="+")
    ap.add_argument("--channel", default="vision")
    a = ap.parse_args()
    real, keys = probe(a.real, a.channel)
    ctrl = []
    for c in a.controls:
        r, k = probe(c, a.channel)
        if k != keys:
            raise SystemExit(f"{c}: 평가 결정 순서가 {a.real} 와 다르다 — 같은 원본 재생이 아니다")
        ctrl.append(r)
    rule = rule_for_dataset(a.real)
    episodes = np.array([s for s, _ in keys])
    cells = {}
    for f, r in real.items():
        if r["classes"] == 1:
            continue
        cells[f] = judge_cell(r["correct"], [c[f]["correct"] for c in ctrl],
                              r["baseline_correct"], episodes, rule)
        cells[f]["judge"] = f == "loom_side"           # 판정은 물체 쪽 하나(결과 전에 정함)
    finalize(cells, rule)
    print(f"채널 {a.channel} · 진짜 대응표 {a.real} · 대조 {', '.join(a.controls)} · {rule['version']}")
    for f, c in cells.items():
        ctl = " / ".join(f"{v:.3f}" for v in c["rewired"]["values"])
        print(f"  {f:14} 진짜 {c['real']:.3f} · 대조 {ctl} · 찍기 {c['baseline']:.3f} · "
              f"p {c.get('p_block_perm', float('nan')):.4f} · "
              f"{'판정 ' + ('통과' if c.get('pass') else '못 넘음') if c['judge'] else '(참고)'}")
    print(json.dumps({"loom_side_pass": cells.get("loom_side", {}).get("pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
