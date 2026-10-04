"""물체 인식 판정(R-5138 S0b · 결과 전 잠금) — 시각 채널 신호로 참값 다섯을 읽을 수 있나.

npc-perceive 결과(features-<뇌>.npz · samples.jsonl)를 에피소드(씨앗) 단위로 학습 70 % · 평가 30 %
로 나누고, 뇌마다 같은 방법(ProbeSet · 능선 회귀)으로 칸마다 잰 뒤 judge_cell · finalize(stats-v3 ·
Holm 은 시각 채널 다섯 칸끼리). 통과 = 진짜 > 섞은 배선 셋 전부 · p(Holm) < 0.05 · 찍기보다 높음.
다른 채널(촉각 · 전부 등)은 참고로만 적는다(판정 아님).
사용: uv run python scripts/50-analysis/npc_perception_probe.py runs/npc-perceive-<시각>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moai_echomind_fly.loop.linear_probe import ProbeSet
from moai_echomind_fly.loop.stats import STATS_RULE_V3, finalize, judge_cell

FIELDS = ("present", "side", "close", "approaching", "person")
JUDGED_CHANNEL = "vision"
TRAIN_FRAC = 0.7
SPLIT_SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--channels", default="vision,touch,all", help="잴 채널(판정은 vision 만)")
    a = ap.parse_args()
    req = json.loads((a.run / "request.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (a.run / "samples.jsonl").read_text("utf-8").splitlines()]
    seeds = np.array([r["seed"] for r in rows])
    uniq = np.unique(seeds)
    perm = np.random.default_rng(SPLIT_SEED).permutation(uniq)
    tr_seeds = set(perm[: int(round(TRAIN_FRAC * len(uniq)))].tolist())
    tr = np.array([s in tr_seeds for s in seeds])
    te = ~tr
    y = {f: [r["labels"][f] for r in rows] for f in FIELDS}
    brains = req["brains"]
    raw: dict[str, dict[str, dict]] = {}
    out = {"run": str(a.run), "rule": "R-5138 S0b · stats-v3", "brains": brains,
           "episodes": {"train": len(tr_seeds), "eval": int(len(uniq) - len(tr_seeds))},
           "decisions": {"train": int(tr.sum()), "eval": int(te.sum())},
           "label_counts": {f: {k: int(v) for k, v in zip(*np.unique(np.array(y[f])[te],
                                                                       return_counts=True))}
                            for f in FIELDS},
           "accuracy": {}}
    chans = a.channels.split(",")
    for b in brains:
        z = np.load(a.run / f"features-{b}.npz")
        n = len(z[z.files[0]])
        assert n == len(rows), f"{b}: 특징 {n} ≠ 결정 {len(rows)}"
        feats = {c: z[c].astype(np.float64) for c in z.files}
        feats["all"] = np.concatenate([feats[c] for c in z.files], axis=1)
        raw[b] = {}
        for c in chans:
            ps = ProbeSet(feats[c][tr], feats[c][te])
            for f in FIELDS:
                ytr = [v for v, k in zip(y[f], tr) if k]
                yte = [v for v, k in zip(y[f], te) if k]
                raw[b][f"{c}|{f}"] = ps.field(ytr, yte)
        out["accuracy"][b] = {k: round(r["accuracy"], 4) for k, r in raw[b].items()}
    rule = STATS_RULE_V3
    cells = {}
    for key, r in raw["real"].items():
        if r["classes"] == 1:
            continue
        cell = judge_cell(r["correct"], [raw[b][key]["correct"] for b in brains[1:]],
                          r["baseline_correct"], seeds[te], rule)
        cell["judge"] = key.startswith(JUDGED_CHANNEL + "|")
        cells[key] = cell
    finalize(cells, rule)
    out["cells"] = cells
    out["passed"] = sorted(k for k, c in cells.items() if c.get("pass"))
    (a.run / "perception-probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(f"평가 에피소드 {out['episodes']['eval']} · 결정 {out['decisions']['eval']}")
    for key, c in cells.items():
        mark = {True: "통과", False: "못 넘음", None: "참고"}[c.get("pass")]
        print(f"{key:24s} 진짜 {c['real']:.3f} · 섞은 {c['rewired']['values']} · 찍기 "
              f"{c['baseline']:.3f} · p(Holm) {c.get('p_holm')} · {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
