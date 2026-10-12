"""S0b 촉각 채널이 찍기보다 낮은 까닭(10-12 · 사후 분석 · 판정 아님).

짐작: 능선 회귀에 절편이 없어(목표 one-hot 을 중심화하지 않음) 정보가 거의 없는 작은 특징에서는 예측이 다수 답으로
돌아가지 못하고 학습 · 평가 나눔의 치우침을 따라 한 칸으로 쏠린다. 확인: ① 예측이 어느 답으로 쏠리나 ② 목표를 중심화
(절편과 같음)하면 찍기 근처로 돌아오나 ③ 특징 크기(진짜 · 섞은 배선). 시각 채널도 같은 식으로 다시 재 판정이 흔들리나 본다.
사용: uv run python scripts/50-analysis/touch_anomaly.py runs/npc-perceive-<시각>  → runs/touch-anomaly-<시각>.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from moai_echomind_fly import paths
from moai_echomind_fly.loop.linear_probe import ProbeSet, _onehot
from moai_echomind_fly.loop.stats import STATS_RULE_V3, finalize, judge_cell

FIELDS = ("present", "side", "close", "approaching", "person")


def main(run: str) -> None:
    d = Path(run)
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (d / "samples.jsonl").read_text("utf-8").splitlines()]
    seeds = np.array([r["seed"] for r in rows])
    uniq = np.unique(seeds)
    perm = np.random.default_rng(0).permutation(uniq)          # npc_perception_probe 와 같은 나눔
    trs = set(perm[: int(round(0.7 * len(uniq)))].tolist())
    tr = np.array([s in trs for s in seeds])
    te = ~tr
    out = {"run": run, "brains": {}}
    raw: dict[str, dict] = {}
    for b in req["brains"]:
        z = np.load(d / f"features-{b}.npz")
        res = {}
        for ch in ("touch", "vision"):
            x = z[ch].astype(np.float64)
            ps = ProbeSet(x[tr], x[te])
            if ch == "vision":                 # 사후 다시 판정용(절편 있는 판 · 같은 칸 · 같은 나눔)
                raw[b] = {f: ps.field([r["labels"][f] for r, m in zip(rows, tr) if m],
                                      [r["labels"][f] for r, m in zip(rows, te) if m],
                                      intercept=True) for f in FIELDS}
            res[ch] = {"abs_max": float(np.abs(x).max()), "nonzero_cols": int((x != 0).any(0).sum()),
                       "train_eval_mean_shift": float(np.linalg.norm(x[tr].mean(0) - x[te].mean(0))),
                       "fields": {}}
            for f in FIELDS:
                ytr = [r["labels"][f] for r, m in zip(rows, tr) if m]
                yte = [r["labels"][f] for r, m in zip(rows, te) if m]
                classes = sorted(set(ytr) | set(yte))
                r0 = ps.field(ytr, yte)
                y = _onehot(ytr, classes)
                prior = y.mean(0)
                lam = r0["lambda"]
                pred_c = (ps.full.predict(y - prior, lam) + prior).argmax(1)   # 절편 있는 판
                acc_c = float(np.mean([classes[j] == t for j, t in zip(pred_c, yte)]))
                pred0 = ps.full.predict(y, lam).argmax(1)
                res[ch]["fields"][f] = {
                    "acc": round(r0["accuracy"], 4), "chance": round(r0["baseline"], 4),
                    "acc_with_intercept": round(acc_c, 4), "lambda": lam,
                    "pred_share": {classes[k]: round(v / len(yte), 3)
                                   for k, v in Counter(pred0.tolist()).items()}}
            print(f"[touch] {b} {ch} 최대 {res[ch]['abs_max']:.4f} · " + " · ".join(
                f"{f} {v['acc']}→{v['acc_with_intercept']}(찍기 {v['chance']})"
                for f, v in res[ch]["fields"].items()), flush=True)
        out["brains"][b] = res
    brains = req["brains"]
    cells = {}
    for f in FIELDS:
        r = raw["real"][f]
        cells[f"vision|{f}"] = judge_cell(r["correct"], [raw[b][f]["correct"] for b in brains[1:]],
                                          r["baseline_correct"], seeds[te], STATS_RULE_V3)
    finalize(cells, STATS_RULE_V3)
    out["posthoc_vision_with_intercept"] = {
        k: {"real": round(c["real"], 4), "rewired": c["rewired"]["values"],
            "rank_pct": c["rank_pct"], "p": c["p_block_perm"], "p_holm": c["p_holm"],
            "pass_if_rule": c["pass"]} for k, c in cells.items()}
    for k, c in out["posthoc_vision_with_intercept"].items():
        print(f"[touch] 사후(절편) {k} 진짜 {c['real']} · 섞은 {c['rewired']} · p {c['p']:.4f} · "
              f"Holm {c['p_holm']:.4f} · 규칙대로면 {c['pass_if_rule']}", flush=True)
    dst = paths.runs_dir() / f"touch-anomaly-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[touch] {dst}")


if __name__ == "__main__":
    main(sys.argv[1])
