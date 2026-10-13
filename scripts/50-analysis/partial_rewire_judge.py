"""부분 섞기 정식 판정(config/plasticity.yaml partial_rewire_test · 결과 전 잠금 · 10-13).

뇌마다 plasticity-controls 결과(runs/plasticity-controls-<뇌>-*.json · 가장 최근 · 짝짓기 · 씨앗 10)를 읽어
씨앗마다 SI 를 내고 갈래마다 진짜와 섞은 셋을 견준다. 사용: python scripts/50-analysis/partial_rewire_judge.py
→ runs/partial-rewire-judge-<시각>.json
"""

from __future__ import annotations

import glob
import itertools
import json
import time

import numpy as np

from moai_echomind_fly import config, paths


def load(brain: str, seeds: list[int]) -> dict:
    f = sorted(glob.glob(str(paths.runs_dir() / f"plasticity-controls-{brain}-*.json")))[-1]
    d = json.loads(open(f, encoding="utf-8").read())
    rows = {r["seed"]: r for r in d["rows"] if r["cond"] == "paired"}
    assert sorted(rows) == seeds, f"{brain}: 씨앗 {sorted(rows)}"
    si, plus_before = [], 0
    for s in seeds:
        (pb, pa), (mb, ma) = rows[s]["cs_plus.reward"], rows[s]["cs_minus.reward"]
        plus_before += pb
        si.append((1 - pa / pb) - (1 - ma / mb) if pb and mb else float("nan"))
    return {"file": f, "eta": rows[seeds[0]]["eta"], "si": si, "si_mean": float(np.nanmean(si)),
            "cs_plus_before_sum": int(plus_before)}


def exact_sign_flip(d: np.ndarray) -> float:
    obs = abs(d.mean())
    n = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        n += abs((d * np.array(signs)).mean()) >= obs - 1e-12
    return n / 2 ** len(d)


def main() -> None:
    cfg = config.load("plasticity")["partial_rewire_test"]
    seeds = [int(x) for x in cfg["seeds"]]
    real = load("real", seeds)
    out = {"rule": cfg, "real": real, "arms": {}}
    print(f"[judge] 진짜 SI 평균 {real['si_mean']:.3f} · 씨앗 {np.round(real['si'], 3).tolist()}")
    for arm, brains in cfg["arms"].items():
        ctl = {b: load(b, seeds) for b in brains}
        measurable = all(c["cs_plus_before_sum"] >= cfg["pass"]["min_cs_plus_spikes"]
                         for c in ctl.values())
        ens = np.nanmean(np.array([c["si"] for c in ctl.values()]), axis=0)
        d = np.array(real["si"]) - ens
        p = exact_sign_flip(d)
        above = all(real["si_mean"] > c["si_mean"] for c in ctl.values())
        ok = (bool(measurable and above and p < cfg["pass"]["sign_flip_p"])
              if measurable else None)
        out["arms"][arm] = {"controls": ctl, "d": d.tolist(), "p": p, "above_all": above,
                            "measurable": measurable, "pass": ok}
        print(f"[judge] {arm}: 섞은 셋 SI {[round(c['si_mean'], 3) for c in ctl.values()]} · "
              f"진짜가 모두 위 {above} · p {p:.4f} · 잴 수 있음 {measurable} · 통과 {ok}",
              flush=True)
    dst = paths.runs_dir() / f"partial-rewire-judge-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[judge] {dst}")


if __name__ == "__main__":
    main()
