"""촉각 ② 가능성 작은 시험(R-1177 F-011 · 요구자 10-18 "가능성 작은 시험 먼저" · 판정 아님).

①(touch_entry) 에서 연결 몫이 섞은 배선보다 뚜렷했던 무리(앞다리 → 같은 쪽 DNa02 · 머리 털 → MDN)를
실제로 LIF 에서 켜 본다: 무리 하나(한쪽)를 포아송 r Hz 로 window_ms 동안 → 몸 하강 뉴런 좌우 발화 ·
하강 뉴런 전체 · 발화가 많은 하강 뉴런 종류 · 진짜 대 섞은 배선 셋. 다음으로 갈지는 config next_rule(결과 전 잠금).
값은 config/eyeworld.yaml touch_probe · 무리 고르기는 touch_entry 와 같다.
사용: python scripts/50-analysis/touch_probe.py → runs/touch-probe-<시각>.json
"""

from __future__ import annotations

import collections
import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.senses.touch import groups
from moai_echomind_fly.sim import Engine, LIFParams


def main() -> None:
    ecfg = config.load("eyeworld")
    cfg, ecfg_t = ecfg["touch_probe"], ecfg["touch_entry"]
    res: dict = {"rule": cfg, "runs": []}
    t0 = time.time()
    with heavy("touch-probe") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base).astype(str)
        allg = groups(base, ecfg_t)
        stims = {k: v for k, v in allg.items() if k.rsplit("_", 1)[0] in cfg["groups"]}
        read = {f"{t}_{s}": np.flatnonzero((typ == t) & (side == s))
                for t in ecfg_t["targets"] for s in "LR"}
        dn = np.flatnonzero(base.columns["superclass"].astype(str) == "descending_neuron")
        for brain in cfg["brains"]:
            g = graph_mod.brain_graph(base, brain)
            for name, idx in stims.items():
                for rate in cfg["rates_hz"]:
                    eng = Engine(g, lif, device="cpu", seed=int(cfg["seed"]))
                    eng.set_poisson(idx, np.full(len(idx), float(rate)))
                    eng.run(float(cfg["window_ms"]))
                    _, neurons = eng.spikes()
                    cnt = np.bincount(neurons, minlength=g.n)
                    cnt[idx] = 0
                    by = collections.Counter()
                    for t, v in zip(typ[dn], cnt[dn]):
                        if v:
                            by[str(t)] += int(v)
                    r = {"brain": brain, "stim": name, "rate_hz": rate, "n_stim": int(len(idx)),
                         "spiking": int((cnt > 0).sum()), "dn_spiking": int((cnt[dn] > 0).sum()),
                         "dn_spikes": int(cnt[dn].sum()),
                         **{k: int(cnt[ix].sum()) for k, ix in read.items()},
                         "top_dn": by.most_common(int(cfg["top_dn"]))}
                    res["runs"].append(r)
                    print(f"[touch] {brain:11} {name:7} {rate:5.0f} Hz · 켬 {r['n_stim']:4}"
                          f" · 퍼짐 {r['spiking']:6} · 하강 {r['dn_spiking']:4}/{r['dn_spikes']:5}"
                          f" · DNa02 {r['DNa02_L']}/{r['DNa02_R']} · DNa01 {r['DNa01_L']}/{r['DNa01_R']}"
                          f" · MDN {r['MDN_L']}/{r['MDN_R']} · DNp09 {r['DNp09_L']}/{r['DNp09_R']}"
                          f" · GF {r['DNp01_L']}/{r['DNp01_R']} · 위 {r['top_dn'][:3]}"
                          f" · {time.time() - t0:.0f} s", flush=True)
    dst = paths.runs_dir() / f"touch-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[touch] {dst}")


if __name__ == "__main__":
    main()
