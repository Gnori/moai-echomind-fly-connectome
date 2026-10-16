"""눈 → 뇌 길 재기(10-16 · 가벼움 · 판정 아님) — 요구자 "초파리가 정말 볼 수 있게 · 3차원 공간".

왜: 광수용체(R1-R8)를 켜면 뇌 전체 발화 0(부호 −1 · 쉬는 상태 침묵). 그렇다면 시각엽의 어느 층부터 넣어야
우리 배선(진짜)이 LC/LPLC · 소엽판(HS · VS) · 하강 뉴런(DNa02 · GF)까지 신호를 나르는지, 그리고 그것이
섞은 배선(전체 섞기 s0~s2)에서도 같은지를 잰다. 한쪽 눈만 켜서 좌우 조향 뉴런이 갈리는지도 본다.
자극(계산 전에 정함): 두 눈 L2 · L3 · Tm2 · T4a · T5a / 왼눈만 T4a · T5a · T4b · T5b — 모두 150 Hz · 300 ms · 씨앗 0.
⚠️ 시각엽 전체를 한꺼번에 켜는 것은 자연 그림이 아니다 — "길이 이어져 있나"만 본다.
사용: python scripts/50-analysis/vision_path.py → runs/vision-path-<시각>.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.sim import Engine, LIFParams

BRAINS = ["real", "rewired-s0", "rewired-s1", "rewired-s2"]
STIMULI = [("L2", None), ("L3", None), ("Tm2", None), ("T4a", None), ("T5a", None),
           ("T4a", "L"), ("T5a", "L"), ("T4b", "L"), ("T5b", "L")]
RATE_HZ, WINDOW_MS = 150.0, 300.0


def readout(g, cnt: np.ndarray, typ: np.ndarray, side: np.ndarray, lc_types: list[str]) -> dict:
    def n_sp(mask: np.ndarray) -> tuple[int, int]:
        return int((cnt[mask] > 0).sum()), int(cnt[mask].sum())

    out: dict = {"spiking": int((cnt > 0).sum())}
    lc = np.isin(typ, lc_types)
    out["lc_lplc"] = dict(zip(("neurons", "spikes"), n_sp(lc)))
    out["lc_by_type"] = {t: n_sp(typ == t)[0] for t in lc_types if n_sp(typ == t)[0] > 0}
    for name, types in (("HS", ["HSE", "HSN", "HSS"]), ("VS", ["VS"]), ("DNa02", ["DNa02"]),
                        ("GF", ["DNp01"]), ("DNp09", ["DNp09"])):
        m = np.isin(typ, types)
        out[name] = {s: n_sp(m & (side == s))[1] for s in ("L", "R")}
    dn = g.columns["superclass"].astype(str) == "descending_neuron"
    out["descending"] = dict(zip(("neurons", "spikes"), n_sp(dn)))
    return out


def main() -> None:
    lc_types = list(config.load("loom")["amplitude"].keys())
    res = {"rule": {"brains": BRAINS, "stimuli": STIMULI, "rate_hz": RATE_HZ, "window_ms": WINDOW_MS,
                    "lc_types": lc_types}, "runs": []}
    with heavy("vision-path") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = base.columns["somaSide"].astype(str)
        for brain in BRAINS:
            g = graph_mod.brain_graph(base, brain)
            for t, s in STIMULI:
                m = typ == t if s is None else (typ == t) & (side == s)
                idx = np.flatnonzero(m)
                eng = Engine(g, lif, device="cpu", seed=0)
                eng.set_poisson(idx, np.full(len(idx), RATE_HZ))
                eng.run(WINDOW_MS)
                _, neurons = eng.spikes()
                cnt = np.bincount(neurons, minlength=g.n)
                cnt[idx] = 0                                 # 자극한 뉴런은 셈에서 뺌
                r = {"brain": brain, "stim": t, "side": s or "both", "n_stim": int(len(idx)),
                     **readout(g, cnt, typ, side, lc_types)}
                res["runs"].append(r)
                print(f"[vision] {brain:11} {t:4} {s or '두눈':3} · 퍼진 {r['spiking']:6} · "
                      f"LC/LPLC {r['lc_lplc']['neurons']:4} · HS L/R {r['HS']['L']}/{r['HS']['R']} · "
                      f"DNa02 L/R {r['DNa02']['L']}/{r['DNa02']['R']} · GF {r['GF']['L']}/{r['GF']['R']} · "
                      f"하강 {r['descending']['neurons']}", flush=True)
    dst = paths.runs_dir() / f"vision-path-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vision] {dst}")


if __name__ == "__main__":
    main()
