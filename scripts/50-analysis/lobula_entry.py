"""소엽 기둥 뉴런 입구 재기(10-16 · 가벼움 · 판정 아님) — V0b(다가옴) 설계 재료.

왜: V0 시간 재기에서 T4/T5 만으로는 다가옴에 LPLC2 가 거의 안 켜졌다. LPLC2 입력의 2/3 · LC4 입력의 거의 전부는
T4/T5 밖 소엽 기둥 뉴런(runs/loom-inputs-20261016-205710.json). 그 종류를 켜면 진짜 배선에서 LC4 · LPLC2 · GF 까지
가는지, 섞은 배선(전체 섞기 s0~s2)에서도 같은지 본다.
자극(계산 전에 정함 · 150 Hz · 300 ms · 씨앗 0): 종류마다 ① 두 눈 전체 ② 왼쪽 앞 30° 둘레 20° 안 기둥만(국소) ·
섞음 둘 ③ LC4 입력 넷(T2 · TmY3 · Tm4 · Tm2) 국소 ④ LPLC2 입력(T4a-d · T5a-d · Tm5Y · Tm20) 국소.
읽는 것: LC4 · LPLC2 · GF(DNp01) · DNp02 좌우 · 하강 뉴런 수.
사용: python scripts/50-analysis/lobula_entry.py → runs/lobula-entry-<시각>.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly import paths
from moai_echomind_fly.eye.geometry import type_positions
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams

BRAINS = ["real", "rewired-s0", "rewired-s1", "rewired-s2"]
SINGLE = ["Tm5Y", "Tm20", "Y3", "TmY5a", "Tm37", "Tm4", "T2", "TmY3", "Tm2", "Tm3"]
MIXES = {"LC4mix": ["T2", "TmY3", "Tm4", "Tm2"],
         "LPLC2mix": [f"T{k}{d}" for k in (4, 5) for d in "abcd"] + ["Tm5Y", "Tm20"]}
CENTER, RADIUS = (30.0, 0.0), 20.0
RATE_HZ, WINDOW_MS = 150.0, 300.0
READ = {"LC4": ["LC4"], "LPLC2": ["LPLC2"], "GF": ["DNp01"], "DNp02": ["DNp02"]}


def local(pos) -> np.ndarray:
    a, e = np.radians(pos.az), np.radians(pos.el)
    ca, ce = np.radians(CENTER[0]), np.radians(CENTER[1])
    cosd = np.sin(e) * np.sin(ce) + np.cos(e) * np.cos(ce) * np.cos(a - ca)
    return pos.idx[np.degrees(np.arccos(np.clip(cosd, -1, 1))) <= RADIUS]


def main() -> None:
    res: dict = {"rule": {"brains": BRAINS, "single": SINGLE, "mixes": MIXES, "center": CENTER,
                          "radius_deg": RADIUS, "rate_hz": RATE_HZ, "window_ms": WINDOW_MS},
                 "runs": []}
    with heavy("lobula-entry") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base)
        groups = {f"{k}_{s}": np.flatnonzero(np.isin(typ, v) & (side == s))
                  for k, v in READ.items() for s in ("L", "R")}
        dn = base.columns["superclass"].astype(str) == "descending_neuron"
        stims: list[tuple[str, np.ndarray]] = []
        for t in SINGLE:
            stims.append((f"{t}-both", np.flatnonzero(typ == t)))
            stims.append((f"{t}-local", local(type_positions(base, [t]))))
        for name, ts in MIXES.items():
            stims.append((f"{name}-local", local(type_positions(base, ts))))
        for brain in BRAINS:
            g = graph_mod.brain_graph(base, brain)
            for name, idx in stims:
                eng = Engine(g, lif, device="cpu", seed=0)
                eng.set_poisson(idx, np.full(len(idx), RATE_HZ))
                eng.run(WINDOW_MS)
                _, neurons = eng.spikes()
                cnt = np.bincount(neurons, minlength=g.n)
                cnt[idx] = 0
                r = {"brain": brain, "stim": name, "n_stim": int(len(idx)),
                     "spiking": int((cnt > 0).sum()), "descending": int((cnt[dn] > 0).sum()),
                     **{k: int(cnt[ix].sum()) for k, ix in groups.items()}}
                res["runs"].append(r)
                print(f"[lob] {brain:11} {name:15} 켬 {r['n_stim']:5} · 퍼짐 {r['spiking']:6}"
                      f" · LC4 {r['LC4_L']}/{r['LC4_R']} · LPLC2 {r['LPLC2_L']}/{r['LPLC2_R']}"
                      f" · GF {r['GF_L']}/{r['GF_R']} · DNp02 {r['DNp02_L']}/{r['DNp02_R']}"
                      f" · 하강 {r['descending']}", flush=True)
    dst = paths.runs_dir() / f"lobula-entry-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lob] {dst}")


if __name__ == "__main__":
    main()
