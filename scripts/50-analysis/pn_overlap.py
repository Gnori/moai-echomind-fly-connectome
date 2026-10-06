"""KC 겹침이 PN 단계에서 이미 생기나 — 냄새마다 켜진 PN 집합의 겹침(10-06 · 냄새 특이성 마무리 · 가벼움).

학습 냄새 넷(식초 + 무작위 3 · fit.yaml kc-mbon-b) + 가소성 대조의 CS−. 씨앗 0 · 300 ms · ORN 150 Hz.
해부 겹침 = 자극 ORN 에서 입력 ≥ 10 % 받는 PN(mb_probe.stim_pns) · 발화 겹침 = 창에서 한 번 이상 발화한 PN(ALPN).
사용: python scripts/50-analysis/pn_overlap.py <뇌>:<회로 학습 폴더|none> ...  → runs/pn-overlap-<시각>.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.cellparams import CellParams
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import extra_odor_sets, run, stim_pns
from moai_echomind_fly.sim import Engine, LIFParams

CS_MINUS = ["ORN_DL5", "ORN_DM5", "ORN_DM6", "ORN_VA1v", "ORN_VA3", "ORN_VM6m"]


def jac(a: np.ndarray, b: np.ndarray) -> float:
    u = len(np.union1d(a, b))
    return round(len(np.intersect1d(a, b)) / u, 4) if u else 0.0


def main(specs: list[str]) -> None:
    fit = config.load("fit")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-b"]["kc_probe"]}
    lif = LIFParams.from_config()
    rows = []
    with heavy("pn-overlap") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        odors = {"vinegar": cfg["stimuli"][cfg["target_stimulus"]]}
        odors.update({f"extra{i}": s for i, s in enumerate(extra_odor_sets(base, cfg))})
        odors["cs_minus"] = CS_MINUS
        for spec in specs:
            brain, folder = spec.split(":", 1)
            g = base if brain == "real" else graph_mod.rewire(base, int(brain.rsplit("-s", 1)[1]))
            cell = None if folder == "none" else CellParams.load(Path(folder) / "params.npz")
            pn = np.flatnonzero(g.columns["class"] == cfg["pn_class"])
            anat, fired, kc = {}, {}, {}
            for name, st in odors.items():
                orn = g.select(type=st)
                anat[name] = stim_pns(g, orn, cfg)
                eng = Engine(g, lif, device="cpu", seed=0, cell=cell)
                eng.set_poisson(orn, np.full(len(orn), float(cfg["orn_rate_hz"])))
                eng.run(float(cfg["loss_window_ms"]))
                _, n = eng.spikes()
                fired[name] = np.intersect1d(pn, n)
                kc[name] = run(g, lif, cfg, st, 0, "cpu", cell, [float(cfg["loss_window_ms"])],
                               keep_kc=True)["kc_on"]
            pairs = {}
            for a, b in itertools.combinations(odors, 2):
                pairs[f"{a}|{b}"] = {"pn_anat": jac(anat[a], anat[b]),
                                     "pn_fired": jac(fired[a], fired[b]),
                                     "kc": jac(kc[a], kc[b])}
            r = {"brain": brain, "cell_from": folder, "pn_total": int(len(pn)),
                 "n_anat": {k: int(len(v)) for k, v in anat.items()},
                 "n_fired": {k: int(len(v)) for k, v in fired.items()},
                 "n_kc": {k: int(len(v)) for k, v in kc.items()}, "pairs": pairs}
            rows.append(r)
            for k in ("pn_anat", "pn_fired", "kc"):
                v = [p[k] for p in pairs.values()]
                print(f"[pn-overlap] {brain} {Path(folder).name} {k} 평균 {np.mean(v):.3f} "
                      f"최대 {max(v):.3f}", flush=True)
            print(f"[pn-overlap]   켜진 PN {r['n_fired']} · 해부 PN {r['n_anat']}", flush=True)
    out = paths.runs_dir() / f"pn-overlap-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"odors": odors, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[pn-overlap] {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
