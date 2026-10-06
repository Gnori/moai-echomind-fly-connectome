"""KC 를 켠 채 냄새끼리 겹침이 낮은 해가 있나 — kc-mbon-b 매개변수에서 묶음 배율 몇 개만 바꿔 본다(10-06 · 가능성 먼저).

kc-mbon-c 회로 학습이 KC 를 끈 해로 몰려 중단(958eef7). 다시 학습하기 전에 ① 아는 해 · 끈 해의 손실(physiology-v7 ·
범위 kc-mbon-c)을 항별로 계산하고 ② 겹침을 낮출 만한 손잡이(KC→KC ↓ · APL→KC ↑ · PN→KC ↑)를 격자로 바꿔
"KC ≥ 3 % 이면서 겹침 ≤ 0.12" 인 점이 있는지 잰다. 씨앗 0 · 300 ms · 학습 냄새 넷(학습 손실과 같은 측정 · trials 1).
사용: python scripts/50-analysis/kc_overlap_grid.py   → runs/kc-overlap-grid-<시각>.json
"""

from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.cellparams import CellParams
from moai_echomind_fly.circuit_fit import physiology_loss
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import measure_kc
from moai_echomind_fly.sim import LIFParams

B = "runs/fit-kc-mbon-b-sim-v3-real-20261006-071233"
KNOWN = [("real", "none"), ("real", "runs/fit-kc-sparse-c-sim-v3-real-20261005-011859"),
         ("real", "runs/fit-kc-mbon-a-sim-v3-real-20261005-114119"), ("real", B),
         ("rewired-s0", "runs/fit-kc-mbon-b-sim-v3-rewired-s0-20261006-105313")]
GRID = {"KC-KC": [1.0, 0.3, 0.1], "APL-KC": [1.0, 2.5, 6.0], "ALPN-KC": [1.0, 1.5]}  # b 값에 곱함


def main() -> None:
    fit, phys = config.load("fit"), config.load("physiology")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-c"]["kc_probe"]}
    nan_tol = float(fit["cmaes"]["nan_tolerances"])
    lif = LIFParams.from_config()
    rows = []

    def one(g, cell, label: dict) -> None:
        m = measure_kc(g, lif, cell, 1, 0, "cpu", cfg)
        tot, parts = physiology_loss(m, "kc-mbon-c", phys, nan_tol)
        r = {**label, "loss_v7": round(tot, 3),
             "loss_without_overlap": round(tot - parts.get("kc_odors_decorrelated", 0.0), 3),
             "parts": parts, **{k: m[k] for k in ("kc_active_fraction", "kc_fraction_min_odor",
                                                 "kc_fraction_max_odor", "kc_overlap_max_odor",
                                                 "mbon_fire_fraction")}}
        rows.append(r)
        print(f"[grid] {label} 손실 {r['loss_v7']} (겹침 빼면 {r['loss_without_overlap']}) · "
              f"KC {r['kc_fraction_min_odor']:.3f}~{r['kc_fraction_max_odor']:.3f} · "
              f"겹침 {r['kc_overlap_max_odor']:.3f} · MBON {r['mbon_fire_fraction']:.3f}",
              flush=True)

    with heavy("kc-overlap-grid") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        rew = graph_mod.rewire(base, 0)
        for brain, folder in KNOWN:
            cell = None if folder == "none" else CellParams.load(Path(folder) / "params.npz")
            one(base if brain == "real" else rew, cell, {"brain": brain, "cell_from": folder})
        b = CellParams.load(Path(B) / "params.npz")
        groups = list(b.meta["groups"])
        for kk, ak, pk in itertools.product(*GRID.values()):
            if (kk, ak, pk) == (1.0, 1.0, 1.0):
                continue
            c = copy.deepcopy(b)
            for name, f in zip(GRID, (kk, ak, pk)):
                c.alpha[groups.index(name)] = min(max(c.alpha[groups.index(name)] * f, 0.1), 10.0)
            one(base, c, {"brain": "real", "cell_from": B,
                          "x": {"KC-KC": kk, "APL-KC": ak, "ALPN-KC": pk}})
    out = paths.runs_dir() / f"kc-overlap-grid-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"grid": GRID, "rows": rows}, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"[grid] {out}")


if __name__ == "__main__":
    main()
