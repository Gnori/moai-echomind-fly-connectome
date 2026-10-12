"""촉각엽 고침이 KC 겹침까지 푸나(10-12 · 가능성 먼저 · 판정 아님).

kc-mbon-b(진짜) 매개변수(KC · APL · MBON 묶음) 위에 흥분성 LN → PN 묶음 배율을 더한 매개변수를 만들어
runs/al-kc-<배율>/params.npz 로 저장하고, 학습 냄새 넷의 KC 켜짐 · 냄새끼리 KC Jaccard · MBON 반응을 잰다
(measure_kc · odor_overlap). 저장한 폴더는 plasticity-controls --cell-from 으로 짝짓기 대조에 그대로 쓴다.
사용: python scripts/50-analysis/al_kc_probe.py 1.0,0.5,0.25 → runs/al-kc-probe-<시각>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import measure_kc
from moai_echomind_fly.sim import LIFParams

B = "runs/fit-kc-mbon-b-sim-v3-real-20261006-071233"


def build(g, lif, factor: float) -> cp.CellParams:
    fit = config.load("fit")
    sc = fit["scopes"]["kc-mbon-b"]
    b = cp.CellParams.load(Path(B) / "params.npz")
    cls, typ, sign = g.columns["class"].astype(str), g.columns["type"].astype(str), g.sign
    ln = cls == "ALLN"
    exc = sorted({t for t in typ[ln & (sign == 1)]} - {t for t in typ[ln & (sign != 1)]})
    groups = list(sc["groups"]) + [{"name": "eLN-PN", "src": {"type": exc},
                                    "dst": {"class": ["ALPN"]}}]
    c = cp.init(g, list(b.types), lif.t_mbr_ms, lif.v_th_mV, None,
                {"scope": "kc-mbon-b+al", "base": B, "eLN_PN": factor}, groups=groups)
    c.tau_ms[:], c.v_th_mV[:] = b.tau_ms, b.v_th_mV
    c.alpha[: len(b.alpha)] = b.alpha
    c.alpha[-1] = factor
    return c


def main(factors: str = "1.0,0.5,0.25") -> None:
    fit = config.load("fit")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-b"]["kc_probe"], "odor_overlap": True}
    rows = []
    with heavy("al-kc-probe") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        g = graph_mod.load()
        lif = LIFParams.from_config()
        for f in [float(x) for x in factors.split(",")]:
            c = build(g, lif, f)
            d = paths.runs_dir() / f"al-kc-x{f:g}"
            d.mkdir(exist_ok=True)
            c.save(d / "params.npz")
            m = measure_kc(g, lif, c, 1, 0, "cpu", cfg)
            rows.append({"eLN_PN": f, "cell_from": str(d), **m})
            print(f"[al-kc] eLN→PN ×{f:g} · KC 식초 {m['kc_active_fraction']:.3f} · 냄새 넷 "
                  f"{m['kc_fraction_min_odor']:.3f}~{m['kc_fraction_max_odor']:.3f} · KC 겹침 최대 "
                  f"{m['kc_overlap_max_odor']:.3f} · MBON {m['mbon_fire_fraction']:.3f} · "
                  f"MBON 최소 냄새 {m['mbon_fire_min_odor']:.3f} · 폴더 {d}", flush=True)
    dst = paths.runs_dir() / f"al-kc-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps({"base": B, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[al-kc] {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
