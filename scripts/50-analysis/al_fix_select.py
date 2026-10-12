"""al-fix-a — 뇌마다 흥분성 LN → PN 배율을 얹고 PN → KC 배율을 잠근 규칙으로 고른다(config/fit.yaml al_fix · 10-12).

고르기 = physiology 범위(select.scope) 항목 손실이 가장 작은 격자 값. 짝짓기 결과는 보지 않는다.
결과: runs/al-fix-a-<뇌>/params.npz + runs/al-fix-a-select-<시각>.json
사용: python scripts/50-analysis/al_fix_select.py real,rewired-s0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.circuit_fit import physiology_loss
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import measure_kc
from moai_echomind_fly.sim import LIFParams


def build(g, lif, base_dir: str, scope_groups: list, eln_pn: float) -> cp.CellParams:
    b = cp.CellParams.load(Path(base_dir) / "params.npz")
    cls, typ, sign = g.columns["class"].astype(str), g.columns["type"].astype(str), g.sign
    ln = cls == "ALLN"
    exc = sorted({t for t in typ[ln & (sign == 1)]} - {t for t in typ[ln & (sign != 1)]})
    groups = list(scope_groups) + [{"name": "eLN-PN", "src": {"type": exc},
                                    "dst": {"class": ["ALPN"]}}]
    c = cp.init(g, list(b.types), lif.t_mbr_ms, lif.v_th_mV, None,
                {"scope": "al-fix-a", "base": base_dir}, groups=groups)
    c.tau_ms[:], c.v_th_mV[:] = b.tau_ms, b.v_th_mV
    c.alpha[: len(b.alpha)] = b.alpha
    c.alpha[-1] = eln_pn
    return c


def main(brains: str = "real,rewired-s0") -> None:
    fit, phys = config.load("fit"), config.load("physiology")
    af = fit["al_fix"]
    sc = fit["scopes"][af["select"]["scope"]]
    cfg = {**fit["kc_probe"], **(sc.get("kc_probe") or {})}
    lo, hi = fit["bounds"]["alpha"]
    out = {"rule": af, "physiology": phys["version"], "brains": {}}
    with heavy("al-fix-select") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        g0 = graph_mod.load()
        lif = LIFParams.from_config()
        for brain in brains.split(","):
            g = g0 if brain == "real" else graph_mod.rewire(g0, int(brain.rsplit("-s", 1)[1]))
            c = build(g, lif, af["base"][brain], sc["groups"], float(af["eln_pn"]))
            k = list(c.meta["groups"]).index("ALPN-KC")
            a0 = float(c.alpha[k])
            rows = []
            for x in af["pn_kc_grid"]:
                c.alpha[k] = min(max(a0 * float(x), lo), hi)
                m = measure_kc(g, lif, c, int(af["select"]["trials"]), int(af["select"]["seed"]),
                               "cpu", cfg)
                loss, parts = physiology_loss(m, af["select"]["scope"], phys,
                                              float(fit["cmaes"]["nan_tolerances"]))
                rows.append({"pn_kc_x": x, "alpha": float(c.alpha[k]), "loss": round(loss, 4),
                             "parts": parts, "measures": m})
                print(f"[al-fix] {brain} PN→KC ×{x} 손실 {loss:.3f} · KC {m['kc_active_fraction']:.3f}"
                      f" · 냄새 넷 {m['kc_fraction_min_odor']:.3f}~{m['kc_fraction_max_odor']:.3f} · "
                      f"MBON {m['mbon_fire_fraction']:.3f}", flush=True)
            best = min(rows, key=lambda r: r["loss"])
            c.alpha[k] = best["alpha"]
            c.meta = {**c.meta, "pn_kc_x": best["pn_kc_x"], "loss": best["loss"]}
            d = paths.runs_dir() / f"al-fix-a-{brain}"
            d.mkdir(exist_ok=True)
            c.save(d / "params.npz")
            out["brains"][brain] = {"grid": rows, "pick": best["pn_kc_x"], "folder": str(d)}
            print(f"[al-fix] {brain} 고름 PN→KC ×{best['pn_kc_x']} · 손실 {best['loss']} · {d}",
                  flush=True)
    dst = paths.runs_dir() / f"al-fix-a-select-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[al-fix] {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
