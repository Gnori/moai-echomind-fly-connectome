"""촉각엽 고침 뒤 KC 를 다시 켤 수 있나(10-12 · 가능성 먼저 · 판정 아님) — 다시 회로 학습하기 전.

al_kc_probe 의 매개변수(runs/al-kc-x<배율>) 위에서 PN → KC 묶음 배율만 올려, 학습 냄새 넷의 KC 켜짐이
3 % 이상으로 돌아오면서 냄새끼리 KC 겹침이 낮게(≤ 0.12 · physiology-v7 상한) 남는 점이 있는지 본다.
사용: python scripts/50-analysis/al_kc_gain.py 0.5,0.25 1.25,1.5,2.0 → runs/al-kc-gain-<시각>.json
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


def main(al: str = "0.5,0.25", gains: str = "1.25,1.5,2.0") -> None:
    fit = config.load("fit")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-b"]["kc_probe"], "odor_overlap": True}
    rows = []
    with heavy("al-kc-gain") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        g = graph_mod.load()
        lif = LIFParams.from_config()
        for a in [float(x) for x in al.split(",")]:
            base = cp.CellParams.load(Path(paths.runs_dir()) / f"al-kc-x{a:g}" / "params.npz")
            k = list(base.meta["groups"]).index("ALPN-KC")
            a0 = float(base.alpha[k])
            for gn in [float(x) for x in gains.split(",")]:
                base.alpha[k] = min(a0 * gn, 10.0)
                m = measure_kc(g, lif, base, 1, 0, "cpu", cfg)
                rows.append({"eLN_PN": a, "ALPN_KC_x": gn, "ALPN_KC": float(base.alpha[k]), **m})
                print(f"[al-kc-gain] eLN→PN ×{a:g} · PN→KC ×{gn:g}({base.alpha[k]:.2f}) · KC 냄새 넷 "
                      f"{m['kc_fraction_min_odor']:.3f}~{m['kc_fraction_max_odor']:.3f} · 겹침 최대 "
                      f"{m['kc_overlap_max_odor']:.3f} · MBON 최소 냄새 {m['mbon_fire_min_odor']:.3f} · "
                      f"나머지 뇌 {m['pop_rate_hz']:.2f} Hz", flush=True)
            base.alpha[k] = a0
    dst = paths.runs_dir() / f"al-kc-gain-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[al-kc-gain] {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
