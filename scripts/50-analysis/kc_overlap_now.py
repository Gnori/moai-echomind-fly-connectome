"""학습 냄새끼리 KC 겹침(kc_overlap_max_odor)을 이미 맞춘 매개변수로 잰다 — kc-mbon-c 상한을 정하기 전 출발점.

사용: python scripts/50-analysis/kc_overlap_now.py <뇌>:<회로 학습 폴더|none> ...  → runs/kc-overlap-now-<시각>.json
학습 냄새 = fit.yaml kc-mbon-b 의 kc_probe(식초 + 무작위 3 묶음) · 씨앗 0 · 300 ms(회로 학습 손실과 같은 창).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.cellparams import CellParams
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import measure_kc
from moai_echomind_fly.sim import LIFParams


def main(specs: list[str]) -> None:
    fit = config.load("fit")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-b"]["kc_probe"], "odor_overlap": True}
    rows = []
    with heavy("kc-overlap-now") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        for spec in specs:
            brain, folder = spec.split(":", 1)
            g = base if brain == "real" else graph_mod.rewire(base, int(brain.rsplit("-s", 1)[1]))
            cell = None if folder == "none" else CellParams.load(Path(folder) / "params.npz")
            m = measure_kc(g, LIFParams.from_config(), cell, 1, 0, "cpu", cfg)
            rows.append({"brain": brain, "cell_from": folder, **m})
            print(f"[kc-overlap-now] {brain} {folder} 겹침 최대 {m['kc_overlap_max_odor']:.4f} · "
                  f"KC {m['kc_fraction_min_odor']:.3f}~{m['kc_fraction_max_odor']:.3f}", flush=True)
    out = paths.runs_dir() / f"kc-overlap-now-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"kc_probe": cfg, "rows": rows}, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"[kc-overlap-now] {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
