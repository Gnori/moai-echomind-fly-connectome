"""촉각엽 번짐 작은 시험(10-12 · 가능성 먼저 · 판정 아님) — 흥분성 LN → PN 세기를 줄이면 번짐이 주나.

al_spread 진단: 번져 켜진 PN 흥분 입력의 86 % 가 흥분성(아세틸콜린) LN · 12.5 % 가 PN → PN.
실제 eLN → PN 은 주로 전기 시냅스(노트 16 §11) — 모형은 화학 흥분으로 넣었다.
배율(묶음 α)을 바꿔 냄새 다섯(식초 + 학습 무작위 3 + 가소성 CS−)에서 잰다:
- 강하게 켜진 uPN 비율(≥ 30 Hz · 목표 ≤ 25 % · 노트 16 §11) · 해부 자극 PN 이 켜지나(감도 유지) · 냄새끼리 강한 uPN Jaccard.
사용: python scripts/50-analysis/al_probe.py → runs/al-probe-<시각>.json
"""

from __future__ import annotations

import itertools
import json
import re
import time

import numpy as np

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import extra_odor_sets, stim_pns
from moai_echomind_fly.sim import Engine, LIFParams

UPN_RE = r"^(?!M_|MZ_|CB)[A-Z][A-Za-z0-9]*_[a-z0-9]*PN$"
STRONG_HZ = 30.0
CS_MINUS = ["ORN_DL5", "ORN_DM5", "ORN_DM6", "ORN_VA1v", "ORN_VA3", "ORN_VM6m"]
GRID = [(1.0, 1.0), (0.5, 1.0), (0.25, 1.0), (0.1, 1.0), (0.0, 1.0), (0.25, 0.3), (0.1, 0.3),
        (0.0, 0.3)]                                  # (흥분성 LN → PN, PN → PN 흥분)


def jac(a: np.ndarray, b: np.ndarray) -> float:
    u = len(np.union1d(a, b))
    return len(np.intersect1d(a, b)) / u if u else 0.0


def main() -> None:
    fit = config.load("fit")
    cfg = {**fit["kc_probe"], **fit["scopes"]["kc-mbon-b"]["kc_probe"]}
    win = float(cfg["loss_window_ms"])
    strong = int(round(STRONG_HZ * win / 1000.0))
    rows = []
    with heavy("al-probe") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        g = graph_mod.load()
        lif = LIFParams.from_config()
        cls, typ = g.columns["class"].astype(str), g.columns["type"].astype(str)
        pn = np.flatnonzero(cls == "ALPN")
        upn = np.array([i for i in pn if re.match(UPN_RE, typ[i])])
        ln = cls == "ALLN"
        exc_types = sorted({t for t in typ[ln & (g.sign == 1)]}
                           - {t for t in typ[ln & (g.sign != 1)]})
        exc_pn_types = sorted({t for t in typ[(cls == "ALPN") & (g.sign == 1)]}
                              - {t for t in typ[(cls == "ALPN") & (g.sign != 1)]})
        cell = cp.init(g, ["APL"], lif.t_mbr_ms,          # APL 은 기본값 그대로(빈 목록은 init 이 못 받음)
                       lif.v_th_mV, None, {"probe": "al"},
                       groups=[{"name": "eLN-PN", "src": {"type": exc_types},
                                "dst": {"class": ["ALPN"]}},
                               {"name": "PN-PN", "src": {"type": exc_pn_types},
                                "dst": {"class": ["ALPN"]}}])
        odors = {"vinegar": cfg["stimuli"][cfg["target_stimulus"]]}
        odors.update({f"extra{i}": s for i, s in enumerate(extra_odor_sets(g, cfg))})
        odors["cs_minus"] = CS_MINUS
        print(f"[al-probe] 흥분성 LN 종류 {len(exc_types)} · 시냅스 {cell.meta['group_synapses']}",
              flush=True)
        for a_ln, a_pn in GRID:
            cell.alpha[:] = [a_ln, a_pn]
            per, sets = {}, {}
            for name, st in odors.items():
                orn = g.select(type=st)
                anat = stim_pns(g, orn, cfg)
                eng = Engine(g, lif, device="cpu", seed=0, cell=cell)
                eng.set_poisson(orn, np.full(len(orn), float(cfg["orn_rate_hz"])))
                eng.run(win)
                _, n = eng.spikes()
                cnt = np.bincount(n, minlength=g.n)
                sets[name] = upn[cnt[upn] >= strong]
                per[name] = {"upn_strong_frac": round(float((cnt[upn] >= strong).mean()), 4),
                             "anat_fired_frac": round(float((cnt[anat] > 0).mean()), 4)
                             if len(anat) else None,
                             "anat_strong_frac": round(float((cnt[anat] >= strong).mean()), 4)
                             if len(anat) else None,
                             "pn_fired": int((cnt[pn] > 0).sum())}
            js = [jac(sets[a], sets[b]) for a, b in itertools.combinations(sets, 2)]
            r = {"eLN_PN": a_ln, "PN_PN": a_pn, "odors": per,
                 "upn_strong_mean": round(float(np.mean([v["upn_strong_frac"]
                                                         for v in per.values()])), 4),
                 "upn_strong_max": max(v["upn_strong_frac"] for v in per.values()),
                 "anat_strong_mean": round(float(np.mean([v["anat_strong_frac"]
                                                          for v in per.values()])), 4),
                 "strong_upn_jaccard_mean": round(float(np.mean(js)), 4),
                 "strong_upn_jaccard_max": round(float(max(js)), 4)}
            rows.append(r)
            print(f"[al-probe] eLN→PN ×{a_ln} · PN→PN ×{a_pn} · 강한 uPN 평균 "
                  f"{r['upn_strong_mean']:.3f} 최대 {r['upn_strong_max']:.3f} · 해부 PN 강함 "
                  f"{r['anat_strong_mean']:.3f} · 강한 uPN 겹침 평균 {r['strong_upn_jaccard_mean']:.3f}"
                  f" 최대 {r['strong_upn_jaccard_max']:.3f}", flush=True)
    dst = paths.runs_dir() / f"al-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps({"grid": GRID, "exc_ln_types": exc_types, "odors": odors,
                               "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[al-probe] {dst}")


if __name__ == "__main__":
    main()
