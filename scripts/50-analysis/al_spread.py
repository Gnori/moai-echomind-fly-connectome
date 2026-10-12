"""촉각엽 번짐 진단(10-12 · 가벼움 · 판정 아님) — 냄새에 직접 자극받지 않은 PN 을 누가 켜나.

식초 6 · ORN 150 Hz · 300 ms · 진짜 배선(회로 학습 없음 — 촉각엽은 학습 범위 밖이라 같음).
PN 을 해부 자극 PN(mb_probe.stim_pns) · 번져 켜진 PN 으로 나누고, 번져 켜진 PN 에 들어온 입력을
"보낸 뉴런의 발화 수 × 시냅스 수 × 부호"로 보낸 쪽 갈래(class · 부호)별로 합친다. 첫 발화 시각도 본다.
사용: python scripts/50-analysis/al_spread.py [--odor vinegar6] → runs/al-spread-<시각>.json
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.mb_probe import stim_pns
from moai_echomind_fly.sim import Engine, LIFParams

UPN_RE = r"^(?!M_|MZ_|CB)[A-Z][A-Za-z0-9]*_[a-z0-9]*PN$"   # 사구체 이름_…PN = uPN(이름으로 · 284)
STRONG_HZ = 30.0                                           # 노트 16 §11 — 강하게 켜짐(Seki 2017)


def main(odor: str = "vinegar6") -> None:
    cfg = config.load("fit")["kc_probe"]
    with heavy("al-spread") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        g = graph_mod.load()
        lif = LIFParams.from_config()
        cls, typ = g.columns["class"].astype(str), g.columns["type"].astype(str)
        orn = g.select(type=cfg["stimuli"][odor])
        eng = Engine(g, lif, device="cpu", seed=0)
        eng.set_poisson(orn, np.full(len(orn), float(cfg["orn_rate_hz"])))
        eng.run(float(cfg["loss_window_ms"]))
        steps, neurons = eng.spikes()
    cnt = np.bincount(neurons, minlength=g.n)
    first = np.full(g.n, np.inf)
    np.minimum.at(first, neurons, steps * lif.dt_ms)
    pn = np.flatnonzero(cls == "ALPN")
    upn_re = re.compile(UPN_RE)
    upn = np.array([i for i in pn if upn_re.match(typ[i])])
    strong = int(round(STRONG_HZ * float(cfg["loss_window_ms"]) / 1000.0))
    anat = stim_pns(g, orn, cfg)
    fired = pn[cnt[pn] > 0]
    spread = np.setdiff1d(fired, anat)
    src = g.src()
    into = np.isin(g.dst, spread) & (cnt[src] > 0)
    drive = collections.Counter()
    for k, s, n, c in zip(cls[src[into]], g.sign[src[into]], g.count[into], cnt[src[into]]):
        drive[f"{k or '(빈칸)'} {int(s):+d}"] += float(n) * float(c) * float(s)
    pos = sum(v for v in drive.values() if v > 0)
    active_by_class = collections.Counter(cls[cnt > 0])
    al_cls = ["olfactory", "ALLN", "ALPN", "ALIN", "ALON"]
    out = {"odor": odor, "orn_types": cfg["stimuli"][odor], "pn_total": int(len(pn)),
           "pn_anat": int(len(anat)), "pn_anat_fired": int(np.isin(anat, fired).sum()),
           "pn_fired": int(len(fired)), "pn_spread": int(len(spread)),
           "upn_total": int(len(upn)), "upn_fired": int((cnt[upn] > 0).sum()),
           "upn_strong": int((cnt[upn] >= strong).sum()),
           "upn_strong_frac": round(float((cnt[upn] >= strong).mean()), 4),
           "upn_strong_not_anat": int(((cnt[upn] >= strong) & ~np.isin(upn, anat)).sum()),
           "mpn_fired": int((cnt[np.setdiff1d(pn, upn)] > 0).sum()),
           "mpn_total": int(len(pn) - len(upn)), "strong_spikes": strong,
           "first_spike_ms": {"anat_median": float(np.median(first[anat][np.isfinite(first[anat])]))
                              if np.isfinite(first[anat]).any() else None,
                              "spread_median": float(np.median(first[spread]))
                              if len(spread) else None},
           "active_neurons": {k: int(active_by_class.get(k, 0)) for k in al_cls},
           "neurons": {k: int((cls == k).sum()) for k in al_cls},
           "drive_into_spread_pn": {k: round(v / pos, 4) for k, v in drive.most_common()},
           "aln_active_by_sign": {f"{s:+d}": int(((cls == "ALLN") & (g.sign == s) & (cnt > 0)).sum())
                                  for s in (-1, 0, 1)},
           "aln_total_by_sign": {f"{s:+d}": int(((cls == "ALLN") & (g.sign == s)).sum())
                                 for s in (-1, 0, 1)},
           "orn_other_types_fired": int(((cls == "olfactory") & (cnt > 0)).sum() - np.isin(
               orn, np.flatnonzero(cnt > 0)).sum())}
    for k, v in out.items():
        print(f"[al-spread] {k}: {v}", flush=True)
    dst = paths.runs_dir() / f"al-spread-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[al-spread] {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
