"""MBON 침묵 진단 (R-7469 F-007 · M3 뒤) — MBON 마다 입력이 문턱에 얼마나 못 미치나.

M3(`plasticity-probe`)에서 KC 가 4.3 % 켜져도 MBON 97 개가 100 ms 에 8 발화 · 보상 구획 0 이었다.
가소성은 KC → MBON 세기를 **약하게만** 하므로 MBON 이 침묵이면 배울 것이 없다. 그래서 먼저 잰다:

- 그래프(실행 없음): MBON 마다 KC 시냅스 · KC 짝 수 · 다른 흥분 · 억제 시냅스 · KC 입력 비율.
- 짧은 모의(냄새 자극): 창(config `mbon_probe.windows_ms`)마다 MBON 발화 · 막전위 평균 · 최고,
  그리고 **입력 출처별 평균 탈분극** — 발화 하나가 g 에 w 를 더하고 g 는 tau_syn 으로 줄어
  막전위 평균이 g 평균에 수렴하므로(dv/dt = (v0 − v + g)/t_mbr)
  출처 s 의 몫 ≈ Σ w · 발화 수 · tau_syn / 창.
  문턱 간격(v_th − v0)과 견줘 "KC 입력을 몇 배 해야 평균이 문턱에 닿나"를 낸다.

평균장 어림이라 요동(동시 입력)은 빠진다 — 그래서 실측 최고 막전위를 함께 둔다.
손으로 고치지 않고 기록한다.
값(자극 · 창 · 판정 기준)은 config/plasticity.yaml `mbon_probe`(결과 전 선언).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config, paths
from .graph import Graph


def mbon_inputs(graph: Graph, w: np.ndarray, mbon: np.ndarray, cfg: dict[str, Any]
                ) -> dict[str, np.ndarray]:
    """그래프에서 MBON 마다 입력 구성(연결 세기 w 는 엔진 값 — mV · 부호 포함)."""
    cls = graph.columns["class"]
    src = graph.src()
    pos = np.full(graph.n, -1)
    pos[mbon] = np.arange(len(mbon))
    e = np.flatnonzero(pos[graph.dst] >= 0)
    k = pos[graph.dst[e]]
    kc = np.isin(cls[src[e]], cfg["targets"]["pre_class"])
    exc, inh = w[e] > 0, w[e] < 0
    m = len(mbon)

    def tot(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.bincount(k[mask], weights=x[mask], minlength=m)

    syn = graph.count[e].astype(float)
    ones = np.ones(len(e))
    all_syn = tot(np.ones(len(e), bool), syn)
    return {"kc_syn": tot(kc, syn), "kc_partners": tot(kc, ones),
            "other_exc_syn": tot(exc & ~kc, syn), "inh_syn": tot(inh, syn),
            "kc_syn_fraction": np.divide(tot(kc, syn), all_syn, out=np.zeros(m),
                                         where=all_syn > 0),
            "kc_w_all_once_mV": tot(kc, w[e])}


def drive(graph: Graph, w: np.ndarray, cnt: np.ndarray, mbon: np.ndarray,
          cfg: dict[str, Any], window_ms: float, tau_syn_ms: float) -> dict[str, np.ndarray]:
    """창 하나의 발화 수(cnt) → MBON 마다 출처별 평균 탈분극(mV) · 켜진 KC 짝 비율."""
    cls = graph.columns["class"]
    src = graph.src()
    pos = np.full(graph.n, -1)
    pos[mbon] = np.arange(len(mbon))
    e = np.flatnonzero(pos[graph.dst] >= 0)
    k = pos[graph.dst[e]]
    kc = np.isin(cls[src[e]], cfg["targets"]["pre_class"])
    x = w[e] * cnt[src[e]] * tau_syn_ms / window_ms
    m = len(mbon)
    on = (cnt[src[e]] > 0).astype(float)
    partners = np.bincount(k[kc], minlength=m)
    return {"inh_top": top_sources(graph.columns["type"][src[e]], k, x, x < 0, m,
                                   int(cfg["mbon_probe"]["top_sources"])),
            "kc_mV": np.bincount(k[kc], weights=x[kc], minlength=m),
            "other_exc_mV": np.bincount(k[~kc & (x > 0)], weights=x[~kc & (x > 0)], minlength=m),
            "inh_mV": np.bincount(k[x < 0], weights=x[x < 0], minlength=m),
            "kc_partners_on": np.divide(np.bincount(k[kc], weights=on[kc], minlength=m),
                                        partners, out=np.zeros(m), where=partners > 0)}


def top_sources(src_type: np.ndarray, k: np.ndarray, x: np.ndarray, mask: np.ndarray,
                m: int, n_top: int) -> list[list[tuple[str, float]]]:
    """MBON 마다 |몫| 이 큰 출발 종류 n_top 개 — [(종류, mV)]."""
    out: list[list[tuple[str, float]]] = [[] for _ in range(m)]
    sums: dict[tuple[int, str], float] = {}
    for kk, t, v in zip(k[mask], src_type[mask], x[mask]):
        sums[(int(kk), str(t))] = sums.get((int(kk), str(t)), 0.0) + float(v)
    for (kk, t), v in sums.items():
        out[kk].append((t, round(v, 3)))
    return [sorted(o, key=lambda tv: -abs(tv[1]))[:n_top] for o in out]


def kc_types_of(graph: Graph, mbon: np.ndarray, cfg: dict[str, Any],
                n_top: int) -> list[list[tuple[str, float]]]:
    """MBON 마다 KC 시냅스를 가장 많이 주는 KC 종류 n_top 개 — [(종류, 시냅스 비율)]."""
    cls, types = graph.columns["class"], graph.columns["type"]
    src = graph.src()
    pos = np.full(graph.n, -1)
    pos[mbon] = np.arange(len(mbon))
    e = np.flatnonzero((pos[graph.dst] >= 0) & np.isin(cls[src], cfg["targets"]["pre_class"]))
    k = pos[graph.dst[e]]
    syn = graph.count[e].astype(float)
    top = top_sources(types[src[e]], k, syn, np.ones(len(e), bool), len(mbon), n_top)
    tot = np.bincount(k, weights=syn, minlength=len(mbon))
    return [[(t, round(v / tot[i], 3)) for t, v in row] for i, row in enumerate(top)]


def verdict(gap: float, kc: float, exc: float, inh: float, spikes: int,
            cfg: dict[str, Any]) -> str:
    """MBON 하나의 침묵 까닭(결과 전 선언 · config `mbon_probe.verdict`)."""
    if spikes > 0:
        return "fires"
    if kc + exc >= gap and kc + exc + inh < gap:
        return "inhibition"
    if kc <= float(cfg["verdict"]["kc_none_mV"]):
        return "no_kc_input"
    return "weak_kc"


def run_case(graph: Graph, lif, cell, cfg: dict[str, Any], stim: list[str], orn_hz: float,
             seed: int, device: str) -> dict[str, Any]:
    """냄새 자극으로 가장 긴 창만큼 한 번 달리고 창마다 잘라 MBON 을 진단한다."""
    import torch

    from .sim import Engine

    pc = cfg["mbon_probe"]
    mbon = np.flatnonzero(np.isin(graph.columns["class"], cfg["targets"]["post_class"]))
    kc_all = np.isin(graph.columns["class"], cfg["targets"]["pre_class"])
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)
    w = eng.w.detach().cpu().numpy().astype(np.float64)
    v_th = eng.v_th
    vth = (v_th.detach().cpu().numpy()[mbon] if isinstance(v_th, torch.Tensor)
           else np.full(len(mbon), float(v_th)))
    gap = vth - lif.v0_mV
    static = mbon_inputs(graph, w, mbon, cfg)
    kc_main = kc_types_of(graph, mbon, cfg, int(pc["top_sources"]))
    kc_type = graph.columns["type"][kc_all].astype(str)
    eng.set_poisson(graph.select(type=stim), orn_hz)
    eng.record_voltage(mbon)
    windows = [float(x) for x in pc["windows_ms"]]
    t0 = time.time()
    eng.run(max(windows))
    wall = time.time() - t0
    steps, neurons = eng.spikes()
    volt = eng.voltage()
    types = graph.columns["type"][mbon].astype(str)
    out_w = []
    for win in windows:
        m = steps * lif.dt_ms < win
        cnt = np.bincount(neurons[m], minlength=graph.n)
        d = drive(graph, w, cnt, mbon, cfg, win, lif.tau_syn_ms)
        vw = volt[: int(round(win / lif.dt_ms))]
        spk = cnt[mbon]
        mean_v, max_v = vw.mean(0), vw.max(0)
        need = np.divide(gap - d["other_exc_mV"] - d["inh_mV"], d["kc_mV"],
                         out=np.full(len(mbon), np.inf), where=d["kc_mV"] > 0)
        why = [verdict(gap[i], d["kc_mV"][i], d["other_exc_mV"][i], d["inh_mV"][i],
                       int(spk[i]), pc) for i in range(len(mbon))]
        rows = [{"type": types[i], "spikes": int(spk[i]),
                 "v_mean_mV": round(float(mean_v[i]), 3), "v_max_mV": round(float(max_v[i]), 3),
                 "gap_mV": round(float(gap[i]), 3),
                 "kc_mV": round(float(d["kc_mV"][i]), 4),
                 "other_exc_mV": round(float(d["other_exc_mV"][i]), 4),
                 "inh_mV": round(float(d["inh_mV"][i]), 4),
                 "kc_partners_on": round(float(d["kc_partners_on"][i]), 4),
                 "inh_top": d["inh_top"][i],
                 "kc_needed_x": (round(float(need[i]), 2) if np.isfinite(need[i]) else None),
                 "why": why[i]} for i in range(len(mbon))]
        fin = need[np.isfinite(need) & (spk == 0)]
        out_w.append({
            "window_ms": win,
            "kc_active_fraction": round(float((cnt[kc_all] > 0).mean()), 5),
            "kc_active_by_type": {t: round(float((cnt[kc_all][kc_type == t] > 0).mean()), 4)
                                  for t in sorted(set(kc_type))
                                  if (kc_type == t).sum() >= int(pc["kc_type_min"])},
            "mbon_spikes": int(spk.sum()), "mbon_firing": int((spk > 0).sum()),
            "why_counts": {k: why.count(k) for k in sorted(set(why))},
            "kc_needed_x_median": round(float(np.median(fin)), 2) if len(fin) else None,
            "kc_needed_x_min": round(float(fin.min()), 2) if len(fin) else None,
            "v_max_below_th_median_mV": round(float(np.median((vth - max_v)[spk == 0])), 3)
            if (spk == 0).any() else None,
            "mbon": rows})
    return {"mbon_neurons": int(len(mbon)), "wall_s": round(wall, 1),
            "static": {"kc_syn_median": float(np.median(static["kc_syn"])),
                       "kc_partners_median": float(np.median(static["kc_partners"])),
                       "kc_syn_fraction_median": round(float(np.median(
                           static["kc_syn_fraction"])), 4),
                       "no_kc_input": int((static["kc_syn"] == 0).sum()),
                       "kc_w_all_once_mV_median": round(float(np.median(
                           static["kc_w_all_once_mV"])), 3),
                       "per_mbon": [{"type": types[i],
                                     **{k: round(float(v[i]), 4) for k, v in static.items()},
                                     "kc_main_types": kc_main[i]}
                                    for i in range(len(mbon))]},
            "windows": out_w}


def parse_case(spec: str) -> tuple[str, str, str]:
    """`이름=뇌[:회로학습폴더]` → (이름, 뇌, 폴더). 뇌 = real | rewired-s<씨앗>."""
    name, rest = spec.split("=", 1)
    brain, _, cell = rest.partition(":")
    return name, brain, cell


def main(args: argparse.Namespace) -> int:
    from . import cellparams as cp
    from . import graph as graph_mod
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("plasticity")
    pc = cfg["mbon_probe"]
    stim = config.load("fit")["kc_probe"]["stimuli"][pc["stimulus"]]
    cases = [parse_case(s) for s in args.case]
    res: dict[str, Any] = {"rule": cfg["version"], "probe": pc, "stimulus": stim, "cases": {}}
    try:
        with heavy("mbon-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            base = graph_mod.load()
            lif = LIFParams.from_config()
            res["sim"] = lif.version
            for name, brain, cell_dir in cases:
                g = (base if brain == "real"
                     else graph_mod.rewire(base, int(brain.rsplit("-s", 1)[1])))
                cell = cp.CellParams.load(Path(cell_dir) / "params.npz") if cell_dir else None
                r = run_case(g, lif, cell, cfg, stim, float(pc["orn_rate_hz"]), args.seed,
                             args.device)
                res["cases"][name] = {"brain": brain, "cell_from": cell_dir or None, **r}
                for wdw in r["windows"]:
                    print(f"[mbon-probe] {name:12} {wdw['window_ms']:6.0f} ms · KC "
                          f"{wdw['kc_active_fraction']:.4f} · MBON 발화 {wdw['mbon_spikes']}"
                          f" ({wdw['mbon_firing']}/{r['mbon_neurons']}) · 까닭 "
                          f"{wdw['why_counts']} · KC 필요 배수 중앙 {wdw['kc_needed_x_median']}"
                          f" · 최고 막전위의 문턱 아래 중앙 {wdw['v_max_below_th_median_mV']} mV",
                          flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"mbon-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mbon-probe] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("mbon-probe",
                       help="MBON 침묵 진단 — 입력이 문턱에 얼마나 못 미치나(무거움 · 짧음)")
    p.add_argument("--case", action="append", required=True,
                   help="이름=뇌[:회로학습폴더] — 여러 번(예: real-fit=real:runs/fit-…)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.set_defaults(func=main)
